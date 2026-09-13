package com.qujindai.facere;

/*
 * SCRFD post-processing is adapted from Parasaran-Python/android-face-fusion
 * (MIT License, Copyright (c) 2026 Parasaran Vedanarayanan).
 */

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public final class FaceDetector implements AutoCloseable {
    private static final int INPUT_SIZE = 640;
    private static final int[] STRIDES = {8, 16, 32};

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;

    public FaceDetector(FaceModelFiles models) throws OrtException {
        session = OrtSessions.create(models.detector);
    }

    public List<DetectedFace> detect(Bitmap source) throws OrtException {
        if (source == null || source.isRecycled()) {
            throw new IllegalArgumentException("source bitmap unavailable");
        }
        float scale = Math.min((float) INPUT_SIZE / source.getWidth(),
                (float) INPUT_SIZE / source.getHeight());
        int scaledWidth = Math.max(1, Math.round(source.getWidth() * scale));
        int scaledHeight = Math.max(1, Math.round(source.getHeight() * scale));
        Bitmap padded = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Bitmap resized = Bitmap.createScaledBitmap(source, scaledWidth, scaledHeight, true);
        try {
            Canvas canvas = new Canvas(padded);
            canvas.drawColor(Color.BLACK);
            canvas.drawBitmap(resized, 0f, 0f, new Paint(Paint.FILTER_BITMAP_FLAG));
            float[] input = preprocess(padded);
            String inputName = session.getInputNames().iterator().next();
            try (OnnxTensor tensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(input), new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
                 OrtSession.Result result = session.run(Collections.singletonMap(inputName, tensor))) {
                if (result.size() != 9) {
                    throw new OrtException("Face-RE requires landmark SCRFD with 9 outputs; got "
                            + result.size());
                }
                List<DetectedFace> faces = parse(result, scale, source.getWidth(), source.getHeight(), 0.5f);
                if (faces.isEmpty()) {
                    faces = parse(result, scale, source.getWidth(), source.getHeight(), 0.3f);
                }
                return nms(faces, 0.4f);
            }
        } finally {
            if (resized != source && !resized.isRecycled()) resized.recycle();
            padded.recycle();
        }
    }

    private static float[] preprocess(Bitmap bitmap) {
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        int plane = pixels.length;
        float[] data = new float[plane * 3];
        for (int i = 0; i < plane; i++) {
            int p = pixels[i];
            data[i] = (((p >> 16) & 0xff) - 127.5f) / 128.0f;
            data[plane + i] = (((p >> 8) & 0xff) - 127.5f) / 128.0f;
            data[plane * 2 + i] = ((p & 0xff) - 127.5f) / 128.0f;
        }
        return data;
    }

    private static List<DetectedFace> parse(
            OrtSession.Result result, float scale, int originalWidth, int originalHeight,
            float threshold) throws OrtException {
        ArrayList<DetectedFace> faces = new ArrayList<>();
        for (int level = 0; level < STRIDES.length; level++) {
            Object scoreValue = result.get(level).getValue();
            Object boxValue = result.get(level + 3).getValue();
            Object landmarkValue = result.get(level + 6).getValue();
            if (!(scoreValue instanceof float[][])
                    || !(boxValue instanceof float[][])
                    || !(landmarkValue instanceof float[][])) {
                throw new OrtException("unexpected SCRFD tensor type at level " + level);
            }
            float[][] scores = (float[][]) scoreValue;
            float[][] boxes = (float[][]) boxValue;
            float[][] landmarks = (float[][]) landmarkValue;
            int stride = STRIDES[level];
            int featWidth = INPUT_SIZE / stride;
            int featHeight = INPUT_SIZE / stride;
            int gridPositions = featWidth * featHeight;
            int anchorsPerPosition = Math.max(1, scores.length / gridPositions);
            if (boxes.length != scores.length || landmarks.length != scores.length) {
                throw new OrtException("SCRFD output lengths disagree at stride " + stride);
            }

            for (int i = 0; i < scores.length; i++) {
                if (scores[i].length < 1 || boxes[i].length < 4 || landmarks[i].length < 10) {
                    throw new OrtException("SCRFD row shape mismatch at stride " + stride);
                }
                float confidence = scores[i][0];
                if (confidence < threshold) continue;
                int gridIndex = i / anchorsPerPosition;
                int ay = gridIndex / featWidth;
                int ax = gridIndex % featWidth;
                float centerX = ax * stride;
                float centerY = ay * stride;

                float x1 = (centerX - boxes[i][0] * stride) / scale;
                float y1 = (centerY - boxes[i][1] * stride) / scale;
                float x2 = (centerX + boxes[i][2] * stride) / scale;
                float y2 = (centerY + boxes[i][3] * stride) / scale;
                x1 = clamp(x1, 0f, originalWidth);
                y1 = clamp(y1, 0f, originalHeight);
                x2 = clamp(x2, 0f, originalWidth);
                y2 = clamp(y2, 0f, originalHeight);
                float width = x2 - x1;
                float height = y2 - y1;
                if (width < 20f || height < 20f) continue;
                float ratio = width / height;
                if (ratio < 0.3f || ratio > 3f) continue;

                float[] points = new float[10];
                for (int p = 0; p < 5; p++) {
                    points[p * 2] = clamp(
                            (centerX + landmarks[i][p * 2] * stride) / scale,
                            0f, originalWidth);
                    points[p * 2 + 1] = clamp(
                            (centerY + landmarks[i][p * 2 + 1] * stride) / scale,
                            0f, originalHeight);
                }
                faces.add(new DetectedFace(new RectF(x1, y1, x2, y2), points, confidence));
            }
        }
        return faces;
    }

    private static List<DetectedFace> nms(List<DetectedFace> input, float threshold) {
        if (input.isEmpty()) return input;
        ArrayList<DetectedFace> sorted = new ArrayList<>(input);
        sorted.sort(Comparator.comparingDouble((DetectedFace f) -> f.score).reversed());
        ArrayList<DetectedFace> result = new ArrayList<>();
        boolean[] suppressed = new boolean[sorted.size()];
        for (int i = 0; i < sorted.size(); i++) {
            if (suppressed[i]) continue;
            DetectedFace keep = sorted.get(i);
            result.add(keep);
            for (int j = i + 1; j < sorted.size(); j++) {
                if (!suppressed[j] && iou(keep.bbox, sorted.get(j).bbox) > threshold) {
                    suppressed[j] = true;
                }
            }
        }
        return result;
    }

    private static float iou(RectF a, RectF b) {
        float left = Math.max(a.left, b.left);
        float top = Math.max(a.top, b.top);
        float right = Math.min(a.right, b.right);
        float bottom = Math.min(a.bottom, b.bottom);
        float intersection = Math.max(0f, right - left) * Math.max(0f, bottom - top);
        float union = Math.max(0f, a.width()) * Math.max(0f, a.height())
                + Math.max(0f, b.width()) * Math.max(0f, b.height()) - intersection;
        return union <= 0f ? 0f : intersection / union;
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
