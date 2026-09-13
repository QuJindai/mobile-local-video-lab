package com.qujindai.facere;

/*
 * INSwapper preprocessing/input resolution is adapted from
 * Parasaran-Python/android-face-fusion (MIT License,
 * Copyright (c) 2026 Parasaran Vedanarayanan).
 */

import android.graphics.Bitmap;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

public final class FaceSwapper implements AutoCloseable {
    private static final int SIZE = 128;
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;
    private final float[][] emap;
    private final String imageInput;
    private final String embeddingInput;

    public FaceSwapper(FaceModelFiles models) throws IOException, OrtException {
        FaceModelStore.validateEmap(models.emap);
        emap = loadEmap(models);
        session = OrtSessions.create(models.swapper);
        String image = null;
        String embedding = null;
        for (Map.Entry<String, NodeInfo> entry : session.getInputInfo().entrySet()) {
            if (!(entry.getValue().getInfo() instanceof TensorInfo)) continue;
            long[] shape = ((TensorInfo) entry.getValue().getInfo()).getShape();
            if (shape.length == 4) image = entry.getKey();
            if (shape.length == 2) embedding = entry.getKey();
        }
        if (image == null || embedding == null) {
            session.close();
            throw new IllegalStateException("cannot resolve INSwapper image/embedding inputs");
        }
        imageInput = image;
        embeddingInput = embedding;
    }

    public Bitmap swap(Bitmap alignedTarget, float[] sourceEmbedding) throws OrtException {
        if (alignedTarget == null || alignedTarget.getWidth() != SIZE || alignedTarget.getHeight() != SIZE) {
            throw new IllegalArgumentException("INSwapper target must be 128x128");
        }
        if (sourceEmbedding == null || sourceEmbedding.length != 512) {
            throw new IllegalArgumentException("source embedding must be 512D");
        }
        float[] mapped = VectorMath.mapAndNormalize(sourceEmbedding, emap);
        float[] imageData = preprocess(alignedTarget);
        try (OnnxTensor imageTensor = OnnxTensor.createTensor(
                     env, FloatBuffer.wrap(imageData), new long[]{1, 3, SIZE, SIZE});
             OnnxTensor embeddingTensor = OnnxTensor.createTensor(
                     env, FloatBuffer.wrap(mapped), new long[]{1, mapped.length})) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(imageInput, imageTensor);
            inputs.put(embeddingInput, embeddingTensor);
            try (OrtSession.Result result = session.run(inputs)) {
                Object value = result.get(0).getValue();
                if (!(value instanceof float[][][][])) {
                    throw new IllegalStateException("unexpected INSwapper output type");
                }
                float[][][][] batch = (float[][][][]) value;
                if (batch.length != 1 || batch[0].length != 3
                        || batch[0][0].length != SIZE || batch[0][0][0].length != SIZE) {
                    throw new IllegalStateException("unexpected INSwapper output shape");
                }
                return toBitmap(batch[0]);
            }
        }
    }

    private static float[][] loadEmap(FaceModelFiles models) throws IOException {
        float[][] matrix = new float[512][512];
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(models.emap)))) {
            byte[] header = new byte[8];
            in.readFully(header);
            ByteBuffer hb = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
            int rows = hb.getInt();
            int cols = hb.getInt();
            if (rows != 512 || cols != 512) throw new IOException("invalid EMAP shape");
            byte[] rowBytes = new byte[512 * 4];
            for (int r = 0; r < 512; r++) {
                in.readFully(rowBytes);
                ByteBuffer rb = ByteBuffer.wrap(rowBytes).order(ByteOrder.LITTLE_ENDIAN);
                for (int c = 0; c < 512; c++) matrix[r][c] = rb.getFloat();
            }
        }
        return matrix;
    }

    private static float[] preprocess(Bitmap bitmap) {
        int[] pixels = new int[SIZE * SIZE];
        bitmap.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE);
        int plane = pixels.length;
        float[] output = new float[plane * 3];
        for (int i = 0; i < plane; i++) {
            int p = pixels[i];
            output[i] = ((p >> 16) & 0xff) / 255f;
            output[plane + i] = ((p >> 8) & 0xff) / 255f;
            output[plane * 2 + i] = (p & 0xff) / 255f;
        }
        return output;
    }

    private static Bitmap toBitmap(float[][][] data) {
        int[] pixels = new int[SIZE * SIZE];
        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                int r = channel(data[0][y][x]);
                int g = channel(data[1][y][x]);
                int b = channel(data[2][y][x]);
                pixels[y * SIZE + x] = 0xff000000 | (r << 16) | (g << 8) | b;
            }
        }
        Bitmap bitmap = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888);
        bitmap.setPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE);
        return bitmap;
    }

    private static int channel(float value) {
        return Math.max(0, Math.min(255, Math.round(value * 255f)));
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
