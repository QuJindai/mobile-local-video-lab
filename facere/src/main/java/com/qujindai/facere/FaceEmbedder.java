package com.qujindai.facere;

/* ArcFace preprocessing is adapted from Parasaran-Python/android-face-fusion (MIT). */

import android.graphics.Bitmap;

import java.nio.FloatBuffer;
import java.util.Collections;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public final class FaceEmbedder implements AutoCloseable {
    private static final int SIZE = 112;
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;

    public FaceEmbedder(FaceModelFiles models) throws OrtException {
        session = OrtSessions.create(models.recognizer);
    }

    public float[] embed(Bitmap aligned) throws OrtException {
        if (aligned.getWidth() != SIZE || aligned.getHeight() != SIZE) {
            throw new IllegalArgumentException("ArcFace input must be 112x112");
        }
        int[] pixels = new int[SIZE * SIZE];
        aligned.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE);
        int plane = pixels.length;
        float[] data = new float[plane * 3];
        for (int i = 0; i < plane; i++) {
            int p = pixels[i];
            data[i] = (((p >> 16) & 0xff) - 127.5f) / 127.5f;
            data[plane + i] = (((p >> 8) & 0xff) - 127.5f) / 127.5f;
            data[plane * 2 + i] = ((p & 0xff) - 127.5f) / 127.5f;
        }
        String inputName = session.getInputNames().iterator().next();
        try (OnnxTensor tensor = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(data), new long[]{1, 3, SIZE, SIZE});
             OrtSession.Result result = session.run(Collections.singletonMap(inputName, tensor))) {
            Object value = result.get(0).getValue();
            float[] embedding;
            if (value instanceof float[][] && ((float[][]) value).length == 1) {
                embedding = ((float[][]) value)[0];
            } else if (value instanceof float[]) {
                embedding = (float[]) value;
            } else {
                throw new IllegalStateException("unexpected ArcFace output type");
            }
            if (embedding.length != 512) {
                throw new IllegalStateException("ArcFace output must be 512D; got " + embedding.length);
            }
            return VectorMath.normalize(embedding);
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
