package com.qujindai.localvideo;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;

public final class DepthAnythingEngine {
    private static final int INPUT = 518;
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    public static final class DepthMap {
        public final int width;
        public final int height;
        public final float[] normalized;
        public final long elapsedMs;
        public final float rawLow;
        public final float rawHigh;

        DepthMap(int width, int height, float[] normalized, long elapsedMs,
                 float rawLow, float rawHigh) {
            this.width = width;
            this.height = height;
            this.normalized = normalized;
            this.elapsedMs = elapsedMs;
            this.rawLow = rawLow;
            this.rawHigh = rawHigh;
        }
    }

    private final Context context;

    public DepthAnythingEngine(Context context) {
        this.context = context.getApplicationContext();
    }

    public DepthMap estimate(Bitmap source) throws Exception {
        long started = SystemClock.elapsedRealtime();
        DepthRuntimeBundle runtime = DepthRuntimeBundle.installAndVerify(context);
        Bitmap resized = Bitmap.createScaledBitmap(source, INPUT, INPUT, true);
        try {
            FloatBuffer input = prepareInput(resized);
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions();
                 OrtSession session = env.createSession(runtime.getModel().getAbsolutePath(), options)) {
                String inputName = session.getInputNames().iterator().next();
                try (OnnxTensor tensor = OnnxTensor.createTensor(
                        env, input, new long[] {1, 3, INPUT, INPUT});
                     OrtSession.Result result = session.run(Collections.singletonMap(inputName, tensor))) {
                    if (result.size() == 0) throw new IllegalStateException("Depth Anything returned no outputs");
                    Object value = result.get(0).getValue();
                    float[] raw = new float[INPUT * INPUT];
                    int count = flatten(value, raw, 0);
                    if (count != raw.length) {
                        throw new IllegalStateException(
                                "Depth Anything output size mismatch: " + count + " != " + raw.length);
                    }
                    return normalize(raw, SystemClock.elapsedRealtime() - started);
                }
            }
        } finally {
            if (resized != source && !resized.isRecycled()) resized.recycle();
        }
    }

    private static FloatBuffer prepareInput(Bitmap bitmap) {
        int[] pixels = new int[INPUT * INPUT];
        bitmap.getPixels(pixels, 0, INPUT, 0, 0, INPUT, INPUT);
        FloatBuffer buffer = ByteBuffer.allocateDirect(pixels.length * 3 * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        for (int channel = 0; channel < 3; channel++) {
            int shift = channel == 0 ? 16 : (channel == 1 ? 8 : 0);
            for (int pixel : pixels) {
                float value = ((pixel >> shift) & 0xff) / 255f;
                buffer.put((value - MEAN[channel]) / STD[channel]);
            }
        }
        buffer.rewind();
        return buffer;
    }

    private static int flatten(Object value, float[] target, int offset) {
        if (value == null) return offset;
        if (value instanceof float[]) {
            float[] array = (float[]) value;
            int remaining = target.length - offset;
            int count = Math.min(remaining, array.length);
            System.arraycopy(array, 0, target, offset, count);
            return offset + count;
        }
        Class<?> type = value.getClass();
        if (!type.isArray()) {
            if (value instanceof Number && offset < target.length) {
                target[offset++] = ((Number) value).floatValue();
            }
            return offset;
        }
        int length = Array.getLength(value);
        for (int i = 0; i < length && offset < target.length; i++) {
            offset = flatten(Array.get(value, i), target, offset);
        }
        return offset;
    }

    private static DepthMap normalize(float[] raw, long elapsedMs) {
        float[] finite = new float[raw.length];
        int finiteCount = 0;
        for (float value : raw) {
            if (Float.isFinite(value)) finite[finiteCount++] = value;
        }
        if (finiteCount < raw.length / 2) {
            throw new IllegalStateException("Depth Anything produced too many non-finite values");
        }
        finite = Arrays.copyOf(finite, finiteCount);
        Arrays.sort(finite);
        float low = finite[Math.max(0, Math.min(finiteCount - 1, (int) (finiteCount * 0.02f)))];
        float high = finite[Math.max(0, Math.min(finiteCount - 1, (int) (finiteCount * 0.98f)))];
        float span = high - low;
        if (!Float.isFinite(span) || span < 1e-6f) {
            throw new IllegalStateException("Depth Anything produced a flat depth map");
        }
        float[] normalized = new float[raw.length];
        for (int i = 0; i < raw.length; i++) {
            float value = Float.isFinite(raw[i]) ? raw[i] : low;
            normalized[i] = Math.max(0f, Math.min(1f, (value - low) / span));
        }
        return new DepthMap(INPUT, INPUT, normalized, elapsedMs, low, high);
    }
}
