package com.qujindai.localvideo;

import java.nio.ByteBuffer;

public final class Yuv420PlaneWriter {
    private Yuv420PlaneWriter() {}

    public static void write(byte[] source,
                             int width,
                             int height,
                             ByteBuffer target,
                             int rowStride,
                             int pixelStride) {
        if (source == null || source.length != width * height) {
            throw new IllegalArgumentException("source plane size mismatch");
        }
        if (target == null || width <= 0 || height <= 0 || rowStride <= 0 || pixelStride <= 0) {
            throw new IllegalArgumentException("invalid target plane layout");
        }
        int base = target.position();
        int last = base + (height - 1) * rowStride + (width - 1) * pixelStride;
        if (last >= target.limit()) {
            throw new IllegalArgumentException("target plane is too small for row/pixel stride");
        }
        for (int row = 0; row < height; row++) {
            int sourceRow = row * width;
            int targetRow = base + row * rowStride;
            for (int col = 0; col < width; col++) {
                target.put(targetRow + col * pixelStride, source[sourceRow + col]);
            }
        }
    }
}
