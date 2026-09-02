package com.qujindai.localvideo;

public final class Yuv420Frame {
    public final byte[] y;
    public final byte[] u;
    public final byte[] v;
    public final int width;
    public final int height;

    private Yuv420Frame(byte[] y, byte[] u, byte[] v, int width, int height) {
        this.y = y;
        this.u = u;
        this.v = v;
        this.width = width;
        this.height = height;
    }

    public int byteSize() {
        return y.length + u.length + v.length;
    }

    public static Yuv420Frame fromArgb(int[] argb, int width, int height) {
        if (width <= 0 || height <= 0 || (width & 1) != 0 || (height & 1) != 0) {
            throw new IllegalArgumentException("YUV420 requires positive even dimensions");
        }
        if (argb == null || argb.length != width * height) {
            throw new IllegalArgumentException("ARGB pixel count does not match dimensions");
        }

        byte[] yPlane = new byte[width * height];
        byte[] uPlane = new byte[width * height / 4];
        byte[] vPlane = new byte[width * height / 4];

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int color = argb[row * width + col];
                int r = (color >>> 16) & 0xff;
                int g = (color >>> 8) & 0xff;
                int b = color & 0xff;
                yPlane[row * width + col] = (byte) clamp(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
            }
        }

        int chromaWidth = width / 2;
        for (int row = 0; row < height; row += 2) {
            for (int col = 0; col < width; col += 2) {
                int uSum = 0;
                int vSum = 0;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int color = argb[(row + dy) * width + col + dx];
                        int r = (color >>> 16) & 0xff;
                        int g = (color >>> 8) & 0xff;
                        int b = color & 0xff;
                        uSum += clamp(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128);
                        vSum += clamp(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128);
                    }
                }
                int index = (row / 2) * chromaWidth + col / 2;
                uPlane[index] = (byte) ((uSum + 2) / 4);
                vPlane[index] = (byte) ((vSum + 2) / 4);
            }
        }
        return new Yuv420Frame(yPlane, uPlane, vPlane, width, height);
    }

    private static int clamp(int value) {
        return Math.max(0, Math.min(255, value));
    }
}
