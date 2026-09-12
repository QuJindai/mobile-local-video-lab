package com.qujindai.localvideo;

import android.graphics.Bitmap;

public final class DepthMotionEndpoint {
    private DepthMotionEndpoint() {}

    public static Bitmap create(
            Bitmap source,
            DepthAnythingEngine.DepthMap depth,
            DepthMotionSpec.Preset preset) {
        int width = source.getWidth();
        int height = source.getHeight();
        int[] input = new int[width * height];
        int[] output = new int[input.length];
        source.getPixels(input, 0, width, 0, 0, width, height);

        float cx = (width - 1) * 0.5f;
        float cy = (height - 1) * 0.5f;
        for (int y = 0; y < height; y++) {
            float depthY = height <= 1 ? 0f : y * (depth.height - 1f) / (height - 1f);
            for (int x = 0; x < width; x++) {
                float depthX = width <= 1 ? 0f : x * (depth.width - 1f) / (width - 1f);
                float d = sampleDepth(depth, depthX, depthY);
                float shiftX = DepthMotionSpec.displayShiftX(preset, d, width);
                float shiftY = DepthMotionSpec.displayShiftY(preset, d, height);
                float zoom = DepthMotionSpec.zoomScale(preset, d);

                float sx = cx + (x - shiftX - cx) / zoom;
                float sy = cy + (y - shiftY - cy) / zoom;
                output[y * width + x] = bilinearColor(input, width, height, sx, sy);
            }
        }

        Bitmap endpoint = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        endpoint.setPixels(output, 0, width, 0, 0, width, height);
        return endpoint;
    }

    private static float sampleDepth(DepthAnythingEngine.DepthMap map, float x, float y) {
        int x0 = clamp((int) Math.floor(x), 0, map.width - 1);
        int y0 = clamp((int) Math.floor(y), 0, map.height - 1);
        int x1 = Math.min(map.width - 1, x0 + 1);
        int y1 = Math.min(map.height - 1, y0 + 1);
        float fx = x - x0;
        float fy = y - y0;
        float a = lerp(map.normalized[y0 * map.width + x0], map.normalized[y0 * map.width + x1], fx);
        float b = lerp(map.normalized[y1 * map.width + x0], map.normalized[y1 * map.width + x1], fx);
        return lerp(a, b, fy);
    }

    private static int bilinearColor(int[] pixels, int width, int height, float x, float y) {
        x = Math.max(0f, Math.min(width - 1f, x));
        y = Math.max(0f, Math.min(height - 1f, y));
        int x0 = (int) x;
        int y0 = (int) y;
        int x1 = Math.min(width - 1, x0 + 1);
        int y1 = Math.min(height - 1, y0 + 1);
        float fx = x - x0;
        float fy = y - y0;
        int c00 = pixels[y0 * width + x0];
        int c10 = pixels[y0 * width + x1];
        int c01 = pixels[y1 * width + x0];
        int c11 = pixels[y1 * width + x1];
        int a = interpolateChannel(c00, c10, c01, c11, 24, fx, fy);
        int r = interpolateChannel(c00, c10, c01, c11, 16, fx, fy);
        int g = interpolateChannel(c00, c10, c01, c11, 8, fx, fy);
        int b = interpolateChannel(c00, c10, c01, c11, 0, fx, fy);
        return (a << 24) | (r << 16) | (g << 8) | b;
    }

    private static int interpolateChannel(
            int c00, int c10, int c01, int c11, int shift, float fx, float fy) {
        float top = lerp((c00 >> shift) & 0xff, (c10 >> shift) & 0xff, fx);
        float bottom = lerp((c01 >> shift) & 0xff, (c11 >> shift) & 0xff, fx);
        return clamp(Math.round(lerp(top, bottom, fy)), 0, 255);
    }

    private static float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }

    private static int clamp(int value, int low, int high) {
        return Math.max(low, Math.min(high, value));
    }
}
