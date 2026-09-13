package com.qujindai.facere;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;

public final class FaceBlender {
    private static final int FACE_SIZE = 128;

    private FaceBlender() {}

    public static Bitmap blend(Bitmap target, Bitmap swapped, Matrix targetToAligned) {
        if (target == null || swapped == null || targetToAligned == null) {
            throw new IllegalArgumentException("blend inputs required");
        }
        Matrix alignedToTarget = new Matrix();
        if (!targetToAligned.invert(alignedToTarget)) {
            throw new IllegalArgumentException("face alignment transform is not invertible");
        }
        int width = target.getWidth();
        int height = target.getHeight();
        Bitmap warpedFace = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Bitmap alignedMask = createMask();
        Bitmap warpedMask = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        try {
            Paint filter = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
            new Canvas(warpedFace).drawBitmap(swapped, alignedToTarget, filter);
            new Canvas(warpedMask).drawBitmap(alignedMask, alignedToTarget, filter);

            int count = width * height;
            int[] base = new int[count];
            int[] face = new int[count];
            int[] mask = new int[count];
            target.getPixels(base, 0, width, 0, 0, width, height);
            warpedFace.getPixels(face, 0, width, 0, 0, width, height);
            warpedMask.getPixels(mask, 0, width, 0, 0, width, height);
            int[] output = new int[count];
            for (int i = 0; i < count; i++) {
                float alpha = ((mask[i] >>> 24) & 0xff) / 255f;
                if (alpha <= 0f) {
                    output[i] = base[i];
                    continue;
                }
                int br = (base[i] >> 16) & 0xff;
                int bg = (base[i] >> 8) & 0xff;
                int bb = base[i] & 0xff;
                int fr = (face[i] >> 16) & 0xff;
                int fg = (face[i] >> 8) & 0xff;
                int fb = face[i] & 0xff;
                int r = Math.round(br + (fr - br) * alpha);
                int g = Math.round(bg + (fg - bg) * alpha);
                int b = Math.round(bb + (fb - bb) * alpha);
                output[i] = 0xff000000 | (r << 16) | (g << 8) | b;
            }
            Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            result.setPixels(output, 0, width, 0, 0, width, height);
            return result;
        } finally {
            warpedFace.recycle();
            warpedMask.recycle();
            alignedMask.recycle();
        }
    }

    private static Bitmap createMask() {
        int[] pixels = new int[FACE_SIZE * FACE_SIZE];
        float cx = 64f;
        float cy = 67f;
        float rx = 50f;
        float ry = 56f;
        for (int y = 0; y < FACE_SIZE; y++) {
            for (int x = 0; x < FACE_SIZE; x++) {
                float dx = (x - cx) / rx;
                float dy = (y - cy) / ry;
                float radius = (float) Math.sqrt(dx * dx + dy * dy);
                float alpha;
                if (radius <= 0.72f) alpha = 1f;
                else if (radius >= 1f) alpha = 0f;
                else {
                    float t = (1f - radius) / 0.28f;
                    alpha = t * t * (3f - 2f * t);
                }
                int a = Math.max(0, Math.min(255, Math.round(alpha * 255f)));
                pixels[y * FACE_SIZE + x] = (a << 24) | 0x00ffffff;
            }
        }
        Bitmap mask = Bitmap.createBitmap(FACE_SIZE, FACE_SIZE, Bitmap.Config.ARGB_8888);
        mask.setPixels(pixels, 0, FACE_SIZE, 0, 0, FACE_SIZE, FACE_SIZE);
        return mask;
    }
}
