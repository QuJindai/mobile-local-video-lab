package com.qujindai.facere;

/* Similarity alignment follows the MIT android-face-fusion/InsightFace geometry. */

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;

public final class FaceAligner {
    private static final float[][] ARC_112 = {
            {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
            {41.5493f, 92.3655f}, {70.7299f, 92.2041f}
    };
    private static final float[][] SWAP_128 = {
            {46.2946f, 51.6963f}, {81.5318f, 51.5014f}, {64.0252f, 71.7366f},
            {49.5493f, 92.3655f}, {78.7299f, 92.2041f}
    };

    public static final class Alignment {
        public final Bitmap bitmap;
        public final Matrix forward;

        Alignment(Bitmap bitmap, Matrix forward) {
            this.bitmap = bitmap;
            this.forward = forward;
        }
    }

    private FaceAligner() {}

    public static Alignment align112(Bitmap source, float[] landmarks) {
        return align(source, landmarks, 112, ARC_112);
    }

    public static Alignment align128(Bitmap source, float[] landmarks) {
        return align(source, landmarks, 128, SWAP_128);
    }

    private static Alignment align(Bitmap source, float[] landmarks, int size, float[][] reference) {
        if (source == null || source.isRecycled()) throw new IllegalArgumentException("image unavailable");
        if (landmarks == null || landmarks.length != 10) {
            throw new IllegalArgumentException("five landmarks required");
        }
        float[][] points = new float[5][2];
        for (int i = 0; i < 5; i++) {
            points[i][0] = landmarks[i * 2];
            points[i][1] = landmarks[i * 2 + 1];
        }
        float[] affine = SimilarityMath.estimate(points, reference);
        Matrix matrix = new Matrix();
        matrix.setValues(new float[]{
                affine[0], affine[1], affine[2],
                affine[3], affine[4], affine[5],
                0f, 0f, 1f
        });
        Bitmap aligned = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(aligned);
        canvas.drawBitmap(source, matrix, new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG));
        return new Alignment(aligned, matrix);
    }
}
