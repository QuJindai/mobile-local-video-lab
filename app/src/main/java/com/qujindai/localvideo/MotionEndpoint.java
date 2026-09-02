package com.qujindai.localvideo;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;

public final class MotionEndpoint {
    private MotionEndpoint() {}

    public static Bitmap create(Bitmap source) {
        return create(source, MotionSpec.Preset.CINEMATIC_AUTO);
    }

    public static Bitmap create(Bitmap source, MotionSpec.Preset preset) {
        int width = source.getWidth();
        int height = source.getHeight();
        Bitmap endpoint = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(endpoint);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);

        MotionSpec spec = MotionSpec.forPreset(preset);
        Matrix matrix = new Matrix();
        matrix.postScale(spec.scale, spec.scale, width / 2f, height / 2f);
        if (spec.rotationDegrees != 0f) {
            matrix.postRotate(spec.rotationDegrees, width / 2f, height / 2f);
        }
        matrix.postTranslate(width * spec.translateXRatio, height * spec.translateYRatio);
        canvas.drawBitmap(source, matrix, paint);
        return endpoint;
    }
}
