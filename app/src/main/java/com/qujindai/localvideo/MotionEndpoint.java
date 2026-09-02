package com.qujindai.localvideo;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;

public final class MotionEndpoint {
    private MotionEndpoint() {}

    public static Bitmap create(Bitmap source) {
        int width = source.getWidth();
        int height = source.getHeight();
        Bitmap endpoint = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(endpoint);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);

        float scale = 1.055f;
        float translateX = -width * 0.018f;
        float translateY = -height * 0.012f;
        Matrix matrix = new Matrix();
        matrix.postScale(scale, scale, width / 2f, height / 2f);
        matrix.postTranslate(translateX, translateY);
        canvas.drawBitmap(source, matrix, paint);
        return endpoint;
    }
}
