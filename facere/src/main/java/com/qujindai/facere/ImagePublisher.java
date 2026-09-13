package com.qujindai.facere;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;

import java.io.IOException;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public final class ImagePublisher {
    private ImagePublisher() {}

    public static Uri publishJpeg(Context context, Bitmap bitmap) throws IOException {
        if (bitmap == null || bitmap.isRecycled()) throw new IllegalArgumentException("result bitmap required");
        ContentResolver resolver = context.getContentResolver();
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, "face_re_" + timestamp + ".jpg");
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/FaceRE");
        values.put(MediaStore.Images.Media.WIDTH, bitmap.getWidth());
        values.put(MediaStore.Images.Media.HEIGHT, bitmap.getHeight());
        values.put(MediaStore.Images.Media.IS_PENDING, 1);
        Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        if (uri == null) throw new IOException("MediaStore insert failed");
        boolean success = false;
        try (OutputStream output = resolver.openOutputStream(uri, "w")) {
            if (output == null) throw new IOException("MediaStore output stream unavailable");
            if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 95, output)) {
                throw new IOException("JPEG encode failed");
            }
            success = true;
        } finally {
            if (!success) resolver.delete(uri, null, null);
        }
        ContentValues ready = new ContentValues();
        ready.put(MediaStore.Images.Media.IS_PENDING, 0);
        if (resolver.update(uri, ready, null, null) <= 0) {
            resolver.delete(uri, null, null);
            throw new IOException("MediaStore finalize failed");
        }
        return uri;
    }
}
