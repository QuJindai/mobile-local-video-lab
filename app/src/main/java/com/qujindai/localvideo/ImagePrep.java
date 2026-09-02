package com.qujindai.localvideo;

import android.content.ContentResolver;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.util.Size;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public final class ImagePrep {
    private ImagePrep() {}

    public static Bitmap loadScaled(ContentResolver resolver, Uri uri) throws IOException {
        ImageDecoder.Source source = ImageDecoder.createSource(resolver, uri);
        Bitmap decoded = ImageDecoder.decodeBitmap(source, (decoder, info, src) -> {
            Size size = info.getSize();
            int[] target = alignedSize(size.getWidth(), size.getHeight());
            decoder.setTargetSize(target[0], target[1]);
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
            decoder.setMemorySizePolicy(ImageDecoder.MEMORY_POLICY_LOW_RAM);
        });

        if (decoded.getConfig() != Bitmap.Config.ARGB_8888) {
            Bitmap converted = decoded.copy(Bitmap.Config.ARGB_8888, false);
            decoded.recycle();
            return converted;
        }
        return decoded;
    }

    static int[] alignedSize(int sourceWidth, int sourceHeight) {
        if (sourceWidth <= 0 || sourceHeight <= 0) {
            throw new IllegalArgumentException("invalid source dimensions");
        }
        double scale = Math.min(1.0, GenerationPlan.MAX_LONG_EDGE /
                (double) Math.max(sourceWidth, sourceHeight));
        int width = align16((int) Math.round(sourceWidth * scale));
        int height = align16((int) Math.round(sourceHeight * scale));
        width = Math.max(64, Math.min(GenerationPlan.MAX_LONG_EDGE, width));
        height = Math.max(64, Math.min(GenerationPlan.MAX_LONG_EDGE, height));
        return new int[] { width, height };
    }

    private static int align16(int value) {
        int aligned = (value / 16) * 16;
        return Math.max(16, aligned);
    }

    public static void writePng(Bitmap bitmap, File file) throws IOException {
        File parent = file.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("cannot create directory: " + parent);
        }
        try (FileOutputStream out = new FileOutputStream(file)) {
            if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)) {
                throw new IOException("PNG compression failed");
            }
        }
    }
}
