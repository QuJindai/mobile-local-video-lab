package com.qujindai.facere;

import android.content.ContentResolver;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.util.Size;

import java.io.IOException;

public final class ImageLoader {
    private ImageLoader() {}

    public static Bitmap load(ContentResolver resolver, Uri uri, int maxDimension) throws IOException {
        if (resolver == null || uri == null) throw new IllegalArgumentException("image uri required");
        ImageDecoder.Source source = ImageDecoder.createSource(resolver, uri);
        return ImageDecoder.decodeBitmap(source, (decoder, info, ignored) -> {
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
            decoder.setMemorySizePolicy(ImageDecoder.MEMORY_POLICY_LOW_RAM);
            Size size = info.getSize();
            int max = Math.max(size.getWidth(), size.getHeight());
            if (maxDimension > 0 && max > maxDimension) {
                float scale = (float) maxDimension / max;
                decoder.setTargetSize(
                        Math.max(1, Math.round(size.getWidth() * scale)),
                        Math.max(1, Math.round(size.getHeight() * scale)));
            }
        });
    }
}
