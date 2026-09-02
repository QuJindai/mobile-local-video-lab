package com.qujindai.localvideo;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;

import java.io.IOException;

public final class VideoThumbnailLoader {
    private VideoThumbnailLoader() {}

    public static Bitmap load(Context context, Uri uri, int targetWidth, int targetHeight)
            throws IOException {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        try {
            retriever.setDataSource(context, uri);
            Bitmap bitmap = retriever.getScaledFrameAtTime(
                    0L,
                    MediaMetadataRetriever.OPTION_CLOSEST_SYNC,
                    Math.max(1, targetWidth),
                    Math.max(1, targetHeight));
            if (bitmap == null) {
                bitmap = retriever.getFrameAtTime(0L, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
            }
            if (bitmap == null) throw new IOException("cannot decode video thumbnail");
            return bitmap;
        } catch (RuntimeException error) {
            throw new IOException("thumbnail decode failed: " + error.getMessage(), error);
        } finally {
            try { retriever.release(); } catch (RuntimeException ignored) {}
        }
    }
}
