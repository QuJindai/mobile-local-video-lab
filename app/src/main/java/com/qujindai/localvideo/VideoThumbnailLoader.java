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
            int outWidth = Math.max(1, targetWidth);
            int outHeight = Math.max(1, targetHeight);
            try {
                int sourceWidth = Integer.parseInt(retriever.extractMetadata(
                        MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH));
                int sourceHeight = Integer.parseInt(retriever.extractMetadata(
                        MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT));
                if (sourceWidth > 0 && sourceHeight > 0) {
                    float scale = Math.min(outWidth / (float) sourceWidth,
                            outHeight / (float) sourceHeight);
                    outWidth = Math.max(1, Math.round(sourceWidth * scale));
                    outHeight = Math.max(1, Math.round(sourceHeight * scale));
                }
            } catch (RuntimeException ignored) {
                // Metadata is optional; frame extraction still has a safe target size.
            }

            Bitmap bitmap = retriever.getScaledFrameAtTime(
                    0L,
                    MediaMetadataRetriever.OPTION_CLOSEST_SYNC,
                    outWidth,
                    outHeight);
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
