package com.qujindai.localvideo;

public final class CheckpointDownloadPolicy {
    private CheckpointDownloadPolicy() {
    }

    public static long resumeOffset(long existingBytes, long expectedBytes) {
        if (existingBytes <= 0L || expectedBytes <= 0L || existingBytes >= expectedBytes) {
            return 0L;
        }
        return existingBytes;
    }

    public static int percent(long downloadedBytes, long totalBytes) {
        if (downloadedBytes <= 0L || totalBytes <= 0L) return 0;
        long value = downloadedBytes * 100L / totalBytes;
        if (value < 0L) value = 0L;
        if (value > 100L) value = 100L;
        return (int) value;
    }
}
