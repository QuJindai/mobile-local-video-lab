package com.qujindai.localvideo;

import android.content.Context;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Locale;

public final class MobileI2VCheckpointDownloader {
    public interface Listener {
        void onProgress(int percent, long downloadedBytes, long totalBytes, long bytesPerSecond, String message);
    }

    public static final class Result {
        public final File file;
        public final String sha256;
        public final MobileI2VDownloadSource source;
        public final boolean fromCache;

        Result(File file, String sha256, MobileI2VDownloadSource source, boolean fromCache) {
            this.file = file;
            this.sha256 = sha256;
            this.source = source;
            this.fromCache = fromCache;
        }
    }

    public static final class CancelledException extends IOException {
        CancelledException() {
            super("download cancelled");
        }
    }

    private volatile boolean cancelled;

    public void cancel() {
        cancelled = true;
    }

    public Result download(Context context, MobileI2VDownloadSource source, Listener listener) throws Exception {
        cancelled = false;
        File dir = checkpointDir(context);
        if (!dir.isDirectory() && !dir.mkdirs()) {
            throw new IOException("cannot create checkpoint directory: " + dir);
        }
        File target = checkpointFile(context);
        File marker = verifiedMarker(context);
        if (isVerified(context)) {
            if (listener != null) {
                listener.onProgress(100, source.expectedBytes, source.expectedBytes, 0L,
                        "已存在并通过 SHA-256 校验");
            }
            return new Result(target, source.expectedSha256, source, true);
        }
        if (target.exists()) {
            target.delete();
            marker.delete();
        }

        File partial = partialFile(context);
        long offset = CheckpointDownloadPolicy.resumeOffset(
                partial.isFile() ? partial.length() : 0L,
                source.expectedBytes);
        if (offset == 0L && partial.exists()) {
            partial.delete();
        }

        long startedAt = System.currentTimeMillis();
        long startedBytes = offset;
        long downloaded = downloadToPartial(source, partial, offset, listener, startedAt, startedBytes);
        if (downloaded != source.expectedBytes || partial.length() != source.expectedBytes) {
            throw new IOException("checkpoint size mismatch: " + partial.length()
                    + " != " + source.expectedBytes);
        }
        if (cancelled) throw new CancelledException();

        if (listener != null) {
            listener.onProgress(100, downloaded, source.expectedBytes, 0L, "正在校验 SHA-256…");
        }
        String actualSha = sha256(partial);
        if (!source.expectedSha256.equalsIgnoreCase(actualSha)) {
            partial.delete();
            marker.delete();
            throw new IOException("checkpoint SHA-256 mismatch: " + actualSha);
        }

        if (target.exists() && !target.delete()) {
            throw new IOException("cannot replace old checkpoint");
        }
        if (!partial.renameTo(target)) {
            copyFile(partial, target);
            if (!partial.delete()) partial.deleteOnExit();
        }
        writeMarker(marker, source.expectedSha256);
        if (listener != null) {
            listener.onProgress(100, source.expectedBytes, source.expectedBytes, 0L,
                    "下载完成 · SHA-256 PASS");
        }
        return new Result(target, actualSha, source, false);
    }

    private long downloadToPartial(
            MobileI2VDownloadSource source,
            File partial,
            long requestedOffset,
            Listener listener,
            long startedAt,
            long startedBytes) throws Exception {
        HttpURLConnection connection = null;
        try {
            connection = (HttpURLConnection) new URL(source.downloadUrl()).openConnection();
            connection.setConnectTimeout(30_000);
            connection.setReadTimeout(60_000);
            connection.setInstanceFollowRedirects(true);
            connection.setRequestProperty("User-Agent", "LocalVideoLab/0.6 Android");
            connection.setRequestProperty("Accept-Encoding", "identity");
            if (requestedOffset > 0L) {
                connection.setRequestProperty("Range", "bytes=" + requestedOffset + "-");
            }
            int code = connection.getResponseCode();
            boolean append = requestedOffset > 0L && code == HttpURLConnection.HTTP_PARTIAL;
            if (code < 200 || code >= 300) {
                throw new IOException("HTTP " + code + " from " + source.label);
            }
            long offset = append ? requestedOffset : 0L;
            if (!append && requestedOffset > 0L) {
                startedAt = System.currentTimeMillis();
                startedBytes = 0L;
            }
            try (InputStream input = new BufferedInputStream(connection.getInputStream(), 1024 * 1024);
                 RandomAccessFile output = new RandomAccessFile(partial, "rw")) {
                if (append) {
                    output.seek(offset);
                } else {
                    output.setLength(0L);
                }
                byte[] buffer = new byte[1024 * 1024];
                long downloaded = offset;
                long lastPublish = 0L;
                while (true) {
                    if (cancelled) throw new CancelledException();
                    int read = input.read(buffer);
                    if (read < 0) break;
                    output.write(buffer, 0, read);
                    downloaded += read;
                    if (downloaded > source.expectedBytes) {
                        throw new IOException("checkpoint exceeded expected size");
                    }
                    long now = System.currentTimeMillis();
                    if (listener != null && (now - lastPublish >= 300L || downloaded == source.expectedBytes)) {
                        long elapsed = Math.max(1L, now - startedAt);
                        long bps = Math.max(0L, (downloaded - startedBytes) * 1000L / elapsed);
                        listener.onProgress(
                                CheckpointDownloadPolicy.percent(downloaded, source.expectedBytes),
                                downloaded,
                                source.expectedBytes,
                                bps,
                                append ? "断点续传 · " + source.label : "下载中 · " + source.label);
                        lastPublish = now;
                    }
                }
                return downloaded;
            }
        } finally {
            if (connection != null) connection.disconnect();
        }
    }

    public static boolean isVerified(Context context) {
        File target = checkpointFile(context);
        File marker = verifiedMarker(context);
        MobileI2VDownloadSource source = MobileI2VDownloadSource.official();
        if (!target.isFile() || target.length() != source.expectedBytes || !marker.isFile()) return false;
        try (FileInputStream in = new FileInputStream(marker)) {
            byte[] bytes = new byte[(int) Math.min(marker.length(), 256L)];
            int read = in.read(bytes);
            if (read <= 0) return false;
            String value = new String(bytes, 0, read, StandardCharsets.UTF_8).trim();
            return source.expectedSha256.equalsIgnoreCase(value);
        } catch (IOException error) {
            return false;
        }
    }

    public static long partialBytes(Context context) {
        File partial = partialFile(context);
        return partial.isFile() ? partial.length() : 0L;
    }

    public static File checkpointFile(Context context) {
        return new File(checkpointDir(context), MobileI2VDownloadSource.official().fileName);
    }

    private static File partialFile(Context context) {
        return new File(checkpointDir(context), MobileI2VDownloadSource.official().fileName + ".part");
    }

    private static File verifiedMarker(Context context) {
        return new File(checkpointDir(context), "hybrid_371.pth.sha256.ok");
    }

    private static File checkpointDir(Context context) {
        return new File(context.getFilesDir(), "mobilei2v/upstream");
    }

    private static String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream input = new BufferedInputStream(new FileInputStream(file), 1024 * 1024)) {
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) >= 0) {
                if (read > 0) digest.update(buffer, 0, read);
            }
        }
        StringBuilder out = new StringBuilder(64);
        for (byte b : digest.digest()) out.append(String.format(Locale.US, "%02x", b & 0xff));
        return out.toString();
    }

    private static void writeMarker(File marker, String sha) throws IOException {
        try (FileOutputStream out = new FileOutputStream(marker, false)) {
            out.write((sha + "\n").getBytes(StandardCharsets.UTF_8));
            out.getFD().sync();
        }
    }

    private static void copyFile(File source, File target) throws IOException {
        try (InputStream in = new FileInputStream(source);
             FileOutputStream out = new FileOutputStream(target, false)) {
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = in.read(buffer)) >= 0) {
                if (read > 0) out.write(buffer, 0, read);
            }
            out.getFD().sync();
        }
    }
}
