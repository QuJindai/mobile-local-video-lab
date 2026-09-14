package com.qujindai.facere;

import android.content.Context;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;
import java.util.Locale;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public final class FaceModelDownloader {
    public interface Progress {
        void onProgress(int percent, String message, long downloaded, long total,
                        long bytesPerSecond, FaceModelDownloadCatalog.SourceMode source);
    }

    public static final class CancelToken {
        private volatile boolean cancelled;
        public void cancel() { cancelled = true; }
        public boolean isCancelled() { return cancelled; }
        void check() throws CancelledException {
            if (cancelled) throw new CancelledException();
        }
    }

    public static final class CancelledException extends IOException {
        CancelledException() { super("download paused"); }
    }

    private static final int CONNECT_TIMEOUT_MS = 20_000;
    private static final int READ_TIMEOUT_MS = 30_000;
    private static final int MAX_REDIRECTS = 8;

    private FaceModelDownloader() {}

    public static FaceModelFiles downloadAndInstall(
            Context context,
            FaceModelDownloadCatalog.SourceMode requested,
            CancelToken token,
            Progress progress) throws IOException {
        if (token == null) token = new CancelToken();
        File cache = new File(context.getFilesDir(), "face-re-downloads");
        if (!cache.exists() && !cache.mkdirs()) throw new IOException("cannot create download cache");

        IOException last = null;
        List<FaceModelDownloadCatalog.SourceMode> attempts =
                FaceModelDownloadCatalog.attemptOrder(requested);
        for (FaceModelDownloadCatalog.SourceMode source : attempts) {
            token.check();
            File staging = FaceModelStore.createStaging(context, "download-staging");
            try {
                report(progress, 0, "连接" + sourceLabel(source) + "…", 0, 0, 0, source);
                if (source == FaceModelDownloadCatalog.SourceMode.CHINA_MIRROR) {
                    prepareFromMirror(cache, staging, token, progress, source);
                } else {
                    prepareFromOfficial(cache, staging, token, progress, source);
                }
                token.check();
                report(progress, 95, "从 INSwapper 本地生成 EMAP…", 0, 0, 0, source);
                OnnxEmapExtractor.extract(new File(staging, "inswapper_128.onnx"),
                        new File(staging, "emap.bin"));
                token.check();
                report(progress, 98, "校验并安装模型…", 0, 0, 0, source);
                FaceModelFiles files = FaceModelStore.activatePrepared(context, staging,
                        message -> report(progress, 99, message, 0, 0, 0, source));
                if (source == FaceModelDownloadCatalog.SourceMode.OFFICIAL) {
                    // buffalo_l.zip is only a transport container; keep installed models, save storage.
                    //noinspection ResultOfMethodCallIgnored
                    new File(cache, "buffalo_l.zip").delete();
                }
                report(progress, 100, "模型 READY · " + sourceLabel(source), 0, 0, 0, source);
                return files;
            } catch (CancelledException cancelled) {
                FaceModelStore.deleteRecursively(staging);
                throw cancelled;
            } catch (IOException error) {
                last = error;
                FaceModelStore.deleteRecursively(staging);
                report(progress, 0, sourceLabel(source) + "失败，" +
                        (attempts.size() > 1 ? "准备切换下载源" : "下载终止") + " · " + error.getMessage(),
                        0, 0, 0, source);
            }
        }
        throw last != null ? last : new IOException("no model download source available");
    }

    private static void prepareFromMirror(File cache, File staging, CancelToken token,
                                          Progress progress,
                                          FaceModelDownloadCatalog.SourceMode source) throws IOException {
        File detector = download(cache, "det_10g.onnx",
                FaceModelDownloadCatalog.MIRROR_DETECTOR_URL,
                FaceModelDownloadCatalog.DETECTOR_SHA256, -1L,
                token, progress, 1, 15, "检测模型", source);
        linkOrCopy(detector, new File(staging, "det_10g.onnx"), token);

        File recognizer = download(cache, "w600k_r50.onnx",
                FaceModelDownloadCatalog.MIRROR_RECOGNIZER_URL,
                FaceModelDownloadCatalog.RECOGNIZER_SHA256, -1L,
                token, progress, 15, 38, "识别模型", source);
        linkOrCopy(recognizer, new File(staging, "w600k_r50.onnx"), token);

        File swapper = download(cache, "inswapper_128.onnx",
                FaceModelDownloadCatalog.MIRROR_SWAPPER_URL,
                FaceModelDownloadCatalog.SWAPPER_SHA256,
                FaceModelDownloadCatalog.SWAPPER_BYTES,
                token, progress, 38, 94, "换脸模型", source);
        linkOrCopy(swapper, new File(staging, "inswapper_128.onnx"), token);
    }

    private static void prepareFromOfficial(File cache, File staging, CancelToken token,
                                            Progress progress,
                                            FaceModelDownloadCatalog.SourceMode source) throws IOException {
        File buffalo = download(cache, "buffalo_l.zip",
                FaceModelDownloadCatalog.OFFICIAL_BUFFALO_URL,
                FaceModelDownloadCatalog.BUFFALO_ZIP_SHA256,
                FaceModelDownloadCatalog.BUFFALO_ZIP_BYTES,
                token, progress, 1, 37, "buffalo_l", source);
        report(progress, 38, "提取 det_10g / w600k_r50…", 0, 0, 0, source);
        extractBuffalo(buffalo, staging, token);
        verifySha(new File(staging, "det_10g.onnx"), FaceModelDownloadCatalog.DETECTOR_SHA256);
        verifySha(new File(staging, "w600k_r50.onnx"), FaceModelDownloadCatalog.RECOGNIZER_SHA256);

        File swapper = download(cache, "inswapper_128.onnx",
                FaceModelDownloadCatalog.OFFICIAL_SWAPPER_URL,
                FaceModelDownloadCatalog.SWAPPER_SHA256,
                FaceModelDownloadCatalog.SWAPPER_BYTES,
                token, progress, 40, 94, "换脸模型", source);
        linkOrCopy(swapper, new File(staging, "inswapper_128.onnx"), token);
    }

    private static File download(File cache, String name, String initialUrl, String expectedSha,
                                 long expectedBytes, CancelToken token, Progress progress,
                                 int phaseStart, int phaseEnd, String label,
                                 FaceModelDownloadCatalog.SourceMode source) throws IOException {
        if (!FaceModelDownloadCatalog.isTrustedUrl(initialUrl)) throw new IOException("untrusted model URL");
        File complete = new File(cache, name);
        if (complete.isFile()) {
            if (matches(complete, expectedSha, expectedBytes)) {
                report(progress, phaseEnd, label + " · 已缓存", complete.length(), complete.length(), 0, source);
                return complete;
            }
            //noinspection ResultOfMethodCallIgnored
            complete.delete();
        }

        File part = new File(cache, name + ".part");
        if (expectedBytes > 0L && part.length() > expectedBytes) {
            //noinspection ResultOfMethodCallIgnored
            part.delete();
        }

        boolean retriedFull = false;
        while (true) {
            token.check();
            long existing = part.isFile() ? part.length() : 0L;
            OpenConnection opened = open(initialUrl, existing);
            HttpURLConnection connection = opened.connection;
            try {
                int code = opened.code;
                DownloadResumePolicy.Action action = DownloadResumePolicy.action(existing, code);
                if (action == DownloadResumePolicy.Action.VERIFY_EXISTING) {
                    if (part.isFile() && matches(part, expectedSha, expectedBytes)) {
                        promote(part, complete);
                        return complete;
                    }
                    //noinspection ResultOfMethodCallIgnored
                    part.delete();
                    if (retriedFull) throw new IOException(label + " range state invalid");
                    retriedFull = true;
                    continue;
                }
                if (code != 200 && code != 206) throw new IOException(label + " HTTP " + code);

                boolean append = action == DownloadResumePolicy.Action.APPEND;
                if (!append) existing = 0L;
                long contentLength = connection.getContentLengthLong();
                long total = expectedBytes > 0L ? expectedBytes
                        : (contentLength > 0L ? existing + contentLength : -1L);
                long downloaded = existing;
                long speedWindowBytes = downloaded;
                long speedWindowTime = System.nanoTime();
                long lastUi = speedWindowTime;

                try (InputStream input = new BufferedInputStream(connection.getInputStream(), 1024 * 1024);
                     BufferedOutputStream output = new BufferedOutputStream(
                             new FileOutputStream(part, append), 1024 * 1024)) {
                    byte[] buffer = new byte[1024 * 1024];
                    int read;
                    while ((read = input.read(buffer)) != -1) {
                        token.check();
                        output.write(buffer, 0, read);
                        downloaded += read;
                        if (expectedBytes > 0L && downloaded > expectedBytes) {
                            throw new IOException(label + " exceeds expected size");
                        }
                        long now = System.nanoTime();
                        if (now - lastUi >= 350_000_000L) {
                            long elapsedNs = Math.max(1L, now - speedWindowTime);
                            long bps = (long) ((downloaded - speedWindowBytes) * 1_000_000_000.0 / elapsedNs);
                            int percent = phasePercent(phaseStart, phaseEnd, downloaded, total);
                            report(progress, percent, label + " · " + human(downloaded) +
                                            (total > 0L ? " / " + human(total) : ""),
                                    downloaded, total, bps, source);
                            speedWindowBytes = downloaded;
                            speedWindowTime = now;
                            lastUi = now;
                        }
                    }
                }
                token.check();
                if (expectedBytes > 0L && part.length() != expectedBytes) {
                    throw new IOException(label + " size mismatch: " + part.length());
                }
                verifySha(part, expectedSha);
                promote(part, complete);
                report(progress, phaseEnd, label + " · 校验通过", complete.length(), complete.length(), 0, source);
                return complete;
            } finally {
                connection.disconnect();
            }
        }
    }

    private static OpenConnection open(String initialUrl, long existing) throws IOException {
        URL url = new URL(initialUrl);
        for (int redirects = 0; redirects <= MAX_REDIRECTS; redirects++) {
            if (!FaceModelDownloadCatalog.isTrustedUrl(url.toString())) {
                throw new IOException("redirected to untrusted host: " + url.getHost());
            }
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setInstanceFollowRedirects(false);
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            connection.setRequestProperty("User-Agent", "Face-RE/0.9.1 Android");
            connection.setRequestProperty("Accept-Encoding", "identity");
            if (existing > 0L) connection.setRequestProperty("Range", "bytes=" + existing + "-");
            int code = connection.getResponseCode();
            if (code >= 300 && code < 400) {
                String location = connection.getHeaderField("Location");
                connection.disconnect();
                if (location == null || location.isEmpty()) throw new IOException("redirect without Location");
                url = new URL(url, location);
                continue;
            }
            return new OpenConnection(connection, code);
        }
        throw new IOException("too many redirects");
    }

    private static void extractBuffalo(File zipFile, File staging, CancelToken token) throws IOException {
        boolean detector = false;
        boolean recognizer = false;
        try (ZipInputStream zip = new ZipInputStream(new BufferedInputStream(new FileInputStream(zipFile), 1024 * 1024))) {
            ZipEntry entry;
            byte[] buffer = new byte[1024 * 1024];
            while ((entry = zip.getNextEntry()) != null) {
                token.check();
                if (entry.isDirectory()) continue;
                String name = new File(entry.getName()).getName();
                if (!name.equals("det_10g.onnx") && !name.equals("w600k_r50.onnx")) continue;
                File out = new File(staging, name);
                try (BufferedOutputStream output = new BufferedOutputStream(new FileOutputStream(out), 1024 * 1024)) {
                    int read;
                    while ((read = zip.read(buffer)) != -1) {
                        token.check();
                        output.write(buffer, 0, read);
                    }
                }
                if (name.equals("det_10g.onnx")) detector = true;
                if (name.equals("w600k_r50.onnx")) recognizer = true;
            }
        }
        if (!detector || !recognizer) throw new IOException("buffalo_l.zip missing required models");
    }

    private static void linkOrCopy(File source, File destination, CancelToken token) throws IOException {
        token.check();
        try {
            Files.createLink(destination.toPath(), source.toPath());
            return;
        } catch (Throwable ignored) {
        }
        try (BufferedInputStream input = new BufferedInputStream(new FileInputStream(source), 1024 * 1024);
             BufferedOutputStream output = new BufferedOutputStream(new FileOutputStream(destination), 1024 * 1024)) {
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                token.check();
                output.write(buffer, 0, read);
            }
        }
    }

    private static boolean matches(File file, String expectedSha, long expectedBytes) throws IOException {
        if (!file.isFile()) return false;
        if (expectedBytes > 0L && file.length() != expectedBytes) return false;
        return sha256(file).equalsIgnoreCase(expectedSha);
    }

    private static void verifySha(File file, String expectedSha) throws IOException {
        String actual = sha256(file);
        if (!actual.equalsIgnoreCase(expectedSha)) {
            throw new IOException(file.getName() + " SHA-256 mismatch: " + actual);
        }
    }

    static String sha256(File file) throws IOException {
        final MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IOException(impossible);
        }
        try (BufferedInputStream input = new BufferedInputStream(new FileInputStream(file), 1024 * 1024)) {
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) digest.update(buffer, 0, read);
        }
        StringBuilder hex = new StringBuilder(64);
        for (byte b : digest.digest()) hex.append(String.format(Locale.US, "%02x", b & 0xff));
        return hex.toString();
    }

    private static void promote(File part, File complete) throws IOException {
        if (complete.exists() && !complete.delete()) throw new IOException("cannot replace download cache");
        if (part.renameTo(complete)) return;
        try (BufferedInputStream input = new BufferedInputStream(new FileInputStream(part), 1024 * 1024);
             BufferedOutputStream output = new BufferedOutputStream(new FileOutputStream(complete), 1024 * 1024)) {
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) output.write(buffer, 0, read);
        }
        //noinspection ResultOfMethodCallIgnored
        part.delete();
    }

    private static int phasePercent(int start, int end, long downloaded, long total) {
        if (total <= 0L) return start;
        double ratio = Math.max(0.0, Math.min(1.0, downloaded / (double) total));
        return start + (int) Math.floor((end - start) * ratio);
    }

    private static void report(Progress progress, int percent, String message, long downloaded,
                               long total, long bps, FaceModelDownloadCatalog.SourceMode source) {
        if (progress != null) progress.onProgress(percent, message, downloaded, total, bps, source);
    }

    static String sourceLabel(FaceModelDownloadCatalog.SourceMode source) {
        return source == FaceModelDownloadCatalog.SourceMode.CHINA_MIRROR ? "国内镜像" : "InsightFace 官方源";
    }

    private static String human(long bytes) {
        if (bytes < 0L) return "?";
        return String.format(Locale.US, "%.1f MB", bytes / 1048576.0);
    }

    private static final class OpenConnection {
        final HttpURLConnection connection;
        final int code;
        OpenConnection(HttpURLConnection connection, int code) {
            this.connection = connection;
            this.code = code;
        }
    }
}
