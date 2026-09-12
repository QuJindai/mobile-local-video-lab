package com.qujindai.localvideo;

import android.content.Context;
import android.net.Uri;
import android.os.StatFs;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashSet;
import java.util.Locale;
import java.util.Properties;
import java.util.Set;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public final class ModelPackInstaller {
    private static final long MAX_PACK_BYTES = 6L * 1024L * 1024L * 1024L;
    private static final long COPY_BUFFER = 1024L * 1024L;
    private static final int MAX_MANIFEST_BYTES = 256 * 1024;

    public interface ProgressListener {
        void onProgress(int percent, String message);
    }

    private static final class ManifestBundle {
        final ModelPackManifest manifest;
        final AcceleratedPackManifest accelerated;

        ManifestBundle(ModelPackManifest manifest, AcceleratedPackManifest accelerated) {
            this.manifest = manifest;
            this.accelerated = accelerated;
        }
    }

    private ModelPackInstaller() {}

    public static InstalledModelPack install(Context context, Uri source, ProgressListener listener)
            throws IOException {
        File cacheZip = new File(context.getCacheDir(), "model-pack-" + UUID.randomUUID() + ".zip");
        File packsRoot = new File(context.getFilesDir(), "modelpacks");
        if (!packsRoot.exists() && !packsRoot.mkdirs()) {
            throw new IOException("cannot create model pack root");
        }
        try {
            progress(listener, 2, "复制模型包到安全临时区");
            copyUri(context, source, cacheZip);
            if (cacheZip.length() <= 0 || cacheZip.length() > MAX_PACK_BYTES) {
                throw new IOException("model pack size out of range: " + cacheZip.length());
            }

            try (ZipFile zip = new ZipFile(cacheZip)) {
                ZipEntry manifestEntry = zip.getEntry("model-pack.properties");
                if (manifestEntry == null || manifestEntry.isDirectory()) {
                    throw new IOException("model-pack.properties missing");
                }
                if (manifestEntry.getSize() > MAX_MANIFEST_BYTES) {
                    throw new IOException("model pack manifest is too large");
                }
                byte[] manifestBytes;
                try (InputStream in = zip.getInputStream(manifestEntry)) {
                    manifestBytes = readLimited(in, MAX_MANIFEST_BYTES);
                }
                ManifestBundle bundle = parseManifest(manifestBytes);
                ModelPackManifest manifest = bundle.manifest;
                progress(listener, 8, "清单有效 · " + manifest.id + " " + manifest.version);

                long declaredBytes = 0;
                Set<String> expected = new HashSet<>(manifest.files);
                for (String path : expected) {
                    ZipEntry entry = zip.getEntry(path);
                    if (entry == null || entry.isDirectory()) {
                        throw new IOException("missing model artifact: " + path);
                    }
                    if (!PackPathPolicy.isSafe(entry.getName())) {
                        throw new IOException("unsafe ZIP entry: " + entry.getName());
                    }
                    if (entry.getSize() > 0) declaredBytes += entry.getSize();
                    if (declaredBytes > MAX_PACK_BYTES) {
                        throw new IOException("expanded model pack exceeds safety limit");
                    }
                }

                long free = new StatFs(packsRoot.getAbsolutePath()).getAvailableBytes();
                long needed = Math.max(cacheZip.length() * 2L, declaredBytes + 256L * 1024L * 1024L);
                if (free < needed) {
                    throw new IOException("insufficient storage for verified install: need ~"
                            + (needed / 1048576L) + " MB, free " + (free / 1048576L) + " MB");
                }

                File tempRoot = new File(packsRoot, ".install-" + UUID.randomUUID());
                if (!tempRoot.mkdirs()) throw new IOException("cannot create temporary model directory");
                long written = 0;
                try {
                    copyManifest(manifestBytes, tempRoot);
                    int index = 0;
                    for (String path : manifest.files) {
                        ZipEntry entry = zip.getEntry(path);
                        File out = safeChild(tempRoot, path);
                        File parent = out.getParentFile();
                        if (parent != null && !parent.exists() && !parent.mkdirs()) {
                            throw new IOException("cannot create artifact directory: " + parent);
                        }
                        String digest = extractAndHash(zip, entry, out);
                        if (!digest.equals(manifest.expectedSha256(path))) {
                            throw new IOException("SHA-256 mismatch for " + path);
                        }
                        written += out.length();
                        index++;
                        progress(listener, 8 + index * 82 / Math.max(1, manifest.files.size()),
                                "已验证 " + index + "/" + manifest.files.size() + " · " + path);
                    }

                    File idRoot = new File(packsRoot, manifest.id);
                    if (!idRoot.exists() && !idRoot.mkdirs()) {
                        throw new IOException("cannot create model id directory");
                    }
                    File finalRoot = new File(idRoot, manifest.version);
                    File backup = new File(idRoot, ".backup-" + UUID.randomUUID());
                    boolean hadOld = finalRoot.exists();
                    if (hadOld && !finalRoot.renameTo(backup)) {
                        throw new IOException("cannot stage previous model version for replacement");
                    }
                    boolean activated = false;
                    try {
                        if (!tempRoot.renameTo(finalRoot)) {
                            throw new IOException("cannot atomically activate model pack");
                        }
                        activated = true;
                    } finally {
                        if (!activated && hadOld && backup.exists()) {
                            //noinspection ResultOfMethodCallIgnored
                            backup.renameTo(finalRoot);
                        }
                    }
                    deleteRecursively(backup);
                    progress(listener, 100, bundle.accelerated == null
                            ? "模型包安装完成"
                            : "GPU 模型包安装完成 · 加速清单已验证");
                    return new InstalledModelPack(
                            manifest, bundle.accelerated, finalRoot, written);
                } catch (Throwable error) {
                    deleteRecursively(tempRoot);
                    if (error instanceof IOException) throw (IOException) error;
                    throw new IOException("model pack install failed: " + error.getMessage(), error);
                }
            }
        } finally {
            //noinspection ResultOfMethodCallIgnored
            cacheZip.delete();
        }
    }

    private static ManifestBundle parseManifest(byte[] bytes) throws IOException {
        Properties properties = new Properties();
        properties.load(new ByteArrayInputStream(bytes));
        String format = properties.getProperty("format", "").trim();
        if (AcceleratedPackManifest.FORMAT.equals(format)) {
            AcceleratedPackManifest accelerated = AcceleratedPackManifest.parse(
                    new ByteArrayInputStream(bytes));
            return new ManifestBundle(ModelPackManifest.fromAccelerated(accelerated), accelerated);
        }
        if (ModelPackManifest.FORMAT.equals(format)) {
            return new ManifestBundle(
                    ModelPackManifest.parse(new ByteArrayInputStream(bytes)), null);
        }
        throw new IllegalArgumentException("unsupported model pack format: " + format);
    }

    private static byte[] readLimited(InputStream input, int maxBytes) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream(Math.min(32 * 1024, maxBytes));
        byte[] buffer = new byte[16 * 1024];
        int total = 0;
        int n;
        while ((n = input.read(buffer)) >= 0) {
            if (n == 0) continue;
            total += n;
            if (total > maxBytes) throw new IOException("model pack manifest is too large");
            out.write(buffer, 0, n);
        }
        return out.toByteArray();
    }

    private static void copyUri(Context context, Uri source, File target) throws IOException {
        long total = 0;
        try (InputStream raw = context.getContentResolver().openInputStream(source)) {
            if (raw == null) throw new IOException("cannot open model pack URI");
            try (BufferedInputStream in = new BufferedInputStream(raw);
                 BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(target))) {
                byte[] buffer = new byte[(int) COPY_BUFFER];
                int n;
                while ((n = in.read(buffer)) >= 0) {
                    if (n == 0) continue;
                    total += n;
                    if (total > MAX_PACK_BYTES) throw new IOException("model pack exceeds 6 GB limit");
                    out.write(buffer, 0, n);
                }
            }
        }
    }

    private static void copyManifest(byte[] bytes, File tempRoot) throws IOException {
        File target = new File(tempRoot, "model-pack.properties");
        try (FileOutputStream out = new FileOutputStream(target)) {
            out.write(bytes);
        }
    }

    private static String extractAndHash(ZipFile zip, ZipEntry entry, File target) throws IOException {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IOException(impossible);
        }
        long written = 0;
        try (InputStream in = new BufferedInputStream(zip.getInputStream(entry));
             BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(target))) {
            byte[] buffer = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buffer)) >= 0) {
                if (n == 0) continue;
                written += n;
                if (written > MAX_PACK_BYTES) throw new IOException("artifact exceeds safety limit");
                digest.update(buffer, 0, n);
                out.write(buffer, 0, n);
            }
        }
        return toHex(digest.digest());
    }

    private static File safeChild(File root, String relative) throws IOException {
        if (!PackPathPolicy.isSafe(relative)) throw new IOException("unsafe model path: " + relative);
        File child = new File(root, relative);
        String rootPath = root.getCanonicalPath() + File.separator;
        String childPath = child.getCanonicalPath();
        if (!childPath.startsWith(rootPath)) throw new IOException("model path escapes install root");
        return child;
    }

    static InstalledModelPack inspect(File root) throws IOException {
        File manifestFile = new File(root, "model-pack.properties");
        if (!manifestFile.isFile()) throw new IOException("installed manifest missing");
        byte[] manifestBytes;
        try (InputStream in = new FileInputStream(manifestFile)) {
            manifestBytes = readLimited(in, MAX_MANIFEST_BYTES);
        }
        ManifestBundle bundle = parseManifest(manifestBytes);
        ModelPackManifest manifest = bundle.manifest;
        long bytes = 0;
        for (String path : manifest.files) {
            File file = safeChild(root, path);
            if (!file.isFile()) throw new IOException("installed artifact missing: " + path);
            bytes += file.length();
        }
        return new InstalledModelPack(manifest, bundle.accelerated, root, bytes);
    }

    static void deleteRecursively(File file) {
        if (file == null || !file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) deleteRecursively(child);
            }
        }
        //noinspection ResultOfMethodCallIgnored
        file.delete();
    }

    private static void progress(ProgressListener listener, int percent, String message) {
        if (listener != null) listener.onProgress(percent, message);
    }

    private static String toHex(byte[] bytes) {
        StringBuilder builder = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) builder.append(String.format(Locale.US, "%02x", b & 0xff));
        return builder.toString();
    }
}
