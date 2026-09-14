package com.qujindai.facere;

import android.content.Context;
import android.net.Uri;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public final class FaceModelStore {
    public interface Progress {
        void onProgress(String message);
    }

    private static final long MAX_UNPACKED_BYTES = 1_400_000_000L;
    private static final long EMAP_BYTES = 8L + 512L * 512L * 4L;

    private FaceModelStore() {}

    private static File storeRoot(Context context) {
        return new File(context.getFilesDir(), "face-re-models");
    }

    private static File activeRoot(Context context) {
        return new File(storeRoot(context), "active");
    }

    public static FaceModelFiles active(Context context) {
        File root = activeRoot(context);
        if (!root.isDirectory()) return null;
        try {
            return inspect(root);
        } catch (IOException | RuntimeException error) {
            return null;
        }
    }

    public static FaceModelFiles inspect(File root) throws IOException {
        FaceModelFiles files = new FaceModelFiles(root);
        for (String name : FacePackContract.requiredFiles()) {
            File file = new File(root, name);
            if (!file.isFile() || file.length() <= 0L) {
                throw new IOException("missing model artifact: " + name);
            }
        }
        validateEmap(files.emap);
        return files;
    }

    static File createStaging(Context context, String prefix) throws IOException {
        File root = storeRoot(context);
        if (!root.exists() && !root.mkdirs()) throw new IOException("cannot create model store");
        File staging = new File(root, prefix + "-" + UUID.randomUUID());
        if (!staging.mkdirs()) throw new IOException("cannot create model staging directory");
        return staging;
    }

    static FaceModelFiles activatePrepared(Context context, File staging, Progress progress)
            throws IOException {
        if (staging == null || !staging.isDirectory()) throw new IOException("model staging directory missing");
        File root = storeRoot(context).getCanonicalFile();
        File canonicalStaging = staging.getCanonicalFile();
        File parent = canonicalStaging.getParentFile();
        if (parent == null || !parent.equals(root)) throw new IOException("model staging path is outside store");
        inspect(canonicalStaging);

        File active = activeRoot(context);
        File backup = new File(root, "backup-" + UUID.randomUUID());
        boolean previousMoved = false;
        boolean activated = false;
        try {
            if (active.exists()) {
                if (!active.renameTo(backup)) throw new IOException("cannot stage previous model pack");
                previousMoved = true;
            }
            if (!canonicalStaging.renameTo(active)) {
                if (previousMoved) backup.renameTo(active);
                throw new IOException("cannot activate model pack");
            }
            activated = true;
            deleteRecursively(backup);
            if (progress != null) progress.onProgress("模型包已安装");
            return inspect(active);
        } finally {
            if (!activated && previousMoved && !active.exists() && backup.exists()) {
                //noinspection ResultOfMethodCallIgnored
                backup.renameTo(active);
            }
        }
    }

    public static FaceModelFiles importZip(Context context, Uri uri, Progress progress)
            throws IOException {
        if (uri == null) throw new IllegalArgumentException("model zip uri required");
        File staging = createStaging(context, "import-staging");
        Set<String> seen = new HashSet<>();
        long unpacked = 0L;
        boolean activated = false;
        try (InputStream raw = context.getContentResolver().openInputStream(uri)) {
            if (raw == null) throw new IOException("cannot open model zip");
            try (ZipInputStream zip = new ZipInputStream(new BufferedInputStream(raw))) {
                ZipEntry entry;
                byte[] buffer = new byte[1024 * 1024];
                while ((entry = zip.getNextEntry()) != null) {
                    String name = entry.getName();
                    if (entry.isDirectory()) {
                        if (!name.endsWith("/") || !name.equals("__MACOSX/")) {
                            throw new IOException("model archive must use root-level files only");
                        }
                        zip.closeEntry();
                        continue;
                    }
                    if (!FacePackContract.isSafeEntry(name)) {
                        throw new IOException("unsafe model archive entry: " + name);
                    }
                    if (!FacePackContract.isRequired(name)) {
                        throw new IOException("unexpected model archive entry: " + name);
                    }
                    if (!seen.add(name)) throw new IOException("duplicate model artifact: " + name);
                    if (progress != null) progress.onProgress("导入 " + name);
                    File out = new File(staging, name);
                    try (BufferedOutputStream output = new BufferedOutputStream(new FileOutputStream(out))) {
                        int read;
                        long entryBytes = 0L;
                        while ((read = zip.read(buffer)) != -1) {
                            entryBytes += read;
                            unpacked += read;
                            if (unpacked > MAX_UNPACKED_BYTES) {
                                throw new IOException("model archive exceeds size limit");
                            }
                            output.write(buffer, 0, read);
                        }
                        if (entryBytes <= 0L) throw new IOException("empty model artifact: " + name);
                    }
                    zip.closeEntry();
                }
            }
            if (!FacePackContract.hasRequired(seen)) {
                throw new IOException("model zip must contain detector, recognizer, swapper and emap");
            }
            FaceModelFiles result = activatePrepared(context, staging, progress);
            activated = true;
            return result;
        } finally {
            if (!activated) deleteRecursively(staging);
        }
    }

    public static void remove(Context context) {
        deleteRecursively(activeRoot(context));
    }

    static void validateEmap(File file) throws IOException {
        if (!file.isFile() || file.length() != EMAP_BYTES) {
            throw new IOException("emap.bin size invalid; expected 512x512 float32 matrix");
        }
        byte[] header = new byte[8];
        try (DataInputStream input = new DataInputStream(new FileInputStream(file))) {
            input.readFully(header);
        }
        ByteBuffer bb = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
        int rows = bb.getInt();
        int cols = bb.getInt();
        if (rows != 512 || cols != 512) {
            throw new IOException("emap.bin dimensions invalid: " + rows + "x" + cols);
        }
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
}
