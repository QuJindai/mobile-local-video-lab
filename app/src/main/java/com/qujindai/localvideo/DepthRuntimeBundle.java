package com.qujindai.localvideo;

import android.content.Context;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;

public final class DepthRuntimeBundle {
    public static final String ASSET = "models/depth-anything-v2/model_int8.onnx";
    public static final String SHA256 = "01aa7a23de3f4a0ee1a2bb9997e6918104c85a9f95dea46d27b9b3fb0c6b9001";

    private final File model;

    private DepthRuntimeBundle(File model) {
        this.model = model;
    }

    public File getModel() {
        return model;
    }

    public static boolean isPackaged(Context context) {
        try (InputStream ignored = context.getAssets().open(ASSET)) {
            return true;
        } catch (IOException error) {
            return false;
        }
    }

    public static DepthRuntimeBundle installAndVerify(Context context) throws IOException {
        File root = new File(context.getFilesDir(), "runtime/depth-anything-v2-int8");
        File model = new File(root, "model_int8.onnx");
        if (!root.exists() && !root.mkdirs()) {
            throw new IOException("cannot create Depth Anything runtime directory");
        }
        if (!model.isFile() || !SHA256.equals(sha256(model))) {
            File temp = new File(root, "model_int8.onnx.tmp");
            copyAsset(context, temp);
            String digest = sha256(temp);
            if (!SHA256.equals(digest)) {
                //noinspection ResultOfMethodCallIgnored
                temp.delete();
                throw new IOException("Depth Anything model SHA-256 mismatch: " + digest);
            }
            if (model.exists() && !model.delete()) {
                throw new IOException("cannot replace stale Depth Anything model");
            }
            if (!temp.renameTo(model)) {
                throw new IOException("cannot activate Depth Anything model");
            }
        }
        return new DepthRuntimeBundle(model);
    }

    private static void copyAsset(Context context, File target) throws IOException {
        try (InputStream raw = context.getAssets().open(ASSET);
             BufferedInputStream in = new BufferedInputStream(raw);
             BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(target))) {
            byte[] buffer = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buffer)) >= 0) {
                if (n > 0) out.write(buffer, 0, n);
            }
        }
    }

    private static String sha256(File file) throws IOException {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IOException(impossible);
        }
        try (BufferedInputStream in = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buffer = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buffer)) >= 0) {
                if (n > 0) digest.update(buffer, 0, n);
            }
        }
        StringBuilder builder = new StringBuilder(64);
        for (byte value : digest.digest()) {
            builder.append(String.format(Locale.US, "%02x", value & 0xff));
        }
        return builder.toString();
    }
}
