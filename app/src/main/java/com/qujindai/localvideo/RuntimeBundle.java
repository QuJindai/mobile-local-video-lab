package com.qujindai.localvideo;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public final class RuntimeBundle {
    private static final String MODEL_ASSET_DIR = "models/rife-v4.6";
    private static final String[] MODEL_FILES = { "flownet.param", "flownet.bin" };

    private final File executable;
    private final File modelDir;

    private RuntimeBundle(File executable, File modelDir) {
        this.executable = executable;
        this.modelDir = modelDir;
    }

    public static RuntimeBundle installAndVerify(Context context) throws IOException {
        File executable = new File(context.getApplicationInfo().nativeLibraryDir, "librife.so");
        if (!executable.isFile() || executable.length() < 100_000) {
            throw new IOException("RIFE native runtime is missing from APK");
        }

        File modelDir = new File(context.getFilesDir(), "runtime/rife-v4.6");
        if (!modelDir.exists() && !modelDir.mkdirs()) {
            throw new IOException("cannot create model directory");
        }

        for (String name : MODEL_FILES) {
            File destination = new File(modelDir, name);
            if (!destination.isFile() || destination.length() == 0) {
                copyAsset(context, MODEL_ASSET_DIR + "/" + name, destination);
            }
        }

        File param = new File(modelDir, "flownet.param");
        File bin = new File(modelDir, "flownet.bin");
        if (param.length() < 1_000 || bin.length() < 1_000_000) {
            throw new IOException("RIFE model payload failed validation");
        }
        return new RuntimeBundle(executable, modelDir);
    }

    private static void copyAsset(Context context, String assetPath, File destination) throws IOException {
        try (InputStream in = context.getAssets().open(assetPath);
             FileOutputStream out = new FileOutputStream(destination)) {
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
            out.getFD().sync();
        }
    }

    public File getExecutable() { return executable; }
    public File getModelDir() { return modelDir; }
}
