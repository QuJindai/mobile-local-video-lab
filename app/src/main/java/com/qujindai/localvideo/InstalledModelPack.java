package com.qujindai.localvideo;

import java.io.File;

public final class InstalledModelPack {
    public final ModelPackManifest manifest;
    public final AcceleratedPackManifest acceleratedManifest;
    public final File root;
    public final long bytes;

    InstalledModelPack(ModelPackManifest manifest, File root, long bytes) {
        this(manifest, null, root, bytes);
    }

    InstalledModelPack(
            ModelPackManifest manifest,
            AcceleratedPackManifest acceleratedManifest,
            File root,
            long bytes) {
        this.manifest = manifest;
        this.acceleratedManifest = acceleratedManifest;
        this.root = root;
        this.bytes = bytes;
    }

    public boolean isAcceleratedMobileI2V() {
        return acceleratedManifest != null && acceleratedManifest.isMobileI2VGpuRunnable();
    }

    public File artifact(String relativePath) {
        return new File(root, relativePath);
    }
}
