package com.qujindai.localvideo;

import java.io.File;

public final class InstalledModelPack {
    public final ModelPackManifest manifest;
    public final File root;
    public final long bytes;

    InstalledModelPack(ModelPackManifest manifest, File root, long bytes) {
        this.manifest = manifest;
        this.root = root;
        this.bytes = bytes;
    }

    public File artifact(String relativePath) {
        return new File(root, relativePath);
    }
}
