package com.qujindai.facere;

import java.io.File;

public final class FaceModelFiles {
    public final File root;
    public final File detector;
    public final File recognizer;
    public final File swapper;
    public final File emap;

    FaceModelFiles(File root) {
        this.root = root;
        this.detector = new File(root, FacePackContract.DETECTOR);
        this.recognizer = new File(root, FacePackContract.RECOGNIZER);
        this.swapper = new File(root, FacePackContract.SWAPPER);
        this.emap = new File(root, FacePackContract.EMAP);
    }

    public long totalBytes() {
        return detector.length() + recognizer.length() + swapper.length() + emap.length();
    }
}
