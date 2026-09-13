package com.qujindai.facere;

import java.io.File;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public final class OrtSessions {
    private OrtSessions() {}

    public static OrtSession create(File model) throws OrtException {
        if (model == null || !model.isFile() || model.length() <= 0L) {
            throw new IllegalArgumentException("invalid ONNX model file");
        }
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(Math.max(1,
                Math.min(4, Runtime.getRuntime().availableProcessors())));
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        try {
            return OrtEnvironment.getEnvironment().createSession(model.getAbsolutePath(), options);
        } finally {
            options.close();
        }
    }
}
