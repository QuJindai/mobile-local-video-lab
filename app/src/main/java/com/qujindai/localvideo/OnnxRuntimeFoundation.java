package com.qujindai.localvideo;

import ai.onnxruntime.OrtEnvironment;

public final class OnnxRuntimeFoundation {
    public static final class Status {
        public final boolean jniLoaded;
        public final boolean mobileI2vExecutionImplemented;
        public final String message;

        Status(boolean jniLoaded, boolean mobileI2vExecutionImplemented, String message) {
            this.jniLoaded = jniLoaded;
            this.mobileI2vExecutionImplemented = mobileI2vExecutionImplemented;
            this.message = message;
        }
    }

    private OnnxRuntimeFoundation() {}

    public static Status probe() {
        try {
            OrtEnvironment.getEnvironment();
            return new Status(true, false,
                    "ONNX Runtime 1.29 JNI 已加载；MobileI2V 图导出/执行层仍处于部署开发阶段");
        } catch (Throwable error) {
            return new Status(false, false,
                    "ONNX Runtime JNI 加载失败: " + error.getClass().getSimpleName()
                            + (error.getMessage() == null ? "" : " · " + error.getMessage()));
        }
    }
}
