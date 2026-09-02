package com.qujindai.localvideo;

public final class BackendRouter {
    public enum Backend {
        RIFE_MOTION,
        DEPTH_RIFE,
        MOBILE_I2V
    }

    public enum Blocker {
        NONE,
        BUILTIN_RUNTIME_MISSING,
        DEPTH_RUNTIME_MISSING,
        MODEL_PACK_MISSING,
        RUNTIME_PENDING,
        INSUFFICIENT_RAM
    }

    public static final class Decision {
        public final Backend backend;
        public final boolean ready;
        public final Blocker blocker;
        public final String message;

        Decision(Backend backend, boolean ready, Blocker blocker, String message) {
            this.backend = backend;
            this.ready = ready;
            this.blocker = blocker;
            this.message = message;
        }
    }

    private BackendRouter() {}

    public static Decision resolve(
            Backend backend,
            boolean rifeRuntimeReady,
            boolean mobileModelPackInstalled,
            boolean mobileRuntimeReady,
            long totalRamMb) {
        return resolve(
                backend,
                rifeRuntimeReady,
                false,
                mobileModelPackInstalled,
                mobileRuntimeReady,
                totalRamMb);
    }

    public static Decision resolve(
            Backend backend,
            boolean rifeRuntimeReady,
            boolean depthRuntimeReady,
            boolean mobileModelPackInstalled,
            boolean mobileRuntimeReady,
            long totalRamMb) {
        if (backend == Backend.RIFE_MOTION) {
            if (rifeRuntimeReady) {
                return new Decision(backend, true, Blocker.NONE,
                        "RIFE v4.6 / ncnn / Vulkan 已就绪");
            }
            return new Decision(backend, false, Blocker.BUILTIN_RUNTIME_MISSING,
                    "内置 RIFE 运行时不完整");
        }

        if (backend == Backend.DEPTH_RIFE) {
            if (!rifeRuntimeReady) {
                return new Decision(backend, false, Blocker.BUILTIN_RUNTIME_MISSING,
                        "RIFE 补帧运行时不完整");
            }
            if (!depthRuntimeReady) {
                return new Decision(backend, false, Blocker.DEPTH_RUNTIME_MISSING,
                        "Depth Anything V2 / ONNX Runtime 模型运行时不完整");
            }
            return new Decision(backend, true, Blocker.NONE,
                    "Depth Anything V2 INT8 + ONNX Runtime + RIFE 已就绪");
        }

        if (!mobileModelPackInstalled) {
            return new Decision(backend, false, Blocker.MODEL_PACK_MISSING,
                    "未安装 MobileI2V 模型包");
        }
        if (!mobileRuntimeReady) {
            return new Decision(backend, false, Blocker.RUNTIME_PENDING,
                    "MobileI2V 模型包已验证，但 Android 推理 runtime 尚未就绪");
        }
        if (totalRamMb > 0 && totalRamMb < 8192) {
            return new Decision(backend, false, Blocker.INSUFFICIENT_RAM,
                    "设备内存低于 MobileI2V 最低门槛 8 GB");
        }
        return new Decision(backend, true, Blocker.NONE,
                "MobileI2V 模型与 runtime 已就绪");
    }
}
