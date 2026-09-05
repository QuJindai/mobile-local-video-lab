package com.qujindai.localvideo;

/**
 * Truth source for MobileI2V generation readiness.
 *
 * A raw checkpoint, legacy pack, or CPU-only runtime can never make the
 * production backend READY. Adreno/MNN OpenCL is preferred because the S24U
 * target is explicitly GPU-first. QNN HTP is accepted as a separately named
 * accelerator when available.
 */
public final class MobileI2VRuntimeProbe {
    public enum Backend {
        MNN_OPENCL,
        QNN_HTP,
        CPU_DIAGNOSTIC,
        NONE
    }

    public enum Blocker {
        NONE,
        ACCELERATED_PACK_MISSING,
        NATIVE_RUNTIME_MISSING,
        ACCELERATOR_UNAVAILABLE,
        INSUFFICIENT_RAM
    }

    public static final class Decision {
        public final boolean ready;
        public final Backend backend;
        public final Blocker blocker;
        public final String message;

        Decision(boolean ready, Backend backend, Blocker blocker, String message) {
            this.ready = ready;
            this.backend = backend;
            this.blocker = blocker;
            this.message = message;
        }

        public boolean isGpu() {
            return ready && backend == Backend.MNN_OPENCL;
        }

        public boolean isAccelerated() {
            return ready && (backend == Backend.MNN_OPENCL || backend == Backend.QNN_HTP);
        }
    }

    private MobileI2VRuntimeProbe() {}

    public static Decision decide(
            boolean acceleratedPackValid,
            boolean nativeRuntimeLoaded,
            boolean openClReady,
            boolean qnnHtpReady,
            long totalRamMb) {
        if (!acceleratedPackValid) {
            return blocked(Blocker.ACCELERATED_PACK_MISSING,
                    "仅有上游 .pth/旧模型包；尚未安装可执行 MobileI2V GPU 模型包");
        }
        if (totalRamMb > 0 && totalRamMb < 8192) {
            return blocked(Blocker.INSUFFICIENT_RAM,
                    "设备内存低于 MobileI2V 加速运行最低门槛 8 GB");
        }
        if (!nativeRuntimeLoaded) {
            return blocked(Blocker.NATIVE_RUNTIME_MISSING,
                    "MobileI2V GPU native runtime 尚未加载");
        }
        if (openClReady) {
            return new Decision(true, Backend.MNN_OPENCL, Blocker.NONE,
                    "MobileI2V · Adreno GPU · MNN OpenCL 已就绪");
        }
        if (qnnHtpReady) {
            return new Decision(true, Backend.QNN_HTP, Blocker.NONE,
                    "MobileI2V · Qualcomm HTP · QNN 已就绪");
        }
        return blocked(Blocker.ACCELERATOR_UNAVAILABLE,
                "加速模型包有效，但 GPU/HTP 均未就绪；不会自动回退 CPU");
    }

    public static Decision cpuDiagnosticOnly(String message) {
        return new Decision(false, Backend.CPU_DIAGNOSTIC,
                Blocker.ACCELERATOR_UNAVAILABLE,
                message == null || message.isEmpty()
                        ? "CPU 仅用于诊断，不允许作为 MobileI2V 生产生成后端"
                        : message);
    }

    private static Decision blocked(Blocker blocker, String message) {
        return new Decision(false, Backend.NONE, blocker, message);
    }
}
