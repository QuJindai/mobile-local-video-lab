package com.qujindai.localvideo;

import java.io.File;

/** JNI bridge for the production MobileI2V accelerated runtime. */
public final class MobileI2VGpuNative {
    public static final int LATENT_CFG_FLOATS = 2 * 128 * 3 * 23 * 40;

    private static final boolean NATIVE_LOADED;
    private static final String NATIVE_ERROR;

    static {
        boolean loaded = false;
        String error = "";
        try {
            System.loadLibrary("mobilei2v_gpu");
            loaded = true;
        } catch (Throwable failure) {
            error = failure.getClass().getSimpleName()
                    + (failure.getMessage() == null ? "" : ": " + failure.getMessage());
        }
        NATIVE_LOADED = loaded;
        NATIVE_ERROR = error;
    }

    private MobileI2VGpuNative() {}

    public static boolean isNativeLoaded() {
        return NATIVE_LOADED;
    }

    public static String nativeError() {
        return NATIVE_ERROR;
    }

    public static Probe probe(InstalledModelPack pack) {
        if (pack == null || !pack.isAcceleratedMobileI2V()) {
            return new Probe(false, false, "加速 MobileI2V 模型包未安装");
        }
        if (!NATIVE_LOADED) {
            return new Probe(false, false, "GPU native runtime 加载失败 · " + NATIVE_ERROR);
        }
        String[] required = {
                "denoiser.mnn",
                "empty_prompt.f16",
                "empty_prompt_mask.bin"
        };
        for (String name : required) {
            File file = pack.artifact(name);
            if (!file.isFile()) return new Probe(false, false, name + " 缺失");
        }
        String result;
        try {
            result = nativeProbe(pack.root.getAbsolutePath());
        } catch (Throwable error) {
            return new Probe(false, false,
                    "MNN OpenCL 探测异常 · " + error.getClass().getSimpleName());
        }
        boolean ready = result != null && result.startsWith("MNN_OPENCL_READY:");
        boolean cache = ready && result.contains("cache-enabled");
        return new Probe(ready, cache, result == null ? "native probe returned null" : result);
    }

    public static Session load(InstalledModelPack pack) {
        Probe probe = probe(pack);
        if (!probe.openClReady) {
            throw new IllegalStateException(probe.message);
        }
        long handle = nativeLoad(pack.root.getAbsolutePath());
        if (handle == 0L) {
            throw new IllegalStateException("MNN OpenCL denoiser session load failed");
        }
        return new Session(handle);
    }

    public static final class Probe {
        public final boolean openClReady;
        public final boolean tuningCacheEnabled;
        public final String message;

        Probe(boolean openClReady, boolean tuningCacheEnabled, String message) {
            this.openClReady = openClReady;
            this.tuningCacheEnabled = tuningCacheEnabled;
            this.message = message;
        }
    }

    public static final class Session implements AutoCloseable {
        private long handle;

        Session(long handle) {
            this.handle = handle;
        }

        /**
         * Prompt embeddings, text attention mask and guide conditioning mask are
         * read once from the verified pack by native code, avoiding multi-MB
         * Java→JNI transfers for every denoising step.
         */
        public synchronized void runDenoiser(
                float[] latentCfg2,
                float timestep,
                float flowScore,
                float[] outputCfg2) {
            if (handle == 0L) throw new IllegalStateException("GPU session already closed");
            requireLength(latentCfg2, LATENT_CFG_FLOATS, "latentCfg2");
            requireLength(outputCfg2, LATENT_CFG_FLOATS, "outputCfg2");
            float[] flowScoreCfg2 = {flowScore, flowScore};
            int code = nativeRunDenoiser(
                    handle, latentCfg2, timestep, flowScoreCfg2, outputCfg2);
            if (code != 0) throw new IllegalStateException("GPU denoiser failed · code=" + code);
        }

        @Override
        public synchronized void close() {
            if (handle != 0L) {
                nativeRelease(handle);
                handle = 0L;
            }
        }
    }

    private static void requireLength(float[] values, int expected, String name) {
        if (values == null || values.length != expected) {
            throw new IllegalArgumentException(name + " must contain " + expected + " floats");
        }
    }

    private static native String nativeProbe(String modelDir);
    private static native long nativeLoad(String modelDir);
    private static native int nativeRunDenoiser(
            long handle,
            float[] latentCfg2,
            float timestep,
            float[] flowScoreCfg2,
            float[] outputCfg2);
    private static native void nativeRelease(long handle);
}
