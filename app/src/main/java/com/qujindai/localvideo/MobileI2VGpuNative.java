package com.qujindai.localvideo;

import java.io.File;

/** JNI bridge for the production MobileI2V Adreno/MNN OpenCL runtime. */
public final class MobileI2VGpuNative {
    public static final int WIDTH = 1280;
    public static final int HEIGHT = 720;
    public static final int OUTPUT_FRAMES = 17;
    public static final int LATENT_CHANNELS = 128;
    public static final int LATENT_FRAMES = 3;
    public static final int LATENT_HEIGHT = 23;
    public static final int LATENT_WIDTH = 40;
    public static final int SINGLE_LATENT_FLOATS = LATENT_CHANNELS * LATENT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH;
    public static final int GUIDE_LATENT_FLOATS = LATENT_CHANNELS * LATENT_HEIGHT * LATENT_WIDTH;
    public static final int LATENT_CFG_FLOATS = 2 * SINGLE_LATENT_FLOATS;
    public static final int ENCODER_INPUT_FLOATS = 3 * HEIGHT * WIDTH;
    public static final int FRAME_ARGB_PIXELS = HEIGHT * WIDTH;

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

    public static boolean isNativeLoaded() { return NATIVE_LOADED; }
    public static String nativeError() { return NATIVE_ERROR; }

    public static Probe probe(InstalledModelPack pack) {
        if (pack == null || !pack.isAcceleratedMobileI2V()) {
            return new Probe(false, false, "加速 MobileI2V 模型包未安装");
        }
        if (!NATIVE_LOADED) {
            return new Probe(false, false, "GPU native runtime 加载失败 · " + NATIVE_ERROR);
        }
        String[] required = {
                "denoiser.mnn", "vae_encoder.mnn", "vae_decoder.mnn",
                "empty_prompt.f16", "empty_prompt_mask.bin"
        };
        for (String name : required) {
            if (!pack.artifact(name).isFile()) return new Probe(false, false, name + " 缺失");
        }
        try {
            String result = nativeProbe(pack.root.getAbsolutePath());
            boolean ready = result != null && result.startsWith("MNN_OPENCL_READY:");
            boolean cache = ready && result.contains("cache-enabled");
            return new Probe(ready, cache, result == null ? "native probe returned null" : result);
        } catch (Throwable error) {
            return new Probe(false, false, "MNN OpenCL 探测异常 · " + error.getClass().getSimpleName());
        }
    }

    public static Session load(InstalledModelPack pack) {
        Probe probe = probe(pack);
        if (!probe.openClReady) throw new IllegalStateException(probe.message);
        long handle = nativeLoad(pack.root.getAbsolutePath());
        if (handle == 0L) throw new IllegalStateException("MNN OpenCL MobileI2V session load failed");
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
        Session(long handle) { this.handle = handle; }

        public synchronized void encode(float[] imageNchw, float[] guideLatent) {
            requireOpen();
            requireLength(imageNchw, ENCODER_INPUT_FLOATS, "imageNchw");
            requireLength(guideLatent, GUIDE_LATENT_FLOATS, "guideLatent");
            int code = nativeEncode(handle, imageNchw, guideLatent);
            if (code != 0) throw new IllegalStateException("GPU VAE encoder failed · code=" + code);
        }

        public synchronized void runDenoiser(
                float[] latentCfg2, float timestep, float flowScore, float[] outputCfg2) {
            requireOpen();
            requireLength(latentCfg2, LATENT_CFG_FLOATS, "latentCfg2");
            requireLength(outputCfg2, LATENT_CFG_FLOATS, "outputCfg2");
            float[] flow = {flowScore, flowScore};
            int code = nativeRunDenoiser(handle, latentCfg2, timestep, flow, outputCfg2);
            if (code != 0) throw new IllegalStateException("GPU denoiser failed · code=" + code);
        }

        public synchronized void decode(float[] latent) {
            requireOpen();
            requireLength(latent, SINGLE_LATENT_FLOATS, "latent");
            int code = nativeDecode(handle, latent);
            if (code != 0) throw new IllegalStateException("GPU VAE decoder failed · code=" + code);
        }

        public synchronized void copyDecodedFrameArgb(int frame, int[] argb) {
            requireOpen();
            if (frame < 0 || frame >= OUTPUT_FRAMES) throw new IllegalArgumentException("invalid frame " + frame);
            if (argb == null || argb.length != FRAME_ARGB_PIXELS) {
                throw new IllegalArgumentException("argb must contain " + FRAME_ARGB_PIXELS + " pixels");
            }
            int code = nativeCopyDecodedFrameArgb(handle, frame, argb);
            if (code != 0) throw new IllegalStateException("decoded frame copy failed · code=" + code);
        }

        public synchronized void clearDecoded() {
            if (handle != 0L) nativeClearDecoded(handle);
        }

        private void requireOpen() {
            if (handle == 0L) throw new IllegalStateException("GPU session already closed");
        }

        @Override public synchronized void close() {
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
    private static native int nativeEncode(long handle, float[] imageNchw, float[] guideLatent);
    private static native int nativeRunDenoiser(long handle, float[] latentCfg2, float timestep,
                                                 float[] flowScoreCfg2, float[] outputCfg2);
    private static native int nativeDecode(long handle, float[] latent);
    private static native int nativeCopyDecodedFrameArgb(long handle, int frame, int[] argb);
    private static native void nativeClearDecoded(long handle);
    private static native void nativeRelease(long handle);
}
