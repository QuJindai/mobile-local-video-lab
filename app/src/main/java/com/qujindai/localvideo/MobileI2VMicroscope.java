package com.qujindai.localvideo;

import java.util.Locale;

/** Immutable evidence record for one accelerated MobileI2V run. */
public final class MobileI2VMicroscope {
    public final String packId;
    public final String packVersion;
    public final String sourceCommit;
    public final String checkpointSha256;
    public final String dreamCommit;
    public final String mnnCommit;
    public final MobileI2VRuntimeProbe.Backend requestedBackend;
    public final MobileI2VRuntimeProbe.Backend actualBackend;
    public final boolean openClReady;
    public final boolean tuningCacheHit;
    public final String vaeImpl;
    public final int steps;
    public final int frames;
    public final int width;
    public final int height;
    public final long encodeMs;
    public final long denoiseMs;
    public final long decodeMs;
    public final long mp4Ms;
    public final long javaHeapBeforeMb;
    public final long javaHeapPeakMb;
    public final long nativeHeapBeforeMb;
    public final long nativeHeapPeakMb;
    public final int thermalBefore;
    public final int thermalAfter;
    public final boolean cpuProductionPathUsed;

    public MobileI2VMicroscope(
            String packId,
            String packVersion,
            String sourceCommit,
            String checkpointSha256,
            String dreamCommit,
            String mnnCommit,
            MobileI2VRuntimeProbe.Backend requestedBackend,
            MobileI2VRuntimeProbe.Backend actualBackend,
            boolean openClReady,
            boolean tuningCacheHit,
            String vaeImpl,
            int steps,
            int frames,
            int width,
            int height,
            long encodeMs,
            long denoiseMs,
            long decodeMs,
            long mp4Ms,
            long javaHeapBeforeMb,
            long javaHeapPeakMb,
            long nativeHeapBeforeMb,
            long nativeHeapPeakMb,
            int thermalBefore,
            int thermalAfter,
            boolean cpuProductionPathUsed) {
        this.packId = safe(packId);
        this.packVersion = safe(packVersion);
        this.sourceCommit = safe(sourceCommit);
        this.checkpointSha256 = safe(checkpointSha256);
        this.dreamCommit = safe(dreamCommit);
        this.mnnCommit = safe(mnnCommit);
        this.requestedBackend = requestedBackend == null
                ? MobileI2VRuntimeProbe.Backend.NONE : requestedBackend;
        this.actualBackend = actualBackend == null
                ? MobileI2VRuntimeProbe.Backend.NONE : actualBackend;
        this.openClReady = openClReady;
        this.tuningCacheHit = tuningCacheHit;
        this.vaeImpl = safe(vaeImpl);
        this.steps = steps;
        this.frames = frames;
        this.width = width;
        this.height = height;
        this.encodeMs = nonNegative(encodeMs);
        this.denoiseMs = nonNegative(denoiseMs);
        this.decodeMs = nonNegative(decodeMs);
        this.mp4Ms = nonNegative(mp4Ms);
        this.javaHeapBeforeMb = nonNegative(javaHeapBeforeMb);
        this.javaHeapPeakMb = nonNegative(javaHeapPeakMb);
        this.nativeHeapBeforeMb = nonNegative(nativeHeapBeforeMb);
        this.nativeHeapPeakMb = nonNegative(nativeHeapPeakMb);
        this.thermalBefore = thermalBefore;
        this.thermalAfter = thermalAfter;
        this.cpuProductionPathUsed = cpuProductionPathUsed;
    }

    public boolean acceleratedEvidenceValid() {
        return !cpuProductionPathUsed
                && (actualBackend == MobileI2VRuntimeProbe.Backend.MNN_OPENCL
                || actualBackend == MobileI2VRuntimeProbe.Backend.QNN_HTP);
    }

    public String format() {
        return String.format(Locale.US,
                "MobileI2V 显微镜\n"
                        + "模型包: %s · %s\n"
                        + "MobileI2V: %s\n"
                        + "checkpoint: %s\n"
                        + "Dream baseline: %s\n"
                        + "MNN: %s\n"
                        + "请求后端: %s\n"
                        + "实际后端: %s\n"
                        + "OpenCL: %s · tuning cache: %s\n"
                        + "VAE: %s\n"
                        + "生成契约: %dx%d · %d 帧 · %d steps\n"
                        + "耗时(ms): encode=%d · denoise=%d · decode=%d · mp4=%d\n"
                        + "Java heap(MB): %d → peak %d\n"
                        + "Native heap(MB): %d → peak %d\n"
                        + "热状态: %d → %d\n"
                        + "CPU生产路径: %s\n"
                        + "加速证据: %s",
                empty(packId), empty(packVersion), shortSha(sourceCommit), shortSha(checkpointSha256),
                shortSha(dreamCommit), shortSha(mnnCommit),
                requestedBackend.name(), actualBackend.name(),
                openClReady ? "READY" : "NO", tuningCacheHit ? "HIT" : "MISS",
                empty(vaeImpl), width, height, frames, steps,
                encodeMs, denoiseMs, decodeMs, mp4Ms,
                javaHeapBeforeMb, javaHeapPeakMb,
                nativeHeapBeforeMb, nativeHeapPeakMb,
                thermalBefore, thermalAfter,
                cpuProductionPathUsed ? "USED · FAIL" : "NO",
                acceleratedEvidenceValid() ? "PASS" : "FAIL");
    }

    private static long nonNegative(long value) {
        return Math.max(0L, value);
    }

    private static String safe(String value) {
        return value == null ? "" : value.trim();
    }

    private static String empty(String value) {
        return value == null || value.isEmpty() ? "unknown" : value;
    }

    private static String shortSha(String value) {
        if (value == null || value.isEmpty()) return "unknown";
        return value.length() <= 12 ? value : value.substring(0, 12);
    }
}
