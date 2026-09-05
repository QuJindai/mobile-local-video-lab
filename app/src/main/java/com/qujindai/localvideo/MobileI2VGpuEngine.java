package com.qujindai.localvideo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Debug;
import android.os.PowerManager;
import android.os.SystemClock;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Properties;
import java.util.Random;

/** Full MobileI2V production pipeline. Neural inference is MNN OpenCL only. */
public final class MobileI2VGpuEngine {
    public static final int STEPS = 28;
    public static final float CFG_SCALE = 4.5f;
    public static final float FLOW_SCORE = 2.0f;

    public interface ProgressListener { void onProgress(int percent, String message); }

    public static final class Result {
        public final Uri uri;
        public final int frames;
        public final int width;
        public final int height;
        public final int fps;
        public final long elapsedMs;
        public final MobileI2VMicroscope microscope;
        Result(Uri uri, int fps, long elapsedMs, MobileI2VMicroscope microscope) {
            this.uri = uri;
            this.frames = MobileI2VGpuNative.OUTPUT_FRAMES;
            this.width = MobileI2VGpuNative.WIDTH;
            this.height = MobileI2VGpuNative.HEIGHT;
            this.fps = fps;
            this.elapsedMs = elapsedMs;
            this.microscope = microscope;
        }
    }

    private final Context context;
    public MobileI2VGpuEngine(Context context) { this.context = context.getApplicationContext(); }

    public Result generate(Uri imageUri, InstalledModelPack pack, int fps, ProgressListener listener)
            throws Exception {
        if (imageUri == null) throw new IllegalArgumentException("imageUri is required");
        if (pack == null || !pack.isAcceleratedMobileI2V()) {
            throw new IllegalStateException("可执行 MobileI2V GPU 模型包未安装");
        }
        MobileI2VGpuNative.Probe probe = MobileI2VGpuNative.probe(pack);
        if (!probe.openClReady) throw new IllegalStateException(probe.message);

        final long started = SystemClock.elapsedRealtime();
        final long javaBefore = javaHeapMb();
        final long nativeBefore = nativeHeapMb();
        final int thermalBefore = thermalStatus();
        long javaPeak = javaBefore;
        long nativePeak = nativeBefore;
        long encodeMs;
        long denoiseMs;
        long decodeMs;
        long mp4Ms;

        File job = new File(context.getCacheDir(), "mobilei2v-" + System.nanoTime());
        File framesDir = new File(job, "frames");
        if (!framesDir.mkdirs()) throw new IllegalStateException("cannot create MobileI2V frame directory");
        try (MobileI2VGpuNative.Session session = MobileI2VGpuNative.load(pack)) {
            progress(listener, 2, "读取主图 · 1280×720");
            float[] image = loadImageNchw(imageUri);
            float[] guide = new float[MobileI2VGpuNative.GUIDE_LATENT_FLOATS];

            long stage = SystemClock.elapsedRealtime();
            progress(listener, 6, "GPU VAE encode · MNN OpenCL");
            session.encode(image, guide);
            encodeMs = SystemClock.elapsedRealtime() - stage;
            image = null;
            javaPeak = Math.max(javaPeak, javaHeapMb());
            nativePeak = Math.max(nativePeak, nativeHeapMb());

            float[] initial = gaussianLatent(1L);
            MobileI2VFlowEuler.lockGuideFirstSlice(
                    initial, guide,
                    MobileI2VGpuNative.LATENT_CHANNELS,
                    MobileI2VGpuNative.LATENT_FRAMES,
                    MobileI2VGpuNative.LATENT_HEIGHT,
                    MobileI2VGpuNative.LATENT_WIDTH);

            stage = SystemClock.elapsedRealtime();
            progress(listener, 12, "MobileI2V · Adreno GPU 去噪 0/" + STEPS);
            float[] sampled = MobileI2VFlowEuler.sample(
                    initial,
                    guide,
                    MobileI2VGpuNative.LATENT_CHANNELS,
                    MobileI2VGpuNative.LATENT_FRAMES,
                    MobileI2VGpuNative.LATENT_HEIGHT,
                    MobileI2VGpuNative.LATENT_WIDTH,
                    STEPS,
                    CFG_SCALE,
                    (cfg2, timestep, out) -> session.runDenoiser(cfg2, timestep, FLOW_SCORE, out),
                    (completed, total, timestep) -> progress(listener,
                            12 + completed * 55 / total,
                            "MobileI2V · Adreno GPU 去噪 " + completed + "/" + total));
            denoiseMs = SystemClock.elapsedRealtime() - stage;
            initial = null;
            guide = null;
            javaPeak = Math.max(javaPeak, javaHeapMb());
            nativePeak = Math.max(nativePeak, nativeHeapMb());

            stage = SystemClock.elapsedRealtime();
            progress(listener, 70, "GPU VAE decode · 17 帧");
            session.decode(sampled);
            decodeMs = SystemClock.elapsedRealtime() - stage;
            sampled = null;
            javaPeak = Math.max(javaPeak, javaHeapMb());
            nativePeak = Math.max(nativePeak, nativeHeapMb());

            progress(listener, 77, "逐帧取回 · GPU 解码结果");
            List<File> frameFiles = new ArrayList<>(MobileI2VGpuNative.OUTPUT_FRAMES);
            int[] argb = new int[MobileI2VGpuNative.FRAME_ARGB_PIXELS];
            for (int frame = 0; frame < MobileI2VGpuNative.OUTPUT_FRAMES; frame++) {
                session.copyDecodedFrameArgb(frame, argb);
                Bitmap bitmap = Bitmap.createBitmap(argb, MobileI2VGpuNative.WIDTH,
                        MobileI2VGpuNative.HEIGHT, Bitmap.Config.ARGB_8888);
                File file = new File(framesDir, String.format(Locale.US, "%03d.png", frame));
                try (FileOutputStream out = new FileOutputStream(file)) {
                    if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)) {
                        throw new IllegalStateException("frame PNG encode failed: " + frame);
                    }
                } finally {
                    bitmap.recycle();
                }
                frameFiles.add(file);
                progress(listener, 77 + (frame + 1) * 7 / MobileI2VGpuNative.OUTPUT_FRAMES,
                        "帧缓存 " + (frame + 1) + "/" + MobileI2VGpuNative.OUTPUT_FRAMES);
            }
            session.clearDecoded();

            stage = SystemClock.elapsedRealtime();
            progress(listener, 85, "H.264 / MediaCodec 输出 MP4");
            File mp4 = new File(job, "mobilei2v-gpu.mp4");
            Mp4Encoder.encode(frameFiles, mp4, MobileI2VGpuNative.WIDTH, MobileI2VGpuNative.HEIGHT,
                    fps, (encoded, total) -> progress(listener,
                            85 + encoded * 10 / Math.max(1, total),
                            "MP4 " + encoded + "/" + total));
            mp4Ms = SystemClock.elapsedRealtime() - stage;

            progress(listener, 96, "写入 Movies/LocalVideoLab");
            String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            Uri uri = MediaStorePublisher.publish(context, mp4,
                    "mobilei2v_gpu_" + timestamp + ".mp4");

            javaPeak = Math.max(javaPeak, javaHeapMb());
            nativePeak = Math.max(nativePeak, nativeHeapMb());
            int thermalAfter = thermalStatus();
            AcceleratedPackManifest a = pack.acceleratedManifest;
            MobileI2VMicroscope microscope = new MobileI2VMicroscope(
                    a.id, a.version, a.sourceCommit, a.checkpointSha256,
                    a.dreamCommit, a.mnnCommit,
                    MobileI2VRuntimeProbe.Backend.MNN_OPENCL,
                    MobileI2VRuntimeProbe.Backend.MNN_OPENCL,
                    true, probe.tuningCacheEnabled,
                    vaeImpl(pack), STEPS, MobileI2VGpuNative.OUTPUT_FRAMES,
                    MobileI2VGpuNative.WIDTH, MobileI2VGpuNative.HEIGHT,
                    encodeMs, denoiseMs, decodeMs, mp4Ms,
                    javaBefore, javaPeak, nativeBefore, nativePeak,
                    thermalBefore, thermalAfter, false);
            if (!microscope.acceleratedEvidenceValid()) {
                throw new IllegalStateException("MobileI2V 加速证据无效");
            }
            progress(listener, 100, "MobileI2V GPU 完成");
            return new Result(uri, fps, SystemClock.elapsedRealtime() - started, microscope);
        } finally {
            deleteRecursively(job);
        }
    }

    private float[] loadImageNchw(Uri uri) throws Exception {
        Bitmap raw;
        try (InputStream in = context.getContentResolver().openInputStream(uri)) {
            if (in == null) throw new IllegalStateException("cannot open input image");
            raw = BitmapFactory.decodeStream(in);
        }
        if (raw == null) throw new IllegalStateException("cannot decode input image");
        Bitmap scaled = raw.getWidth() == MobileI2VGpuNative.WIDTH && raw.getHeight() == MobileI2VGpuNative.HEIGHT
                ? raw : Bitmap.createScaledBitmap(raw, MobileI2VGpuNative.WIDTH, MobileI2VGpuNative.HEIGHT, true);
        if (scaled != raw) raw.recycle();
        int pixels = MobileI2VGpuNative.FRAME_ARGB_PIXELS;
        int[] argb = new int[pixels];
        scaled.getPixels(argb, 0, MobileI2VGpuNative.WIDTH, 0, 0,
                MobileI2VGpuNative.WIDTH, MobileI2VGpuNative.HEIGHT);
        scaled.recycle();
        float[] out = new float[MobileI2VGpuNative.ENCODER_INPUT_FLOATS];
        int g = pixels, b = pixels * 2;
        for (int i = 0; i < pixels; i++) {
            int p = argb[i];
            out[i] = (((p >>> 16) & 255) / 127.5f) - 1f;
            out[g + i] = (((p >>> 8) & 255) / 127.5f) - 1f;
            out[b + i] = ((p & 255) / 127.5f) - 1f;
        }
        return out;
    }

    private static float[] gaussianLatent(long seed) {
        float[] latent = new float[MobileI2VGpuNative.SINGLE_LATENT_FLOATS];
        Random random = new Random(seed);
        for (int i = 0; i < latent.length; i++) latent[i] = (float) random.nextGaussian();
        return latent;
    }

    private static String vaeImpl(InstalledModelPack pack) {
        File file = pack.artifact("runtime.properties");
        if (!file.isFile()) return "LTX-Video";
        Properties p = new Properties();
        try (FileInputStream in = new FileInputStream(file)) {
            p.load(in);
            return p.getProperty("vae.impl", "LTX-Video");
        } catch (Exception ignored) {
            return "LTX-Video";
        }
    }

    private int thermalStatus() {
        try {
            PowerManager pm = context.getSystemService(PowerManager.class);
            return pm == null ? -1 : pm.getCurrentThermalStatus();
        } catch (Throwable ignored) { return -1; }
    }

    private static long javaHeapMb() {
        Runtime r = Runtime.getRuntime();
        return Math.max(0L, (r.totalMemory() - r.freeMemory()) / 1048576L);
    }
    private static long nativeHeapMb() { return Math.max(0L, Debug.getNativeHeapAllocatedSize() / 1048576L); }
    private static void progress(ProgressListener l, int p, String m) { if (l != null) l.onProgress(p, m); }
    private static void deleteRecursively(File f) {
        if (f == null || !f.exists()) return;
        if (f.isDirectory()) {
            File[] children = f.listFiles();
            if (children != null) for (File c : children) deleteRecursively(c);
        }
        //noinspection ResultOfMethodCallIgnored
        f.delete();
    }
}
