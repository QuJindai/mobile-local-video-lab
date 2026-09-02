package com.qujindai.localvideo;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.SystemClock;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.SimpleDateFormat;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public final class RifeEngine {
    public interface ProgressListener {
        void onProgress(int percent, String message);
    }

    public static final class Result {
        public final Uri uri;
        public final int frames;
        public final int width;
        public final int height;
        public final int fps;
        public final long elapsedMs;
        public final boolean singleImageMode;

        Result(Uri uri, GenerationPlan plan, long elapsedMs, boolean singleImageMode) {
            this.uri = uri;
            this.frames = plan.getFrames();
            this.width = plan.getWidth();
            this.height = plan.getHeight();
            this.fps = plan.getFps();
            this.elapsedMs = elapsedMs;
            this.singleImageMode = singleImageMode;
        }
    }

    private final Context context;

    public RifeEngine(Context context) {
        this.context = context.getApplicationContext();
    }

    public Result generate(Uri primaryUri, Uri secondaryUri, int frameCount, int fps,
                           ProgressListener listener) throws Exception {
        long started = SystemClock.elapsedRealtime();
        progress(listener, 2, "校验本地 RIFE/ncnn/Vulkan 运行时");
        RuntimeBundle runtime = RuntimeBundle.installAndVerify(context);

        progress(listener, 7, "读取并缩放输入图像");
        Bitmap primary = ImagePrep.loadScaled(context.getContentResolver(), primaryUri);
        Bitmap secondary = null;
        boolean singleImageMode = secondaryUri == null;
        File jobDir = new File(context.getCacheDir(), "rife-job-" + SystemClock.elapsedRealtime());
        try {
            if (secondaryUri != null) {
                Bitmap decoded = ImagePrep.loadScaled(context.getContentResolver(), secondaryUri);
                if (decoded.getWidth() != primary.getWidth() || decoded.getHeight() != primary.getHeight()) {
                    secondary = Bitmap.createScaledBitmap(decoded, primary.getWidth(), primary.getHeight(), true);
                    decoded.recycle();
                } else {
                    secondary = decoded;
                }
            } else {
                secondary = MotionEndpoint.create(primary);
            }

            GenerationPlan plan = new GenerationPlan(
                    primary.getWidth(), primary.getHeight(), frameCount, fps);
            File inputDir = new File(jobDir, "input");
            File framesDir = new File(jobDir, "frames");
            if (!inputDir.mkdirs() || !framesDir.mkdirs()) {
                throw new IOException("cannot create generation workspace");
            }
            ImagePrep.writePng(primary, new File(inputDir, "00000001.png"));
            ImagePrep.writePng(secondary, new File(inputDir, "00000002.png"));

            progress(listener, 12, singleImageMode
                    ? "已构造本地运动端点，启动 RIFE 神经插帧"
                    : "启动 RIFE 双图神经插帧");
            List<String> command = RifeCommand.build(
                    runtime.getExecutable().getAbsolutePath(),
                    inputDir.getAbsolutePath(),
                    framesDir.getAbsolutePath(),
                    frameCount,
                    runtime.getModelDir().getAbsolutePath());

            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.directory(jobDir);
            processBuilder.redirectErrorStream(true);
            Process process = processBuilder.start();
            ArrayDeque<String> tail = new ArrayDeque<>();
            int completed = 0;
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (tail.size() == 20) tail.removeFirst();
                    tail.addLast(line);
                    if (line.contains(" done")) {
                        completed++;
                        int percent = 12 + Math.min(58, completed * 58 / frameCount);
                        progress(listener, percent,
                                "RIFE 推理 " + Math.min(completed, frameCount) + "/" + frameCount);
                    }
                }
            }
            int exit = process.waitFor();
            if (exit != 0) {
                throw new IOException("RIFE exited with " + exit + ": " + String.join(" | ", tail));
            }

            List<File> frameFiles = listFrames(framesDir);
            if (frameFiles.size() != frameCount) {
                throw new IOException("RIFE frame count mismatch: expected " + frameCount
                        + ", got " + frameFiles.size());
            }

            progress(listener, 72, "H.264 编码 MP4");
            File mp4 = new File(jobDir, "local-video.mp4");
            Mp4Encoder.encode(frameFiles, mp4, plan.getWidth(), plan.getHeight(), plan.getFps(),
                    (encoded, total) -> progress(listener,
                            72 + encoded * 20 / Math.max(1, total),
                            "MP4 编码 " + encoded + "/" + total));

            progress(listener, 94, "写入系统相册 Movies/LocalVideoLab");
            String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            Uri uri = MediaStorePublisher.publish(context, mp4,
                    "local_video_" + timestamp + ".mp4");
            progress(listener, 100, "完成");
            return new Result(uri, plan, SystemClock.elapsedRealtime() - started, singleImageMode);
        } finally {
            if (secondary != null && !secondary.isRecycled()) secondary.recycle();
            if (!primary.isRecycled()) primary.recycle();
            deleteRecursively(jobDir);
        }
    }

    private static List<File> listFrames(File directory) throws IOException {
        File[] files = directory.listFiles((dir, name) -> name.toLowerCase(Locale.US).endsWith(".png"));
        if (files == null) throw new IOException("cannot list RIFE output frames");
        Arrays.sort(files, Comparator.comparing(File::getName));
        return new ArrayList<>(Arrays.asList(files));
    }

    private static void progress(ProgressListener listener, int percent, String message) {
        if (listener != null) listener.onProgress(percent, message);
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) deleteRecursively(child);
            }
        }
        //noinspection ResultOfMethodCallIgnored
        file.delete();
    }
}
