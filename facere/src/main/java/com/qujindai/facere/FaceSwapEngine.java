package com.qujindai.facere;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.SystemClock;

import java.util.List;

public final class FaceSwapEngine {
    public interface Progress {
        void onProgress(int percent, String message);
    }

    public static final class Result {
        public final Bitmap bitmap;
        public final int sourceFaceCount;
        public final int targetFaceCount;
        public final long elapsedMs;

        Result(Bitmap bitmap, int sourceFaceCount, int targetFaceCount, long elapsedMs) {
            this.bitmap = bitmap;
            this.sourceFaceCount = sourceFaceCount;
            this.targetFaceCount = targetFaceCount;
            this.elapsedMs = elapsedMs;
        }
    }

    private final Context context;

    public FaceSwapEngine(Context context) {
        this.context = context.getApplicationContext();
    }

    public Result swap(Uri sourceUri, Uri targetUri, Progress progress) throws Exception {
        FaceModelFiles models = FaceModelStore.active(context);
        if (models == null) throw new IllegalStateException("请先导入有效的 Face-RE 模型包");
        long started = SystemClock.elapsedRealtime();
        update(progress, 3, "读取源人脸");
        Bitmap source = ImageLoader.load(context.getContentResolver(), sourceUri, 2048);
        update(progress, 7, "读取目标图片");
        Bitmap target = ImageLoader.load(context.getContentResolver(), targetUri, 4096);
        FaceAligner.Alignment sourceAlignment = null;
        FaceAligner.Alignment targetAlignment = null;
        Bitmap swapped = null;
        try {
            update(progress, 12, "SCRFD 检测源脸和目标脸");
            List<DetectedFace> sourceFaces;
            List<DetectedFace> targetFaces;
            try (FaceDetector detector = new FaceDetector(models)) {
                sourceFaces = detector.detect(source);
                targetFaces = detector.detect(target);
            }
            DetectedFace sourceFace = DetectedFace.largest(sourceFaces);
            DetectedFace targetFace = DetectedFace.largest(targetFaces);
            if (sourceFace == null) throw new IllegalStateException("源图片未检测到人脸");
            if (targetFace == null) throw new IllegalStateException("目标图片未检测到人脸");

            update(progress, 37, "ArcFace 提取源身份特征");
            sourceAlignment = FaceAligner.align112(source, sourceFace.landmarks);
            float[] embedding;
            try (FaceEmbedder embedder = new FaceEmbedder(models)) {
                embedding = embedder.embed(sourceAlignment.bitmap);
            }

            update(progress, 58, "对齐目标人脸");
            targetAlignment = FaceAligner.align128(target, targetFace.landmarks);
            update(progress, 66, "INSwapper 本地换脸推理");
            try (FaceSwapper swapper = new FaceSwapper(models)) {
                swapped = swapper.swap(targetAlignment.bitmap, embedding);
            }

            update(progress, 88, "融合回目标图片");
            Bitmap output = FaceBlender.blend(target, swapped, targetAlignment.forward);
            update(progress, 96, "换脸结果已生成");
            return new Result(output, sourceFaces.size(), targetFaces.size(),
                    SystemClock.elapsedRealtime() - started);
        } finally {
            if (sourceAlignment != null && !sourceAlignment.bitmap.isRecycled()) sourceAlignment.bitmap.recycle();
            if (targetAlignment != null && !targetAlignment.bitmap.isRecycled()) targetAlignment.bitmap.recycle();
            if (swapped != null && !swapped.isRecycled()) swapped.recycle();
            if (!source.isRecycled()) source.recycle();
            if (!target.isRecycled()) target.recycle();
        }
    }

    private static void update(Progress progress, int percent, String message) {
        if (progress != null) progress.onProgress(percent, message);
    }
}
