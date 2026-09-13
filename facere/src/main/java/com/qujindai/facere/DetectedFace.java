package com.qujindai.facere;

import android.graphics.RectF;

import java.util.List;

public final class DetectedFace {
    public final RectF bbox;
    public final float[] landmarks;
    public final float score;

    public DetectedFace(RectF bbox, float[] landmarks, float score) {
        this.bbox = new RectF(bbox);
        this.landmarks = landmarks.clone();
        this.score = score;
    }

    public float area() {
        return Math.max(0f, bbox.width()) * Math.max(0f, bbox.height());
    }

    public static DetectedFace largest(List<DetectedFace> faces) {
        if (faces == null || faces.isEmpty()) return null;
        DetectedFace best = faces.get(0);
        for (int i = 1; i < faces.size(); i++) {
            DetectedFace candidate = faces.get(i);
            if (candidate.area() > best.area()) best = candidate;
        }
        return best;
    }
}
