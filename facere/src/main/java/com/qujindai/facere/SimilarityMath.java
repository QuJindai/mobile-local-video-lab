package com.qujindai.facere;

public final class SimilarityMath {
    private SimilarityMath() {}

    public static float[] estimate(float[][] src, float[][] dst) {
        if (src == null || dst == null || src.length != dst.length || src.length < 2) {
            throw new IllegalArgumentException("point sets must have equal size >= 2");
        }
        int n = src.length;
        double srcCx = 0, srcCy = 0, dstCx = 0, dstCy = 0;
        for (int i = 0; i < n; i++) {
            validatePoint(src[i]);
            validatePoint(dst[i]);
            srcCx += src[i][0];
            srcCy += src[i][1];
            dstCx += dst[i][0];
            dstCy += dst[i][1];
        }
        srcCx /= n;
        srcCy /= n;
        dstCx /= n;
        dstCy /= n;

        double srcNorm = 0.0;
        double a = 0.0;
        double b = 0.0;
        for (int i = 0; i < n; i++) {
            double sx = src[i][0] - srcCx;
            double sy = src[i][1] - srcCy;
            double dx = dst[i][0] - dstCx;
            double dy = dst[i][1] - dstCy;
            srcNorm += sx * sx + sy * sy;
            a += sx * dx + sy * dy;
            b += sx * dy - sy * dx;
        }
        if (srcNorm < 1e-12) {
            throw new IllegalArgumentException("source landmarks are degenerate");
        }

        double am = a / srcNorm;
        double bm = -b / srcNorm;
        double dm = b / srcNorm;
        double em = a / srcNorm;
        double cm = dstCx - (am * srcCx + bm * srcCy);
        double fm = dstCy - (dm * srcCx + em * srcCy);

        float[] result = {(float) am, (float) bm, (float) cm,
                (float) dm, (float) em, (float) fm};
        for (float value : result) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("similarity transform is non-finite");
            }
        }
        return result;
    }

    private static void validatePoint(float[] point) {
        if (point == null || point.length != 2
                || !Float.isFinite(point[0]) || !Float.isFinite(point[1])) {
            throw new IllegalArgumentException("invalid 2D point");
        }
    }
}
