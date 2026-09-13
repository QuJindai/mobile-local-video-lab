package com.qujindai.localvideo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Android-independent face geometry and model math. Inputs are never modified.
 * Malformed, nonfinite or numerically degenerate inputs throw IllegalArgumentException.
 * Affines are row-major 2-by-3 matrices mapping pixel-center coordinates.
 */
public final class FaceSwapMath {
    private static final int IDENTITY_SIZE = 512;
    private static final int MAX_DETECTOR_SIZE = 1024;
    private static final int MAX_NMS_CANDIDATES = 2048;
    private static final double NMS_IOU = .4;
    private static final double CONDITION_EPSILON = 1e-12;
    private static final double[][] ARCFACE_112 = {
            {38.2946, 51.6963}, {73.5318, 51.5014}, {56.0252, 71.7366},
            {41.5493, 92.3655}, {70.7299, 92.2041}
    };

    private FaceSwapMath() {}

    /**
     * A detection in original image coordinates, with continuous (not inclusive) box
     * edges. Fields cannot be reassigned. Landmarks are deep-copied on construction;
     * the public array is a caller-owned snapshot and should be treated as read-only.
     * Java arrays themselves cannot be immutable; no internal detector state aliases it.
     */
    public static final class Detection {
        public final float x1, y1, x2, y2, score;
        public final float[][] landmarks;

        public Detection(float x1, float y1, float x2, float y2, float score, float[][] landmarks) {
            requireFinite(x1, "x1");
            requireFinite(y1, "y1");
            requireFinite(x2, "x2");
            requireFinite(y2, "y2");
            requireFinite(score, "score");
            if (!(x2 > x1) || !(y2 > y1) || score < 0 || score > 1) {
                throw new IllegalArgumentException("invalid detection box or score");
            }
            validateLandmarks(landmarks);
            if (!hasLandmarkSpread(landmarks)) {
                throw new IllegalArgumentException("degenerate detection landmarks");
            }
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.score = score;
            this.landmarks = new float[5][2];
            for (int i = 0; i < 5; i++) {
                this.landmarks[i][0] = landmarks[i][0];
                this.landmarks[i][1] = landmarks[i][1];
            }
        }
    }

    /**
     * Decodes SCRFD's nine flattened tensors: scores at strides 8/16/32, then boxes
     * at those strides, then five (x,y) landmark offsets. Each row-major cell has two
     * adjacent anchors centered at (col*stride,row*stride); distances/offsets are in
     * stride units. Scores must already be probabilities, not logits.
     *
     * The resized image occupies the TOP LEFT of a padded square. Source x and y are
     * scaled independently by sourceWidth/resizedWidth and sourceHeight/resizedHeight.
     * Anchors, box centers or landmarks in padding/outside the resized image are
     * discarded; valid edge-crossing boxes are clipped to the image. Negative/empty
     * boxes and degenerate landmarks are discarded. Invalid sizes, shapes or any
     * nonfinite tensor entry (including below threshold) throw IllegalArgumentException.
     *
     * Input size must be a multiple of 32 in [32,1024] (at most 43,008 anchors).
     * Keeps at most the highest-scoring 2,048 valid candidates before continuous-area
     * IoU > .4 NMS, bounding memory and quadratic work. Returns an unmodifiable list
     * in descending score order; ties retain stride/cell/anchor traversal order.
     */
    public static List<Detection> decodeScrfd(float[][] outputs, int inputSize,
            int sourceWidth, int sourceHeight, int resizedWidth, int resizedHeight, float threshold) {
        if (inputSize < 32 || inputSize > MAX_DETECTOR_SIZE || inputSize % 32 != 0
                || sourceWidth <= 0 || sourceHeight <= 0
                || resizedWidth <= 0 || resizedWidth > inputSize
                || resizedHeight <= 0 || resizedHeight > inputSize) {
            throw new IllegalArgumentException("invalid SCRFD image dimensions");
        }
        requireFinite(threshold, "score threshold");
        if (threshold < 0 || threshold > 1 || outputs == null || outputs.length != 9) {
            throw new IllegalArgumentException("SCRFD requires nine tensors and threshold in [0,1]");
        }
        for (int level = 0; level < 3; level++) {
            int side = inputSize / (8 << level);
            int count = side * side * 2;
            validateTensor(outputs[level], count, true);
            validateTensor(outputs[level + 3], count * 4, false);
            validateTensor(outputs[level + 6], count * 10, false);
        }
        double scaleX = (double) sourceWidth / resizedWidth;
        double scaleY = (double) sourceHeight / resizedHeight;
        PriorityQueue<Candidate> best = new PriorityQueue<>(MAX_NMS_CANDIDATES, WORST_FIRST);
        int order = 0;
        for (int level = 0; level < 3; level++) {
            int stride = 8 << level;
            int side = inputSize / stride;
            float[] scores = outputs[level];
            float[] boxes = outputs[level + 3];
            float[] kps = outputs[level + 6];
            for (int anchor = 0; anchor < scores.length; anchor++, order++) {
                float score = scores[anchor];
                if (score < threshold || (best.size() == MAX_NMS_CANDIDATES
                        && score <= best.peek().detection.score)) continue;
                int cell = anchor / 2;
                int cx = (cell % side) * stride, cy = (cell / side) * stride;
                if (cx >= resizedWidth || cy >= resizedHeight) continue;
                int box = anchor * 4;
                if (boxes[box] < 0 || boxes[box + 1] < 0
                        || boxes[box + 2] < 0 || boxes[box + 3] < 0) continue;
                double left = cx - (double) boxes[box] * stride;
                double top = cy - (double) boxes[box + 1] * stride;
                double right = cx + (double) boxes[box + 2] * stride;
                double bottom = cy + (double) boxes[box + 3] * stride;
                double midX = (left + right) * .5, midY = (top + bottom) * .5;
                if (midX < 0 || midX >= resizedWidth || midY < 0 || midY >= resizedHeight) continue;
                float x1 = (float) (Math.max(0, left) * scaleX);
                float y1 = (float) (Math.max(0, top) * scaleY);
                float x2 = (float) (Math.min(resizedWidth, right) * scaleX);
                float y2 = (float) (Math.min(resizedHeight, bottom) * scaleY);
                if (!(x2 > x1) || !(y2 > y1)) continue;
                float[][] landmarks = new float[5][2];
                boolean valid = true;
                for (int point = 0; point < 5; point++) {
                    int at = anchor * 10 + point * 2;
                    double x = cx + (double) kps[at] * stride;
                    double y = cy + (double) kps[at + 1] * stride;
                    if (x < 0 || x >= resizedWidth || y < 0 || y >= resizedHeight) {
                        valid = false;
                        break;
                    }
                    landmarks[point][0] = (float) (x * scaleX);
                    landmarks[point][1] = (float) (y * scaleY);
                }
                if (!valid || !hasLandmarkSpread(landmarks)) continue;
                Candidate candidate = new Candidate(new Detection(x1, y1, x2, y2, score, landmarks), order);
                if (best.size() == MAX_NMS_CANDIDATES) best.poll();
                best.add(candidate);
            }
        }
        List<Candidate> sorted = new ArrayList<>(best);
        Collections.sort(sorted, Collections.reverseOrder(WORST_FIRST));
        List<Detection> kept = new ArrayList<>();
        for (Candidate candidate : sorted) {
            boolean suppressed = false;
            for (Detection prior : kept) {
                if (iou(candidate.detection, prior) > NMS_IOU) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) kept.add(candidate.detection);
        }
        return Collections.unmodifiableList(kept);
    }

    private static final class Candidate {
        final Detection detection;
        final int order;

        Candidate(Detection detection, int order) {
            this.detection = detection;
            this.order = order;
        }
    }

    private static final Comparator<Candidate> WORST_FIRST = new Comparator<Candidate>() {
        @Override public int compare(Candidate a, Candidate b) {
            int byScore = Float.compare(a.detection.score, b.detection.score);
            return byScore != 0 ? byScore : Integer.compare(b.order, a.order);
        }
    };

    private static double iou(Detection a, Detection b) {
        double width = Math.max(0, (double) Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
        double height = Math.max(0, (double) Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
        double intersection = width * height;
        double areaA = ((double) a.x2 - a.x1) * ((double) a.y2 - a.y1);
        double areaB = ((double) b.x2 - b.x1) * ((double) b.y2 - b.y1);
        return intersection / (areaA + areaB - intersection);
    }

    private static void validateTensor(float[] tensor, int count, boolean scores) {
        if (tensor == null || tensor.length != count) {
            throw new IllegalArgumentException("SCRFD tensor length mismatch; expected " + count);
        }
        for (float value : tensor) {
            requireFinite(value, "SCRFD tensor");
            if (scores && (value < 0 || value > 1)) {
                throw new IllegalArgumentException("SCRFD scores must be probabilities");
            }
        }
    }

    private static boolean hasLandmarkSpread(float[][] landmarks) {
        double cx = 0, cy = 0;
        for (float[] point : landmarks) {
            cx += point[0];
            cy += point[1];
        }
        cx /= 5;
        cy /= 5;
        double xx = 0, xy = 0, yy = 0;
        for (float[] point : landmarks) {
            double x = point[0] - cx, y = point[1] - cy;
            xx += x * x;
            xy += x * y;
            yy += y * y;
        }
        double energy = xx + yy;
        return energy > 0 && xx * yy - xy * xy > CONDITION_EPSILON * energy * energy;
    }

    /**
     * Fits the least-squares, orientation-preserving similarity original -> aligned.
     * Landmarks must be five finite, noncollinear (x,y) pairs in eye/eye/nose/mouth/mouth
     * order. Supported crops are 112 (ArcFace) and 128 (INSwapper: x + 8, no scaling).
     * Returns [a,-b,tx,b,a,ty]; the fit minimizes total squared landmark error.
     */
    public static double[] fit(float[][] fivePoints, int outputSize) {
        if (outputSize != 112 && outputSize != 128) {
            throw new IllegalArgumentException("aligned size must be 112 or 128");
        }
        validateLandmarks(fivePoints);
        double sx = 0, sy = 0, dx = 0, dy = 0;
        double offset = outputSize == 128 ? 8 : 0;
        for (int i = 0; i < 5; i++) {
            sx += fivePoints[i][0];
            sy += fivePoints[i][1];
            dx += ARCFACE_112[i][0] + offset;
            dy += ARCFACE_112[i][1];
        }
        sx /= 5;
        sy /= 5;
        dx /= 5;
        dy /= 5;
        double xx = 0, xy = 0, yy = 0, dot = 0, cross = 0;
        for (int i = 0; i < 5; i++) {
            double x = fivePoints[i][0] - sx;
            double y = fivePoints[i][1] - sy;
            double u = ARCFACE_112[i][0] + offset - dx;
            double v = ARCFACE_112[i][1] - dy;
            xx += x * x;
            xy += x * y;
            yy += y * y;
            dot += x * u + y * v;
            cross += x * v - y * u;
        }
        double energy = xx + yy;
        if (!(energy > 0) || xx * yy - xy * xy <= CONDITION_EPSILON * energy * energy) {
            throw new IllegalArgumentException("landmarks are coincident or collinear");
        }
        double a = dot / energy;
        double b = cross / energy;
        if (!(Math.hypot(a, b) > 0)) {
            throw new IllegalArgumentException("landmarks do not determine a nonzero similarity");
        }
        double[] affine = {a, -b, dx - a * sx + b * sy, b, a, dy - b * sx - a * sy};
        validateAffine(affine);
        return affine;
    }

    /** Returns a new inverse for any finite, nonsingular row-major 2-by-3 affine. */
    public static double[] inverse(double[] affine) {
        validateAffine(affine);
        // Scale first: even a well-conditioned transform can overflow its determinant.
        double scale = Math.max(Math.max(Math.abs(affine[0]), Math.abs(affine[1])),
                Math.max(Math.abs(affine[3]), Math.abs(affine[4])));
        if (scale == 0) throw new IllegalArgumentException("singular affine");
        double a = affine[0] / scale, b = affine[1] / scale;
        double c = affine[3] / scale, d = affine[4] / scale;
        double determinant = a * d - b * c;
        if (Math.abs(determinant) <= CONDITION_EPSILON) {
            throw new IllegalArgumentException("singular or ill-conditioned affine");
        }
        double factor = (1 / scale) / determinant;
        double ia = d * factor, ib = -b * factor;
        double ic = -c * factor, id = a * factor;
        double[] result = {ia, ib, -ia * affine[2] - ib * affine[5],
                ic, id, -ic * affine[2] - id * affine[5]};
        validateAffine(result);
        return result;
    }

    /** Applies an affine to a finite (x,y), rejecting an unrepresentable result. */
    public static double[] transform(double[] affine, double x, double y) {
        validateAffine(affine);
        requireFinite(x, "x");
        requireFinite(y, "y");
        double u = affine[0] * x + affine[1] * y + affine[2];
        double v = affine[3] * x + affine[4] * y + affine[5];
        requireFinite(u, "transformed x");
        requireFinite(v, "transformed y");
        return new double[] {u, v};
    }

    /**
     * L2-normalizes a raw 512-vector, multiplies it as a ROW vector by the row-major
     * 512-by-512 emap, then L2-normalizes the mapped vector. All entries are checked,
     * including matrix rows multiplied by zero. Double accumulation avoids float
     * underflow/overflow; zero raw or mapped norms are rejected. Uses O(512) scratch.
     */
    public static float[] projectIdentity(float[] embedding, float[] rowMajorEmap) {
        if (embedding == null || embedding.length != IDENTITY_SIZE
                || rowMajorEmap == null || rowMajorEmap.length != IDENTITY_SIZE * IDENTITY_SIZE) {
            throw new IllegalArgumentException("identity requires 512 values and a 512-by-512 emap");
        }
        double normSquared = 0;
        for (float value : embedding) {
            requireFinite(value, "embedding");
            normSquared += (double) value * value;
        }
        if (!(normSquared > 0)) throw new IllegalArgumentException("zero embedding norm");
        double norm = Math.sqrt(normSquared);
        double[] mapped = new double[IDENTITY_SIZE];
        for (int row = 0; row < IDENTITY_SIZE; row++) {
            double value = embedding[row] / norm;
            int base = row * IDENTITY_SIZE;
            for (int col = 0; col < IDENTITY_SIZE; col++) {
                float weight = rowMajorEmap[base + col];
                requireFinite(weight, "emap");
                mapped[col] += value * weight;
            }
        }
        double mappedNormSquared = 0;
        for (double value : mapped) mappedNormSquared += value * value;
        if (!(mappedNormSquared > 0) || !Double.isFinite(mappedNormSquared)) {
            throw new IllegalArgumentException("zero or nonfinite mapped identity norm");
        }
        double mappedNorm = Math.sqrt(mappedNormSquared);
        float[] result = new float[IDENTITY_SIZE];
        for (int i = 0; i < IDENTITY_SIZE; i++) result[i] = (float) (mapped[i] / mappedNorm);
        return result;
    }

    private static void validateLandmarks(float[][] landmarks) {
        if (landmarks == null || landmarks.length != 5) {
            throw new IllegalArgumentException("exactly five landmarks required");
        }
        for (float[] point : landmarks) {
            if (point == null || point.length != 2) {
                throw new IllegalArgumentException("each landmark must contain x and y");
            }
            requireFinite(point[0], "landmark x");
            requireFinite(point[1], "landmark y");
        }
    }

    static void validateAffine(double[] affine) {
        if (affine == null || affine.length != 6) {
            throw new IllegalArgumentException("affine must contain exactly six values");
        }
        for (double value : affine) requireFinite(value, "affine");
    }

    private static void requireFinite(double value, String name) {
        if (!Double.isFinite(value)) throw new IllegalArgumentException(name + " must be finite");
    }
}
