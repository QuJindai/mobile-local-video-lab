package com.qujindai.localvideo;

import java.util.Arrays;
import java.util.List;

/** Standalone known-answer tests; no Android, model downloads or test framework needed. */
public final class FaceSwapCoreTest {
    private static int passed;
    private static int failed;

    public static void main(String[] args) {
        run("fit ArcFace 112 identity", FaceSwapCoreTest::fitIdentity);
        run("fit known rotation scale translation", FaceSwapCoreTest::fitKnownTransform);
        run("fit ArcFace 128 uses x offset only", FaceSwapCoreTest::fit128Offset);
        run("fit least squares across five landmarks", FaceSwapCoreTest::fitLeastSquares);
        run("fit cannot introduce reflection", FaceSwapCoreTest::fitNoReflection);
        run("fit rejects malformed and degenerate points", FaceSwapCoreTest::fitInvalid);
        run("affine inverse known answer and roundtrip", FaceSwapCoreTest::affineRoundtrip);
        run("affine rejects singular nonfinite and overflow", FaceSwapCoreTest::affineInvalid);
        run("identity projection row vector orientation", FaceSwapCoreTest::projectionOrientation);
        run("identity projection extreme finite values", FaceSwapCoreTest::projectionRange);
        run("identity projection rejects malformed zero nonfinite", FaceSwapCoreTest::projectionInvalid);
        run("SCRFD stride8 second anchor and independent xy scaling", FaceSwapCoreTest::scrfdScale);
        run("SCRFD stride16 stride32 and score order", FaceSwapCoreTest::scrfdStrides);
        run("SCRFD continuous IoU suppression", FaceSwapCoreTest::scrfdNms);
        run("SCRFD padding rejection and border clipping", FaceSwapCoreTest::scrfdPadding);
        run("SCRFD rejects bad boxes and landmarks", FaceSwapCoreTest::scrfdCandidates);
        run("SCRFD rejects tensor and size errors", FaceSwapCoreTest::scrfdInvalid);
        run("detection validates and snapshots landmarks", FaceSwapCoreTest::detectionSnapshot);
        run("SCRFD bounded highest score selection", FaceSwapCoreTest::scrfdBounded);
        run("warp identity retains exact integer samples", FaceSwapCoreTest::warpIdentity);
        run("warp independent fractional bilinear answer", FaceSwapCoreTest::warpBilinear);
        run("warp fractional negative coordinates use black border", FaceSwapCoreTest::warpBorder);
        run("RGB channel order and normalization", FaceSwapCoreTest::rgbOrder);
        run("RGB output clipping and rounding", FaceSwapCoreTest::rgbOutput);
        run("pixel shape finite and allocation guards", FaceSwapCoreTest::pixelsInvalid);
        run("paste back translation retains outside target", FaceSwapCoreTest::compositeTranslation);
        run("paste back rotation and subpixel sampling", FaceSwapCoreTest::compositeRotation);
        run("paste back smooth feather and exact exterior", FaceSwapCoreTest::compositeFeather);
        run("paste back offscreen is unchanged", FaceSwapCoreTest::compositeOffscreen);
        System.out.println("FaceSwapCoreTest: " + passed + " passed, " + failed + " failed");
        if (failed != 0) throw new AssertionError("Core test failures: " + failed);
    }

    private static float[][] template() {
        return new float[][] {{38.2946f, 51.6963f}, {73.5318f, 51.5014f},
                {56.0252f, 71.7366f}, {41.5493f, 92.3655f}, {70.7299f, 92.2041f}};
    }

    private static void fitIdentity() {
        float[][] points = template();
        float[][] before = template();
        near(new double[] {1, 0, 0, 0, 1, 0}, FaceSwapMath.fit(points, 112), 1e-5);
        check(Arrays.deepEquals(points, before), "fit mutated its landmarks");
    }

    private static void fitKnownTransform() {
        float[][] points = template();
        for (float[] p : points) {
            float x = p[0];
            p[0] = 30 - 2 * p[1];
            p[1] = 2 * x - 12;
        }
        // Original = rotate 90 degrees, scale 2, translate (30,-12).
        near(new double[] {0, .5, 6, -.5, 0, 15}, FaceSwapMath.fit(points, 112), 1e-5);
    }

    private static void fit128Offset() {
        near(new double[] {1, 0, 8, 0, 1, 0}, FaceSwapMath.fit(template(), 128), 1e-5);
    }

    private static void fitLeastSquares() {
        // Source centroid (1,1), sum squared distances 8; all five targets contribute.
        float[][] points = {{0, 0}, {2, 0}, {1, 1}, {0, 2}, {2, 2}};
        near(new double[] {18.2237125, .1011375, 37.70131,
                -.1011375, 18.2237125, 53.778205}, FaceSwapMath.fit(points, 112), 1e-5);
    }

    private static void fitNoReflection() {
        float[][] mirrored = template();
        for (float[] p : mirrored) p[0] = -p[0];
        double[] m = FaceSwapMath.fit(mirrored, 112);
        check(m[0] * m[4] - m[1] * m[3] > 0, "fit reflected the face");
        near(m[0], m[4], 0);
        near(-m[1], m[3], 0);
    }

    private static void fitInvalid() {
        rejects(() -> FaceSwapMath.fit(null, 112));
        rejects(() -> FaceSwapMath.fit(new float[4][2], 112));
        rejects(() -> FaceSwapMath.fit(new float[5][2], 112));
        rejects(() -> FaceSwapMath.fit(new float[][] {{0, 0}, {1, 1}, {2, 2},
                {3, 3}, {4, 4}}, 112));
        float[][] nonfinite = template();
        nonfinite[2][0] = Float.NaN;
        rejects(() -> FaceSwapMath.fit(nonfinite, 112));
        nonfinite[2][0] = Float.POSITIVE_INFINITY;
        rejects(() -> FaceSwapMath.fit(nonfinite, 112));
        float[][] ragged = template();
        ragged[3] = new float[3];
        rejects(() -> FaceSwapMath.fit(ragged, 112));
        ragged[3] = null;
        rejects(() -> FaceSwapMath.fit(ragged, 112));
        rejects(() -> FaceSwapMath.fit(template(), 0));
        rejects(() -> FaceSwapMath.fit(template(), Integer.MAX_VALUE));
    }

    private static void affineRoundtrip() {
        double[] m = {2, 1, 10, -1, 3, -7};
        double[] inv = FaceSwapMath.inverse(m);
        near(new double[] {3.0 / 7, -1.0 / 7, -37.0 / 7,
                1.0 / 7, 2.0 / 7, 4.0 / 7}, inv, 1e-12);
        double[] aligned = FaceSwapMath.transform(m, 4, -2);
        near(new double[] {16, -17}, aligned, 1e-12);
        near(new double[] {4, -2}, FaceSwapMath.transform(inv, aligned[0], aligned[1]), 1e-12);
        near(new double[] {2, 1, 10, -1, 3, -7}, m, 0);
        double[] huge = {1e200, 0, 0, 0, 1e200, 0};
        near(1, FaceSwapMath.inverse(huge)[0] * 1e200, 1e-12);
        double[] tiny = {1e-200, 0, 0, 0, 1e-200, 0};
        near(1, FaceSwapMath.inverse(tiny)[0] * 1e-200, 1e-12);
    }

    private static void affineInvalid() {
        rejects(() -> FaceSwapMath.inverse(new double[] {1, 2, 0, 2, 4, 0}));
        rejects(() -> FaceSwapMath.inverse(new double[6]));
        rejects(() -> FaceSwapMath.inverse(new double[7]));
        rejects(() -> FaceSwapMath.inverse(null));
        rejects(() -> FaceSwapMath.inverse(new double[] {1, 0, 0, 0, 1, Double.NaN}));
        rejects(() -> FaceSwapMath.transform(new double[] {1, 0, 0, 0, 1, 0}, Double.NaN, 0));
        rejects(() -> FaceSwapMath.transform(new double[] {1, 0, 0, 0, 1, 0}, 0,
                Double.POSITIVE_INFINITY));
        rejects(() -> FaceSwapMath.transform(new double[] {Double.MAX_VALUE, 0, 0, 0, 1, 0}, 2, 0));
        rejects(() -> FaceSwapMath.transform(new double[5], 0, 0));
    }

    private static void projectionOrientation() {
        float[] embedding = new float[512];
        embedding[0] = 3;
        embedding[1] = 4;
        float[] emap = new float[512 * 512];
        emap[0] = 2;
        emap[1] = 1;
        emap[512 + 1] = 3;
        emap[511] = 5;
        emap[512 + 511] = -3.75f; // Last output cancels exactly.
        float[] result = FaceSwapMath.projectIdentity(embedding, emap);
        check(result.length == 512, "identity output shape");
        // [3,4] * [[2,1],[0,3]] = [6,15]; transpose would give [10,12].
        near(2 / Math.sqrt(29), result[0], 1e-6);
        near(5 / Math.sqrt(29), result[1], 1e-6);
        for (int i = 2; i < 512; i++) near(0, result[i], 1e-6);
        near(3, embedding[0], 0);
        near(-3.75, emap[512 + 511], 0);
    }

    private static void projectionRange() {
        float[] embedding = new float[512];
        float[] emap = new float[512 * 512];
        embedding[511] = Float.MIN_VALUE;
        emap[511 * 512 + 17] = Float.MAX_VALUE;
        near(1, FaceSwapMath.projectIdentity(embedding, emap)[17], 1e-6);
        Arrays.fill(embedding, Float.MAX_VALUE);
        Arrays.fill(emap, Float.MAX_VALUE);
        float[] result = FaceSwapMath.projectIdentity(embedding, emap);
        for (float v : result) near(1 / Math.sqrt(512), v, 1e-6);
    }

    private static void projectionInvalid() {
        float[] e = new float[512];
        float[] m = new float[512 * 512];
        rejects(() -> FaceSwapMath.projectIdentity(e, m));
        e[0] = 1;
        rejects(() -> FaceSwapMath.projectIdentity(e, m));
        m[0] = 1;
        m[m.length - 1] = Float.NaN; // Must check even a zero embedding row.
        rejects(() -> FaceSwapMath.projectIdentity(e, m));
        m[m.length - 1] = Float.NEGATIVE_INFINITY;
        rejects(() -> FaceSwapMath.projectIdentity(e, m));
        m[m.length - 1] = 0;
        e[511] = Float.NaN;
        rejects(() -> FaceSwapMath.projectIdentity(e, m));
        rejects(() -> FaceSwapMath.projectIdentity(new float[511], m));
        rejects(() -> FaceSwapMath.projectIdentity(new float[512], new float[1]));
        rejects(() -> FaceSwapMath.projectIdentity(null, m));
        rejects(() -> FaceSwapMath.projectIdentity(new float[512], null));
    }

    private static float[][] outputs(int size) {
        float[][] outputs = new float[9][];
        for (int level = 0; level < 3; level++) {
            int side = size / (8 << level);
            int anchors = side * side * 2;
            outputs[level] = new float[anchors];
            outputs[level + 3] = new float[anchors * 4];
            outputs[level + 6] = new float[anchors * 10];
        }
        return outputs;
    }

    private static void anchor(float[][] o, int level, int index, float score,
                               float[] distances, float[] offsets) {
        o[level][index] = score;
        System.arraycopy(distances, 0, o[level + 3], index * 4, 4);
        System.arraycopy(offsets, 0, o[level + 6], index * 10, 10);
    }

    private static float[] faceOffsets() {
        return new float[] {-.25f, -.25f, .25f, -.25f, 0, 0, -.2f, .25f, .2f, .25f};
    }

    private static void scrfdScale() {
        float[][] o = outputs(32);
        // Row1,col2,anchor1: index=13, center=(16,8), distances in stride units.
        anchor(o, 0, 13, .9f, new float[] {1, .5f, 1.5f, 1},
                new float[] {-.5f, -.25f, .5f, -.25f, 0, .25f, -.25f, .75f, .25f, .75f});
        List<FaceSwapMath.Detection> detections = FaceSwapMath.decodeScrfd(o, 32, 80, 60, 32, 20, .5f);
        check(detections.size() == 1, "second anchor missing");
        FaceSwapMath.Detection d = detections.get(0);
        near(new double[] {20, 12, 70, 48, .9},
                new double[] {d.x1, d.y1, d.x2, d.y2, d.score}, 1e-5);
        double[][] expected = {{30, 18}, {50, 18}, {40, 30}, {35, 42}, {45, 42}};
        for (int i = 0; i < 5; i++) {
            near(expected[i][0], d.landmarks[i][0], 1e-5);
            near(expected[i][1], d.landmarks[i][1], 1e-5);
        }
        near(.9, o[0][13], 1e-6);
        o[6][13 * 10] = 99;
        near(30, d.landmarks[0][0], 0);
    }

    private static void scrfdStrides() {
        float[][] o = outputs(64);
        anchor(o, 1, 11, .6f, new float[] {.25f, .25f, .25f, .25f}, faceOffsets());
        anchor(o, 2, 6, .95f, new float[] {.125f, .125f, .125f, .125f},
                new float[] {-.1f, -.1f, .1f, -.1f, 0, 0, -.1f, .1f, .1f, .1f});
        List<FaceSwapMath.Detection> ds = FaceSwapMath.decodeScrfd(o, 64, 64, 64, 64, 64, .6f);
        check(ds.size() == 2, "stride16/32 or inclusive threshold lost");
        near(new double[] {28, 28, 36, 36}, box(ds.get(0)), 1e-5);
        near(new double[] {12, 12, 20, 20}, box(ds.get(1)), 1e-5);
        near(.95, ds.get(0).score, 1e-6);
    }

    private static void scrfdNms() {
        float[][] o = outputs(64);
        float[] box = {1.25f, 1.25f, 1.25f, 1.25f};
        anchor(o, 0, 36, .7f, box, faceOffsets()); // [6,6,26,26]
        anchor(o, 0, 37, .95f, box, faceOffsets()); // Duplicate second anchor wins.
        anchor(o, 0, 38, .9f, box, faceOffsets()); // Shift8: IoU=240/560 > .4.
        anchor(o, 0, 39, .8f, new float[] {1.125f, 1.25f, 1.375f, 1.25f}, faceOffsets());
        // Shift9: IoU=220/580 < .4. Pixel-inclusive (+1) IoU would wrongly suppress it.
        List<FaceSwapMath.Detection> ds = FaceSwapMath.decodeScrfd(o, 64, 64, 64, 64, 64, .5f);
        check(ds.size() == 2, "continuous IoU .4 NMS");
        near(.95, ds.get(0).score, 1e-6);
        near(.8, ds.get(1).score, 1e-6);
        near(new double[] {15, 6, 35, 26}, box(ds.get(1)), 1e-5);
    }

    private static void scrfdPadding() {
        float[][] o = outputs(32);
        anchor(o, 0, 13, .8f, new float[] {3, 2, 2, 2}, faceOffsets());
        // Anchor center in bottom padding, despite a box reaching into content.
        anchor(o, 0, 21, .99f, new float[] {1, 2, 1, .5f}, faceOffsets());
        List<FaceSwapMath.Detection> ds = FaceSwapMath.decodeScrfd(o, 32, 64, 32, 32, 16, .5f);
        check(ds.size() == 1, "padding anchor accepted or border face discarded");
        near(new double[] {0, 0, 64, 32}, box(ds.get(0)), 1e-5);
        o = outputs(32);
        anchor(o, 0, 15, .9f, new float[] {1, 1, 1, 1}, faceOffsets());
        check(FaceSwapMath.decodeScrfd(o, 32, 20, 32, 20, 32, .5f).isEmpty(),
                "right padding accepted");
        o = outputs(32);
        anchor(o, 0, 13, .9f, new float[] {1, 1, 1, 1},
                new float[] {-.5f, 0, .5f, 0, 0, .5f, -.25f, 1, .25f, 1});
        check(FaceSwapMath.decodeScrfd(o, 32, 32, 16, 32, 16, .5f).isEmpty(),
                "landmarks in padding accepted");
    }

    private static void scrfdCandidates() {
        float[][] o = outputs(32);
        anchor(o, 0, 13, .9f, new float[] {-1, 1, 1, 1}, faceOffsets());
        check(decode32(o).isEmpty(), "negative box distance accepted");
        anchor(o, 0, 13, .9f, new float[] {0, 1, 0, 1}, faceOffsets());
        check(decode32(o).isEmpty(), "zero width accepted");
        anchor(o, 0, 13, .9f, new float[] {1, 1, 1, 1}, new float[10]);
        check(decode32(o).isEmpty(), "coincident landmarks accepted");
        float[] offsets = faceOffsets();
        offsets[0] = Float.MAX_VALUE;
        anchor(o, 0, 13, .9f, new float[] {1, 1, 1, 1}, offsets);
        check(decode32(o).isEmpty(), "out of image landmark accepted");
    }

    private static List<FaceSwapMath.Detection> decode32(float[][] o) {
        return FaceSwapMath.decodeScrfd(o, 32, 32, 32, 32, 32, .5f);
    }

    private static void scrfdInvalid() {
        check(decode32(outputs(32)).isEmpty(), "empty scores should give empty detections");
        rejects(() -> decode32(null));
        rejects(() -> decode32(new float[8][]));
        float[][] o = outputs(32);
        o[4] = new float[31];
        rejects(() -> decode32(o));
        o[4] = null;
        rejects(() -> decode32(o));
        float[][] nan = outputs(32);
        nan[8][0] = Float.NaN; // Validate tensors even below the score threshold.
        rejects(() -> decode32(nan));
        nan[8][0] = Float.NEGATIVE_INFINITY;
        rejects(() -> decode32(nan));
        nan[8][0] = 0;
        nan[0][0] = 1.1f;
        rejects(() -> decode32(nan));
        rejects(() -> FaceSwapMath.decodeScrfd(outputs(32), 32, 32, 32, 32, 32, Float.NaN));
        rejects(() -> FaceSwapMath.decodeScrfd(outputs(32), 32, 32, 32, 32, 32, -.1f));
        rejects(() -> FaceSwapMath.decodeScrfd(outputs(32), 32, 32, 32, 32, 32, 1.1f));
        rejects(() -> FaceSwapMath.decodeScrfd(outputs(32), 32, 32, 32, 33, 32, .5f));
        rejects(() -> FaceSwapMath.decodeScrfd(outputs(32), 32, 0, 32, 32, 32, .5f));
        rejects(() -> FaceSwapMath.decodeScrfd(outputs(32), 32, 32, 32, 0, 32, .5f));
        rejects(() -> FaceSwapMath.decodeScrfd(outputs(32), 31, 32, 32, 31, 31, .5f));
        rejects(() -> FaceSwapMath.decodeScrfd(outputs(32), Integer.MAX_VALUE, 32, 32, 32, 32, .5f));
    }

    private static void detectionSnapshot() {
        float[][] landmarks = template();
        FaceSwapMath.Detection d = new FaceSwapMath.Detection(0, 0, 100, 110, .8f, landmarks);
        landmarks[0][0] = 0;
        landmarks[1] = new float[2];
        near(38.2946, d.landmarks[0][0], 1e-5);
        near(73.5318, d.landmarks[1][0], 1e-5);
        rejects(() -> new FaceSwapMath.Detection(1, 0, 1, 110, .8f, template()));
        rejects(() -> new FaceSwapMath.Detection(0, 0, 100, Float.NaN, .8f, template()));
        rejects(() -> new FaceSwapMath.Detection(0, 0, 100, 110, Float.NaN, template()));
        rejects(() -> new FaceSwapMath.Detection(0, 0, 100, 110, .8f, new float[5][2]));
    }

    private static void scrfdBounded() {
        float[][] o = outputs(320);
        // 3042 distinct tiny faces on stride8 anchors; NMS itself suppresses none.
        float[] small = {-.006f, -.006f, .006f, -.006f, 0, 0, -.004f, .006f, .004f, .006f};
        for (int row = 1; row < 40; row++) {
            for (int col = 1; col < 40; col++) {
                int index = (row * 40 + col) * 2;
                anchor(o, 0, index, .6f, new float[] {.01f, .01f, .01f, .01f}, small);
                float[] shifted = small.clone();
                for (int i = 0; i < 10; i += 2) shifted[i] += .05f;
                anchor(o, 0, index + 1, .7f, new float[] {0, .01f, .1f, .01f}, shifted);
            }
        }
        o[0][3199] = .99f;
        List<FaceSwapMath.Detection> ds = FaceSwapMath.decodeScrfd(o, 320, 320, 320, 320, 320, .5f);
        check(ds.size() <= 2048 && ds.size() > 1500, "NMS candidate budget or selection broken");
        near(.99, ds.get(0).score, 1e-6);
        for (int i = 1; i < ds.size(); i++) {
            check(ds.get(i - 1).score >= ds.get(i).score, "scores out of order");
        }
    }

    private static double[] box(FaceSwapMath.Detection d) {
        return new double[] {d.x1, d.y1, d.x2, d.y2};
    }

    private static double[] identityAffine() {
        return new double[] {1, 0, 0, 0, 1, 0};
    }

    private static void warpIdentity() {
        int[] pixels = {0xff000000, 0xffabcdef, 0xff123456, 0x80112233};
        int[] before = pixels.clone();
        int[] result = FaceSwapPixels.warp(pixels, 2, 2, identityAffine(), 2);
        check(Arrays.equals(pixels, result), "identity warp changed integer pixels or alpha");
        check(Arrays.equals(pixels, before) && pixels != result, "warp aliases or mutates input");
        int[] moved = FaceSwapPixels.warp(pixels, 2, 2, new double[] {1, 0, 1, 0, 1, 0}, 3);
        equal(0xff000000, moved[0]);
        equal(0xffabcdef, moved[2]);
        equal(0xff123456, moved[4]);
    }

    private static void warpBilinear() {
        int[] pixels = {0xff000000, 0xff641428, 0xff14643c, 0xff787864};
        int[] result = FaceSwapPixels.warp(pixels, 2, 2,
                new double[] {1, 0, -.25, 0, 1, -.75}, 1);
        // At (.25,.75), weights are 3/16,1/16,9/16,3/16: RGB=(40,80,55).
        equal(0xff285037, result[0]);
    }

    private static void warpBorder() {
        int[] white = {0xffffffff};
        equal(0xffbfbfbf, FaceSwapPixels.warp(white, 1, 1,
                new double[] {1, 0, .25, 0, 1, 0}, 1)[0]);
        equal(0xffbfbfbf, FaceSwapPixels.warp(white, 1, 1,
                new double[] {1, 0, -.25, 0, 1, 0}, 1)[0]);
        equal(0xff404040, FaceSwapPixels.warp(white, 1, 1,
                new double[] {1, 0, .5, 0, 1, .5}, 1)[0]);
        equal(0xff000000, FaceSwapPixels.warp(white, 1, 1,
                new double[] {1, 0, 2, 0, 1, 0}, 1)[0]);
        equal(0xff000000, FaceSwapPixels.warp(white, 1, 1,
                new double[] {1, 0, 1e100, 0, 1, -1e100}, 1)[0]);
    }

    private static void rgbOrder() {
        int[] argb = {0x00102030, 0xffabcdef};
        float[] result = FaceSwapPixels.rgbNchw(argb, 10, 2);
        double[] expected = {3, 80.5, 11, 97.5, 19, 114.5};
        check(result.length == expected.length, "NCHW length");
        for (int i = 0; i < expected.length; i++) near(expected[i], result[i], 0);
        float[] modelInput = FaceSwapPixels.rgbNchw(new int[] {0xff007fff}, 127.5f, 128f);
        near(-.99609375, modelInput[0], 0);
        near(-.00390625, modelInput[1], 0);
        near(.99609375, modelInput[2], 0);
        near(1, FaceSwapPixels.rgbNchw(new int[] {0xffffffff}, 0, 255)[0], 0);
        equal(0x00102030, argb[0]);
    }

    private static void rgbOutput() {
        float[] rgb = {-.2f, 0, .5f, 1.2f, 1, .25f, 0, 1, .5f, 1, 1.5f, -1};
        float[] before = rgb.clone();
        int[] result = FaceSwapPixels.fromRgbNchw(rgb, 2);
        check(Arrays.equals(new int[] {0xff00ff80, 0xff0040ff, 0xff8000ff, 0xffffff00}, result),
                "RGB clipping, channel planes or 8-bit rounding");
        check(Arrays.equals(before, rgb), "fromRgbNchw mutated input");
        int[] original = {0xff012345, 0xff6789ab, 0xffcdef01, 0xff223344};
        check(Arrays.equals(original, FaceSwapPixels.fromRgbNchw(
                FaceSwapPixels.rgbNchw(original, 0, 255), 2)), "8-bit RGB roundtrip");
    }

    private static void pixelsInvalid() {
        int[] one = {0xffffffff};
        int[] swapped = new int[128 * 128];
        rejects(() -> FaceSwapPixels.warp(null, 1, 1, identityAffine(), 1));
        rejects(() -> FaceSwapPixels.warp(one, 2, 2, identityAffine(), 1));
        rejects(() -> FaceSwapPixels.warp(one, 0, 1, identityAffine(), 1));
        rejects(() -> FaceSwapPixels.warp(one, 65536, 65536, identityAffine(), 1));
        rejects(() -> FaceSwapPixels.warp(one, Integer.MAX_VALUE, Integer.MAX_VALUE, identityAffine(), 1));
        rejects(() -> FaceSwapPixels.warp(one, 4097, 4096, identityAffine(), 1));
        rejects(() -> FaceSwapPixels.warp(one, 1, 1, identityAffine(), 1025));
        rejects(() -> FaceSwapPixels.warp(one, 1, 1, identityAffine(), Integer.MAX_VALUE));
        rejects(() -> FaceSwapPixels.warp(one, 1, 1, new double[6], 1));
        rejects(() -> FaceSwapPixels.warp(one, 1, 1,
                new double[] {1, 0, Double.NaN, 0, 1, 0}, 1));
        rejects(() -> FaceSwapPixels.rgbNchw(null, 0, 1));
        rejects(() -> FaceSwapPixels.rgbNchw(new int[0], 0, 1));
        rejects(() -> FaceSwapPixels.rgbNchw(new int[1024 * 1024 + 1], 0, 255));
        rejects(() -> FaceSwapPixels.rgbNchw(one, Float.NaN, 1));
        rejects(() -> FaceSwapPixels.rgbNchw(one, 0, Float.POSITIVE_INFINITY));
        rejects(() -> FaceSwapPixels.rgbNchw(one, 0, 0));
        rejects(() -> FaceSwapPixels.rgbNchw(one, 0, -1));
        rejects(() -> FaceSwapPixels.rgbNchw(one, 0, Float.MIN_VALUE));
        rejects(() -> FaceSwapPixels.fromRgbNchw(null, 1));
        rejects(() -> FaceSwapPixels.fromRgbNchw(new float[2], 1));
        rejects(() -> FaceSwapPixels.fromRgbNchw(new float[3], 0));
        rejects(() -> FaceSwapPixels.fromRgbNchw(new float[3], Integer.MAX_VALUE));
        rejects(() -> FaceSwapPixels.fromRgbNchw(new float[] {0, Float.NaN, 0}, 1));
        rejects(() -> FaceSwapPixels.fromRgbNchw(new float[] {0, 0, Float.POSITIVE_INFINITY}, 1));
        rejects(() -> FaceSwapPixels.composite(null, 1, 1, swapped, identityAffine()));
        rejects(() -> FaceSwapPixels.composite(one, 1, 1, null, identityAffine()));
        rejects(() -> FaceSwapPixels.composite(one, 1, 1, one, identityAffine()));
        rejects(() -> FaceSwapPixels.composite(one, 65536, 65536, swapped, identityAffine()));
        rejects(() -> FaceSwapPixels.composite(one, 1, 1, swapped, new double[6]));
        rejects(() -> FaceSwapPixels.composite(one, 1, 1, swapped,
                new double[] {1, 0, 0, 0, 1, Double.POSITIVE_INFINITY}));
    }

    private static int[] cropGradient() {
        int[] pixels = new int[128 * 128];
        for (int y = 0; y < 128; y++) {
            for (int x = 0; x < 128; x++) pixels[y * 128 + x] = 0xff000025 | (2 * x << 16) | (2 * y << 8);
        }
        return pixels;
    }

    private static void compositeTranslation() {
        int width = 192, height = 176;
        int[] target = new int[width * height];
        for (int i = 0; i < target.length; i++) target[i] = 0x71000000 | ((i * 2017) & 0xffffff);
        int[] before = target.clone(), swapped = cropGradient(), swapBefore = swapped.clone();
        double[] m = {1, 0, -32, 0, 1, -16};
        int[] result = FaceSwapPixels.composite(target, width, height, swapped, m);
        equal(0x71808025, result[80 * width + 96]); // Original(96,80) -> aligned(64,64).
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                if (x < 32 || x >= 160 || y < 16 || y >= 144) equal(target[index], result[index]);
                equal(target[index] >>> 24, result[index] >>> 24);
            }
        }
        equal(target[16 * width + 32], result[16 * width + 32]); // Crop corner stays original.
        check(result != target && Arrays.equals(before, target), "paste back mutated/aliased target");
        check(Arrays.equals(swapBefore, swapped), "paste back mutated crop");
        near(new double[] {1, 0, -32, 0, 1, -16}, m, 0);
    }

    private static void compositeRotation() {
        int[] target = new int[192 * 176];
        Arrays.fill(target, 0xff112233);
        int[] result = FaceSwapPixels.composite(target, 192, 176, cropGradient(),
                new double[] {0, 1, -10, -1, 0, 150});
        equal(0xff8c7825, result[80 * 192 + 90]); // (90,80) -> (70,60).
        equal(0xff112233, result[0]);
        result = FaceSwapPixels.composite(target, 192, 176, cropGradient(),
                new double[] {.5, 0, -16.25, 0, .5, -8.75});
        equal(0xff807f25, result[144 * 192 + 160]); // (63.75,63.25) -> RGB(127.5,126.5,37).
    }

    private static void compositeFeather() {
        int[] target = new int[128 * 128], swapped = new int[target.length];
        Arrays.fill(target, 0xff102030);
        Arrays.fill(swapped, 0xfff0b080);
        int[] result = FaceSwapPixels.composite(target, 128, 128, swapped, identityAffine());
        equal(0xfff0b080, result[64 * 128 + 64]);
        for (int i = 0; i < 128; i++) {
            equal(target[i], result[i]);
            equal(target[127 * 128 + i], result[127 * 128 + i]);
            equal(target[i * 128], result[i * 128]);
            equal(target[i * 128 + 127], result[i * 128 + 127]);
        }
        equal(0xff102030, result[12 * 128 + 12]);
        equal(0xff102030, result[64 * 128 + 116]);
        int partial = 0, previous = 240;
        for (int x = 90; x <= 120; x++) {
            int red = (result[64 * 128 + x] >>> 16) & 255;
            check(red <= previous && previous - red <= 30, "hard or nonmonotonic feather edge");
            if (red > 16 && red < 240) partial++;
            previous = red;
        }
        check(partial >= 6, "feather lacks a smooth transition band");
    }

    private static void compositeOffscreen() {
        int[] target = {0xff112233, 0x80778899, 0xff445566, 0xffaabbcc};
        int[] result = FaceSwapPixels.composite(target, 2, 2, cropGradient(),
                new double[] {1, 0, 10000, 0, 1, -10000});
        check(result != target && Arrays.equals(target, result), "offscreen paste changed target");
    }

    private static void equal(int expected, int actual) {
        if (expected != actual) {
            throw new AssertionError("expected 0x" + Integer.toHexString(expected)
                    + ", got 0x" + Integer.toHexString(actual));
        }
    }

    private static void run(String name, Runnable test) {
        try {
            test.run();
            passed++;
            System.out.println("PASS " + name);
        } catch (AssertionError | RuntimeException e) {
            failed++;
            System.err.println("FAIL " + name + ": " + e);
        }
    }

    private static void rejects(Runnable action) {
        try {
            action.run();
        } catch (IllegalArgumentException expected) {
            return;
        }
        throw new AssertionError("expected IllegalArgumentException");
    }

    private static void near(double expected, double actual, double tolerance) {
        if (!Double.isFinite(actual) || Math.abs(expected - actual) > tolerance) {
            throw new AssertionError("expected " + expected + ", got " + actual);
        }
    }

    private static void near(double[] expected, double[] actual, double tolerance) {
        check(actual != null && expected.length == actual.length, "array shape mismatch");
        for (int i = 0; i < expected.length; i++) near(expected[i], actual[i], tolerance);
    }

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
