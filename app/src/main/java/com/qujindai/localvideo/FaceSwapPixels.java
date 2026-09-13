package com.qujindai.localvideo;

/**
 * Android-independent, nonmutating ARGB pixel operations for the photo face-swap core.
 * Coordinates refer to integer pixel centers. Invalid dimensions, shapes, transforms
 * or nonfinite values throw IllegalArgumentException before unsafe indexing/allocation.
 * Source/target images are limited to 16,777,216 pixels (64 MiB per ARGB array); model
 * crops/tensors to 1,048,576 pixels (4 MiB ARGB / 12 MiB RGB float output).
 */
public final class FaceSwapPixels {
    private static final int MAX_IMAGE_PIXELS = 16 * 1024 * 1024;
    private static final int MAX_MODEL_SIZE = 1024;
    private static final int MAX_TENSOR_PIXELS = MAX_MODEL_SIZE * MAX_MODEL_SIZE;
    private static final int SWAP_SIZE = 128;
    private static final int BLACK = 0xff000000;
    private static final double MASK_CENTER = 63.5;
    private static final double MASK_RADIUS_X = 52;
    private static final double MASK_RADIUS_Y = 60;
    private static final double MASK_INNER_RADIUS = .75;

    private FaceSwapPixels() {}

    /**
     * Inverse-samples an outSize-square crop using the original -> aligned affine.
     * Uses bilinear interpolation of straight ARGB channels, rounded to nearest byte;
     * EACH out-of-image neighbor contributes opaque black (including fractional borders).
     * outSize must be in [1,1024]. Returns a fresh array, retaining exact integer samples.
     */
    public static int[] warp(int[] pixels, int width, int height, double[] originalToAligned, int outSize) {
        validateImage(pixels, width, height);
        int count = modelPixelCount(outSize);
        double[] inverse = FaceSwapMath.inverse(originalToAligned);
        mappedBounds(inverse, 0, 0, outSize - 1, outSize - 1); // Reject coordinate overflow first.
        int[] result = new int[count];
        for (int y = 0; y < outSize; y++) {
            for (int x = 0; x < outSize; x++) {
                double sx = inverse[0] * x + inverse[1] * y + inverse[2];
                double sy = inverse[3] * x + inverse[4] * y + inverse[5];
                result[y * outSize + x] = bilinear(pixels, width, height, sx, sy);
            }
        }
        return result;
    }

    /**
     * Converts up to 1,048,576 ARGB pixels to flattened [1,3,H,W] RGB (R plane, G, B).
     * Ignores alpha. Each value is (unsignedByte - mean) / std, with mean/std in BYTE
     * units: (127.5,128) for SCRFD, (127.5,127.5) for ArcFace, (0,255) for INSwapper.
     * std must be positive and finite; values unrepresentable as floats are rejected.
     */
    public static float[] rgbNchw(int[] argb, float mean, float std) {
        if (argb == null || argb.length == 0 || argb.length > MAX_TENSOR_PIXELS) {
            throw new IllegalArgumentException("RGB input must have 1..1048576 pixels");
        }
        if (!Float.isFinite(mean) || !Float.isFinite(std) || std <= 0) {
            throw new IllegalArgumentException("normalization requires finite mean and positive std");
        }
        int count = argb.length;
        // count is bounded before multiplying by 3.
        float[] result = new float[count * 3];
        for (int i = 0; i < count; i++) {
            int color = argb[i];
            result[i] = normalize((color >>> 16) & 255, mean, std);
            result[count + i] = normalize((color >>> 8) & 255, mean, std);
            result[2 * count + i] = normalize(color & 255, mean, std);
        }
        return result;
    }

    /**
     * Converts exactly 3*size*size RGB NCHW floats to opaque ARGB. Finite samples are
     * clipped to [0,1], multiplied by 255 and rounded to the nearest byte. Nonfinite
     * samples are rejected even if clipping would hide them. size must be in [1,1024].
     */
    public static int[] fromRgbNchw(float[] rgb, int size) {
        int count = modelPixelCount(size);
        if (rgb == null || rgb.length != count * 3) {
            throw new IllegalArgumentException("RGB tensor must contain exactly 3*size*size values");
        }
        for (float value : rgb) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException("RGB tensor must be finite");
        }
        int[] result = new int[count];
        for (int i = 0; i < count; i++) {
            result[i] = BLACK | (toByte(rgb[i]) << 16)
                    | (toByte(rgb[count + i]) << 8) | toByte(rgb[2 * count + i]);
        }
        return result;
    }

    /**
     * Pastes an actual 128-by-128 swapped crop back into a COPY of the original target.
     * The inverse affine locates the crop's image footprint; each original pixel in
     * that bounded rectangle samples the crop with originalToAligned, avoiding holes
     * under rotation or scaling. Sampling is bilinear. Target alpha is retained.
     *
     * The interior-face ellipse is centered at (63.5,63.5), radii (52,60). Opacity is
     * one inside normalized radius .75 and smoothly falls to zero at radius 1 using
     * cubic smoothstep. It never reaches crop edges. All pixels outside the ellipse
     * remain bit-for-bit identical, including background, crop corners and alpha.
     * No color correction, full-frame filtering or input mutation is performed.
     */
    public static int[] composite(int[] target, int width, int height, int[] swapped128,
                                  double[] originalToAligned) {
        validateImage(target, width, height);
        if (swapped128 == null || swapped128.length != SWAP_SIZE * SWAP_SIZE) {
            throw new IllegalArgumentException("swapped crop must contain 128*128 pixels");
        }
        double[] inverse = FaceSwapMath.inverse(originalToAligned);
        double[] bounds = mappedBounds(inverse,
                MASK_CENTER - MASK_RADIUS_X, MASK_CENTER - MASK_RADIUS_Y,
                MASK_CENTER + MASK_RADIUS_X, MASK_CENTER + MASK_RADIUS_Y);
        mappedBounds(originalToAligned, 0, 0, width - 1, height - 1);
        int[] result = target.clone();
        if (bounds[2] < 0 || bounds[3] < 0 || bounds[0] > width - 1 || bounds[1] > height - 1) {
            return result;
        }
        // Clamp in double before narrowing, so even huge offscreen transforms are safe.
        int xStart = (int) Math.max(0, Math.ceil(bounds[0]));
        int yStart = (int) Math.max(0, Math.ceil(bounds[1]));
        int xEnd = (int) Math.min(width - 1, Math.floor(bounds[2]));
        int yEnd = (int) Math.min(height - 1, Math.floor(bounds[3]));
        for (int y = yStart; y <= yEnd; y++) {
            for (int x = xStart; x <= xEnd; x++) {
                double u = originalToAligned[0] * x + originalToAligned[1] * y + originalToAligned[2];
                double v = originalToAligned[3] * x + originalToAligned[4] * y + originalToAligned[5];
                double radius = Math.hypot((u - MASK_CENTER) / MASK_RADIUS_X,
                        (v - MASK_CENTER) / MASK_RADIUS_Y);
                if (radius >= 1) continue;
                double opacity = 1;
                if (radius > MASK_INNER_RADIUS) {
                    double t = (1 - radius) / (1 - MASK_INNER_RADIUS);
                    opacity = t * t * (3 - 2 * t);
                }
                int at = y * width + x;
                int sample = bilinear(swapped128, SWAP_SIZE, SWAP_SIZE, u, v);
                result[at] = blend(target[at], sample, opacity);
            }
        }
        return result;
    }

    private static int validateImage(int[] pixels, int width, int height) {
        long count = (long) width * height;
        if (width <= 0 || height <= 0 || count > MAX_IMAGE_PIXELS
                || pixels == null || pixels.length != count) {
            throw new IllegalArgumentException("image must match dimensions and contain at most 16777216 pixels");
        }
        return (int) count;
    }

    private static int modelPixelCount(int size) {
        if (size <= 0 || size > MAX_MODEL_SIZE) {
            throw new IllegalArgumentException("model crop size must be in [1,1024]");
        }
        return size * size;
    }

    private static float normalize(int channel, float mean, float std) {
        float value = (float) ((channel - (double) mean) / std);
        if (!Float.isFinite(value)) throw new IllegalArgumentException("normalized RGB overflows float");
        return value;
    }

    private static int toByte(float value) {
        return (int) Math.round(Math.max(0, Math.min(1, (double) value)) * 255);
    }

    private static double[] mappedBounds(double[] affine, double left, double top,
                                          double right, double bottom) {
        double minX = Double.POSITIVE_INFINITY, minY = Double.POSITIVE_INFINITY;
        double maxX = Double.NEGATIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
        for (int corner = 0; corner < 4; corner++) {
            double[] point = FaceSwapMath.transform(affine,
                    (corner & 1) == 0 ? left : right, (corner & 2) == 0 ? top : bottom);
            minX = Math.min(minX, point[0]);
            minY = Math.min(minY, point[1]);
            maxX = Math.max(maxX, point[0]);
            maxY = Math.max(maxY, point[1]);
        }
        return new double[] {minX, minY, maxX, maxY};
    }

    private static int bilinear(int[] pixels, int width, int height, double x, double y) {
        if (!Double.isFinite(x) || !Double.isFinite(y)) {
            throw new IllegalArgumentException("sample coordinates must be finite");
        }
        if (x <= -1 || y <= -1 || x >= width || y >= height) return BLACK;
        int x0 = (int) Math.floor(x), y0 = (int) Math.floor(y);
        double dx = x - x0, dy = y - y0;
        int p00 = pixelAt(pixels, width, height, x0, y0);
        int p10 = pixelAt(pixels, width, height, x0 + 1, y0);
        int p01 = pixelAt(pixels, width, height, x0, y0 + 1);
        int p11 = pixelAt(pixels, width, height, x0 + 1, y0 + 1);
        double w00 = (1 - dx) * (1 - dy), w10 = dx * (1 - dy);
        double w01 = (1 - dx) * dy, w11 = dx * dy;
        int color = 0;
        for (int shift = 0; shift <= 24; shift += 8) {
            int channel = (int) Math.round(((p00 >>> shift) & 255) * w00
                    + ((p10 >>> shift) & 255) * w10 + ((p01 >>> shift) & 255) * w01
                    + ((p11 >>> shift) & 255) * w11);
            color |= channel << shift;
        }
        return color;
    }

    private static int pixelAt(int[] pixels, int width, int height, int x, int y) {
        return x < 0 || x >= width || y < 0 || y >= height ? BLACK : pixels[y * width + x];
    }

    private static int blend(int target, int sample, double opacity) {
        int color = target & 0xff000000;
        for (int shift = 0; shift <= 16; shift += 8) {
            int oldChannel = (target >>> shift) & 255;
            int newChannel = (sample >>> shift) & 255;
            int channel = (int) Math.round(oldChannel + opacity * (newChannel - oldChannel));
            color |= channel << shift;
        }
        return color;
    }
}
