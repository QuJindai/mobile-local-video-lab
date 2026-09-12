package com.qujindai.localvideo;

/** Maps encoded frames to generated PNGs without allocating duplicate images. */
public final class FrameSequence {
    private FrameSequence() {}

    public static int outputCount(int sourceFrames, boolean pingPong) {
        if (sourceFrames < 3 || sourceFrames > 10000) {
            throw new IllegalArgumentException("source frames must be within 3..10000");
        }
        return pingPong ? 2 * (sourceFrames - 1) : sourceFrames;
    }

    public static int[] indices(int sourceFrames, boolean pingPong) {
        int[] order = new int[outputCount(sourceFrames, pingPong)];
        for (int i = 0; i < order.length; i++) {
            order[i] = i < sourceFrames ? i : 2 * (sourceFrames - 1) - i;
        }
        return order;
    }
}
