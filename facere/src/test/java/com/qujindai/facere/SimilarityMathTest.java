package com.qujindai.facere;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class SimilarityMathTest {
    @Test
    public void estimateMapsScaleAndTranslationExactly() {
        float[][] src = {{0f, 0f}, {1f, 0f}, {0f, 1f}};
        float[][] dst = {{2f, 3f}, {4f, 3f}, {2f, 5f}};
        float[] m = SimilarityMath.estimate(src, dst);
        assertArrayEquals(new float[]{2f, 0f, 2f, 0f, 2f, 3f}, m, 1e-5f);
    }

    @Test
    public void estimateMapsQuarterTurn() {
        float[][] src = {{0f, 0f}, {1f, 0f}, {0f, 1f}};
        float[][] dst = {{5f, 7f}, {5f, 9f}, {3f, 7f}};
        float[] m = SimilarityMath.estimate(src, dst);
        assertArrayEquals(new float[]{0f, -2f, 5f, 2f, 0f, 7f}, m, 1e-5f);
    }
}
