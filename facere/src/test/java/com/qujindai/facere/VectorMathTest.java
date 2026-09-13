package com.qujindai.facere;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class VectorMathTest {
    @Test
    public void normalizeProducesUnitVector() {
        assertArrayEquals(new float[]{0.6f, 0.8f},
                VectorMath.normalize(new float[]{3f, 4f}), 1e-6f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void normalizeRejectsZeroVector() {
        VectorMath.normalize(new float[]{0f, 0f, 0f});
    }

    @Test
    public void mapAndNormalizeUsesRowVectorTimesMatrix() {
        float[] mapped = VectorMath.mapAndNormalize(
                new float[]{1f, 0f},
                new float[][]{{0f, 1f}, {1f, 0f}});
        assertArrayEquals(new float[]{0f, 1f}, mapped, 1e-6f);
    }
}
