package com.qujindai.facere;

public final class VectorMath {
    private static final double EPSILON = 1e-20;

    private VectorMath() {}

    public static float[] normalize(float[] input) {
        if (input == null || input.length == 0) {
            throw new IllegalArgumentException("vector is empty");
        }
        double sumSquares = 0.0;
        for (float value : input) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("vector contains non-finite value");
            }
            sumSquares += (double) value * value;
        }
        if (sumSquares <= EPSILON) {
            throw new IllegalArgumentException("vector norm is zero");
        }
        double norm = Math.sqrt(sumSquares);
        float[] result = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = (float) (input[i] / norm);
        }
        return result;
    }

    public static float[] mapAndNormalize(float[] vector, float[][] matrix) {
        if (vector == null || vector.length == 0 || matrix == null
                || matrix.length != vector.length) {
            throw new IllegalArgumentException("embedding/map dimensions do not match");
        }
        int n = vector.length;
        for (float[] row : matrix) {
            if (row == null || row.length != n) {
                throw new IllegalArgumentException("embedding map must be square");
            }
        }
        float[] mapped = new float[n];
        for (int column = 0; column < n; column++) {
            double sum = 0.0;
            for (int row = 0; row < n; row++) {
                float weight = matrix[row][column];
                if (!Float.isFinite(weight)) {
                    throw new IllegalArgumentException("embedding map contains non-finite value");
                }
                sum += (double) vector[row] * weight;
            }
            mapped[column] = (float) sum;
        }
        return normalize(mapped);
    }
}
