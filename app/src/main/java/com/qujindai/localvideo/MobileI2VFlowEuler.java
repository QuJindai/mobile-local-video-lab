package com.qujindai.localvideo;

import java.util.Arrays;

/**
 * Minimal dependency-free port of the exact scheduler shape used by
 * hustvl/MobileI2V with diffusers 0.35.2 FlowMatchEulerDiscreteScheduler(shift=3).
 *
 * The expensive model call is delegated to an accelerated denoiser. This class
 * only performs scheduler/CFG arithmetic on the ~1.4 MB latent, which is not a
 * production model fallback and does not replace GPU denoising/VAE execution.
 */
public final class MobileI2VFlowEuler {
    public static final int TRAIN_TIMESTEPS = 1000;
    public static final float SHIFT = 3.0f;
    public static final int DEFAULT_STEPS = 28;

    public interface Denoiser {
        /** cfg2 contains [unconditioned latent, conditioned latent]. */
        void run(float[] cfg2, float timestep, float[] outputCfg2);
    }

    public interface Progress {
        void onStep(int completed, int total, float timestep);
    }

    private MobileI2VFlowEuler() {}

    /**
     * Matches diffusers==0.35.2 scheduler construction + set_timesteps:
     * constructor first shifts the training sigma grid (which defines sigmaMin),
     * then set_timesteps linspaces sigmaMax..sigmaMin and shifts that grid again.
     */
    public static float[] sigmas(int inferenceSteps) {
        if (inferenceSteps < 2 || inferenceSteps > 1000) {
            throw new IllegalArgumentException("inferenceSteps must be in [2, 1000]");
        }
        float sigmaMax = shift(1.0f); // 1.0
        float sigmaMin = shift(1.0f / TRAIN_TIMESTEPS);
        float[] result = new float[inferenceSteps + 1];
        for (int i = 0; i < inferenceSteps; i++) {
            float ratio = inferenceSteps == 1 ? 0f : (float) i / (float) (inferenceSteps - 1);
            float initial = sigmaMax + ratio * (sigmaMin - sigmaMax);
            result[i] = shift(initial);
        }
        result[inferenceSteps] = 0.0f;
        return result;
    }

    public static float[] timesteps(int inferenceSteps) {
        float[] sigma = sigmas(inferenceSteps);
        float[] timesteps = new float[inferenceSteps];
        for (int i = 0; i < inferenceSteps; i++) {
            timesteps[i] = sigma[i] * TRAIN_TIMESTEPS;
        }
        return timesteps;
    }

    public static void applyCfg(float[] unconditioned, float[] conditioned, float scale, float[] output) {
        if (unconditioned == null || conditioned == null || output == null
                || unconditioned.length != conditioned.length
                || output.length != unconditioned.length) {
            throw new IllegalArgumentException("CFG buffers must have identical lengths");
        }
        for (int i = 0; i < output.length; i++) {
            float u = unconditioned[i];
            output[i] = u + scale * (conditioned[i] - u);
        }
    }

    public static void eulerStepInPlace(float[] sample, float[] modelOutput, float sigma, float sigmaNext) {
        if (sample == null || modelOutput == null || sample.length != modelOutput.length) {
            throw new IllegalArgumentException("Euler sample/model buffers must have identical lengths");
        }
        float dt = sigmaNext - sigma;
        for (int i = 0; i < sample.length; i++) {
            sample[i] += dt * modelOutput[i];
        }
    }

    /**
     * Upstream does `latents[:, :, :1, :, :] = guide_image` after every step.
     * The flattened layout here is contiguous C,T,H,W.
     */
    public static void lockGuideFirstSlice(
            float[] latent,
            float[] guideFirstSlice,
            int channels,
            int frames,
            int height,
            int width) {
        int plane = checkedProduct(height, width);
        int expectedLatent = checkedProduct(channels, frames, plane);
        int expectedGuide = checkedProduct(channels, plane);
        if (latent == null || latent.length != expectedLatent) {
            throw new IllegalArgumentException("latent length does not match C,T,H,W");
        }
        if (guideFirstSlice == null || guideFirstSlice.length != expectedGuide) {
            throw new IllegalArgumentException("guide length does not match C,H,W");
        }
        int channelStride = frames * plane;
        for (int c = 0; c < channels; c++) {
            System.arraycopy(guideFirstSlice, c * plane, latent, c * channelStride, plane);
        }
    }

    public static float[] sample(
            float[] initial,
            float[] guideFirstSlice,
            int channels,
            int frames,
            int height,
            int width,
            int steps,
            float cfgScale,
            Denoiser denoiser,
            Progress progress) {
        if (denoiser == null) throw new IllegalArgumentException("accelerated denoiser is required");
        if (!Float.isFinite(cfgScale) || cfgScale < 0f || cfgScale > 32f) {
            throw new IllegalArgumentException("invalid CFG scale: " + cfgScale);
        }
        int single = checkedProduct(channels, frames, height, width);
        if (initial == null || initial.length != single) {
            throw new IllegalArgumentException("initial latent length does not match shape");
        }
        int guide = checkedProduct(channels, height, width);
        if (guideFirstSlice == null || guideFirstSlice.length != guide) {
            throw new IllegalArgumentException("guide latent length does not match first temporal slice");
        }

        float[] sigma = sigmas(steps);
        float[] latent = Arrays.copyOf(initial, initial.length);
        float[] cfgInput = new float[single * 2];
        float[] cfgOutput = new float[single * 2];
        float[] guided = new float[single];

        for (int step = 0; step < steps; step++) {
            System.arraycopy(latent, 0, cfgInput, 0, single);
            System.arraycopy(latent, 0, cfgInput, single, single);
            Arrays.fill(cfgOutput, 0f);
            float timestep = sigma[step] * TRAIN_TIMESTEPS;
            denoiser.run(cfgInput, timestep, cfgOutput);
            for (int i = 0; i < single; i++) {
                float u = cfgOutput[i];
                float c = cfgOutput[single + i];
                guided[i] = u + cfgScale * (c - u);
            }
            eulerStepInPlace(latent, guided, sigma[step], sigma[step + 1]);
            lockGuideFirstSlice(latent, guideFirstSlice, channels, frames, height, width);
            if (progress != null) progress.onStep(step + 1, steps, timestep);
        }
        return latent;
    }

    private static float shift(float sigma) {
        return SHIFT * sigma / (1.0f + (SHIFT - 1.0f) * sigma);
    }

    private static int checkedProduct(int... values) {
        long product = 1;
        for (int value : values) {
            if (value <= 0) throw new IllegalArgumentException("shape dimensions must be positive");
            product *= value;
            if (product > Integer.MAX_VALUE) throw new IllegalArgumentException("shape is too large");
        }
        return (int) product;
    }
}
