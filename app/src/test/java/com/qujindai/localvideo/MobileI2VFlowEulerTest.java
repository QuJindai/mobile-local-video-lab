package com.qujindai.localvideo;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class MobileI2VFlowEulerTest {
    @Test
    public void scheduleMatchesDiffusers0352Shift3Formula() {
        float[] sigmas = MobileI2VFlowEuler.sigmas(4);
        assertEquals(5, sigmas.length);
        assertEquals(1.0f, sigmas[0], 1e-6f);
        // diffusers 0.35.2: constructor computes sigma_min with shift=3,
        // set_timesteps linspaces to that value, then applies shift=3 again.
        float ctorMin = shift3(0.001f);
        float expectedLast = shift3(ctorMin);
        assertEquals(expectedLast, sigmas[3], 1e-6f);
        assertEquals(0.0f, sigmas[4], 0.0f);
        for (int i = 0; i < sigmas.length - 1; i++) {
            assertTrue(sigmas[i] > sigmas[i + 1]);
        }
    }

    @Test
    public void cfgMathMatchesUpstream() {
        float[] uncond = {1f, -2f, 4f};
        float[] text = {3f, 2f, -2f};
        float[] guided = new float[3];
        MobileI2VFlowEuler.applyCfg(uncond, text, 4.5f, guided);
        assertArrayEquals(new float[] {10f, 16f, -23f}, guided, 1e-6f);
    }

    @Test
    public void eulerStepUsesSigmaNextMinusSigma() {
        float[] sample = {2f, -3f};
        float[] model = {4f, 10f};
        MobileI2VFlowEuler.eulerStepInPlace(sample, model, 0.8f, 0.5f);
        assertArrayEquals(new float[] {0.8f, -6f}, sample, 1e-6f);
    }

    @Test
    public void guideFirstTemporalSliceIsLockedAfterEveryStep() {
        // C=2, T=3, H=1, W=2 => flat C,T,H,W length 12.
        float[] latent = {
                9, 9,  2, 3,  4, 5,
                8, 8,  6, 7,  10, 11
        };
        float[] guide = {100, 101, 200, 201};
        MobileI2VFlowEuler.lockGuideFirstSlice(latent, guide, 2, 3, 1, 2);
        assertArrayEquals(new float[] {
                100, 101,  2, 3,  4, 5,
                200, 201,  6, 7,  10, 11
        }, latent, 0f);
    }

    @Test
    public void sampleLocksGuideAndUsesCfgOnEachDenoiseStep() {
        final int channels = 1, frames = 3, height = 1, width = 1;
        float[] initial = {10f, 20f, 30f};
        float[] guide = {7f};
        final int[] calls = {0};
        MobileI2VFlowEuler.Denoiser denoiser = (cfg2, timestep, outputCfg2) -> {
            calls[0]++;
            assertEquals(6, cfg2.length);
            // First half uncond prediction=1; second half text prediction=3.
            for (int i = 0; i < 3; i++) {
                outputCfg2[i] = 1f;
                outputCfg2[3 + i] = 3f;
            }
        };
        float[] out = MobileI2VFlowEuler.sample(
                initial, guide, channels, frames, height, width,
                3, 2f, denoiser, null);
        assertEquals(3, calls[0]);
        assertEquals(7f, out[0], 0f);
        assertTrue(Float.isFinite(out[1]));
        assertTrue(Float.isFinite(out[2]));
    }

    @Test(expected = IllegalArgumentException.class)
    public void productionSamplerRejectsTooFewSteps() {
        MobileI2VFlowEuler.sigmas(1);
    }

    private static float shift3(float sigma) {
        return 3f * sigma / (1f + 2f * sigma);
    }
}
