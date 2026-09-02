package com.qujindai.localvideo;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class GenerationPlanTest {
    @Test
    public void validPlanKeepsRequestedValues() {
        GenerationPlan plan = new GenerationPlan(512, 720, 17, 8);
        assertEquals(512, plan.getWidth());
        assertEquals(720, plan.getHeight());
        assertEquals(17, plan.getFrames());
        assertEquals(8, plan.getFps());
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsEvenFrameCount() {
        new GenerationPlan(512, 512, 16, 8);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsInvalidFps() {
        new GenerationPlan(512, 512, 17, 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsResolutionNotAlignedTo16() {
        new GenerationPlan(510, 512, 17, 8);
    }
}
