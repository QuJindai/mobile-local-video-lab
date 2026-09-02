package com.qujindai.localvideo;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class BackendRouterTest {
    @Test
    public void rifeIsReadyWhenRuntimeIsReady() {
        BackendRouter.Decision d = BackendRouter.resolve(
                BackendRouter.Backend.RIFE_MOTION, true, false, false, 8192);
        assertTrue(d.ready);
        assertEquals(BackendRouter.Blocker.NONE, d.blocker);
    }

    @Test
    public void depthBackendNeedsPackagedDepthRuntime() {
        BackendRouter.Decision d = BackendRouter.resolve(
                BackendRouter.Backend.DEPTH_RIFE,
                true, false, false, false, 12288);
        assertFalse(d.ready);
        assertEquals(BackendRouter.Blocker.DEPTH_RUNTIME_MISSING, d.blocker);
    }

    @Test
    public void depthBackendNeedsRifeToo() {
        BackendRouter.Decision d = BackendRouter.resolve(
                BackendRouter.Backend.DEPTH_RIFE,
                false, true, false, false, 12288);
        assertFalse(d.ready);
        assertEquals(BackendRouter.Blocker.BUILTIN_RUNTIME_MISSING, d.blocker);
    }

    @Test
    public void depthBackendReadyWhenBothLocalRuntimesExist() {
        BackendRouter.Decision d = BackendRouter.resolve(
                BackendRouter.Backend.DEPTH_RIFE,
                true, true, false, false, 12288);
        assertTrue(d.ready);
        assertEquals(BackendRouter.Blocker.NONE, d.blocker);
    }

    @Test
    public void mobileI2vNeedsModelPackFirst() {
        BackendRouter.Decision d = BackendRouter.resolve(
                BackendRouter.Backend.MOBILE_I2V, true, false, true, 12288);
        assertFalse(d.ready);
        assertEquals(BackendRouter.Blocker.MODEL_PACK_MISSING, d.blocker);
    }

    @Test
    public void mobileI2vDoesNotPretendReadyWithoutRuntime() {
        BackendRouter.Decision d = BackendRouter.resolve(
                BackendRouter.Backend.MOBILE_I2V, true, true, false, 12288);
        assertFalse(d.ready);
        assertEquals(BackendRouter.Blocker.RUNTIME_PENDING, d.blocker);
    }

    @Test
    public void mobileI2vEnforcesMinimumRam() {
        BackendRouter.Decision d = BackendRouter.resolve(
                BackendRouter.Backend.MOBILE_I2V, true, true, true, 6144);
        assertFalse(d.ready);
        assertEquals(BackendRouter.Blocker.INSUFFICIENT_RAM, d.blocker);
    }

    @Test
    public void mobileI2vReadyOnlyWhenAllGatesPass() {
        BackendRouter.Decision d = BackendRouter.resolve(
                BackendRouter.Backend.MOBILE_I2V, true, true, true, 12288);
        assertTrue(d.ready);
        assertEquals(BackendRouter.Blocker.NONE, d.blocker);
    }
}
