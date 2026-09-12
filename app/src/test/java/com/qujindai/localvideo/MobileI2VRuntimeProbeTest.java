package com.qujindai.localvideo;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class MobileI2VRuntimeProbeTest {
    @Test
    public void rawCheckpointNeverMakesMobileReady() {
        MobileI2VRuntimeProbe.Decision d = MobileI2VRuntimeProbe.decide(
                false, true, true, false, 11000);
        assertFalse(d.ready);
        assertEquals(MobileI2VRuntimeProbe.Backend.NONE, d.backend);
        assertEquals(MobileI2VRuntimeProbe.Blocker.ACCELERATED_PACK_MISSING, d.blocker);
    }

    @Test
    public void openClPackMakesGpuReady() {
        MobileI2VRuntimeProbe.Decision d = MobileI2VRuntimeProbe.decide(
                true, true, true, false, 11000);
        assertTrue(d.ready);
        assertEquals(MobileI2VRuntimeProbe.Backend.MNN_OPENCL, d.backend);
        assertEquals(MobileI2VRuntimeProbe.Blocker.NONE, d.blocker);
    }

    @Test
    public void openClIsPreferredWhenBothAcceleratorsExist() {
        MobileI2VRuntimeProbe.Decision d = MobileI2VRuntimeProbe.decide(
                true, true, true, true, 11000);
        assertTrue(d.ready);
        assertEquals(MobileI2VRuntimeProbe.Backend.MNN_OPENCL, d.backend);
    }

    @Test
    public void qnnHtpIsAcceptedAsASeparateAcceleratedBackend() {
        MobileI2VRuntimeProbe.Decision d = MobileI2VRuntimeProbe.decide(
                true, true, false, true, 11000);
        assertTrue(d.ready);
        assertEquals(MobileI2VRuntimeProbe.Backend.QNN_HTP, d.backend);
    }

    @Test
    public void noSilentCpuFallback() {
        MobileI2VRuntimeProbe.Decision d = MobileI2VRuntimeProbe.decide(
                true, true, false, false, 11000);
        assertFalse(d.ready);
        assertEquals(MobileI2VRuntimeProbe.Backend.NONE, d.backend);
        assertEquals(MobileI2VRuntimeProbe.Blocker.ACCELERATOR_UNAVAILABLE, d.blocker);
    }

    @Test
    public void nativeCoreMustActuallyLoad() {
        MobileI2VRuntimeProbe.Decision d = MobileI2VRuntimeProbe.decide(
                true, false, true, true, 11000);
        assertFalse(d.ready);
        assertEquals(MobileI2VRuntimeProbe.Blocker.NATIVE_RUNTIME_MISSING, d.blocker);
    }

    @Test
    public void memoryGateRemainsMandatory() {
        MobileI2VRuntimeProbe.Decision d = MobileI2VRuntimeProbe.decide(
                true, true, true, false, 6144);
        assertFalse(d.ready);
        assertEquals(MobileI2VRuntimeProbe.Blocker.INSUFFICIENT_RAM, d.blocker);
    }

    @Test
    public void productionReadyNeverReportsCpuDiagnostic() {
        MobileI2VRuntimeProbe.Decision d = MobileI2VRuntimeProbe.decide(
                true, true, true, false, 12288);
        assertTrue(d.ready);
        assertFalse(d.backend == MobileI2VRuntimeProbe.Backend.CPU_DIAGNOSTIC);
    }
}
