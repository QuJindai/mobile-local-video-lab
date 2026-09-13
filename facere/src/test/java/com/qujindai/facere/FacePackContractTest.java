package com.qujindai.facere;

import org.junit.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class FacePackContractTest {
    @Test
    public void requiredSetMustContainAllFourRootFiles() {
        Set<String> files = new HashSet<>(Arrays.asList(
                "det_10g.onnx", "w600k_r50.onnx", "inswapper_128.onnx", "emap.bin"));
        assertTrue(FacePackContract.hasRequired(files));
        files.remove("emap.bin");
        assertFalse(FacePackContract.hasRequired(files));
    }

    @Test
    public void archiveEntryMustBeRootLevelAndTraversalFree() {
        assertTrue(FacePackContract.isSafeEntry("det_10g.onnx"));
        assertTrue(FacePackContract.isSafeEntry("emap.bin"));
        assertFalse(FacePackContract.isSafeEntry("../emap.bin"));
        assertFalse(FacePackContract.isSafeEntry("models/emap.bin"));
        assertFalse(FacePackContract.isSafeEntry("/emap.bin"));
        assertFalse(FacePackContract.isSafeEntry("models\\emap.bin"));
    }
}
