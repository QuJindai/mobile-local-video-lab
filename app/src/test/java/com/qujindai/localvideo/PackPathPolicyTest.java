package com.qujindai.localvideo;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class PackPathPolicyTest {
    @Test
    public void acceptsNormalRelativePaths() {
        assertTrue(PackPathPolicy.isSafe("model-pack.properties"));
        assertTrue(PackPathPolicy.isSafe("models/dit.onnx"));
        assertTrue(PackPathPolicy.isSafe("constants/text_embeddings.bin"));
    }

    @Test
    public void rejectsTraversalAndAbsolutePaths() {
        assertFalse(PackPathPolicy.isSafe("../escape"));
        assertFalse(PackPathPolicy.isSafe("models/../../escape"));
        assertFalse(PackPathPolicy.isSafe("/absolute/path"));
        assertFalse(PackPathPolicy.isSafe("\\windows\\absolute"));
        assertFalse(PackPathPolicy.isSafe("C:/escape"));
    }

    @Test
    public void rejectsEmptyAndDirectoryOnlyNames() {
        assertFalse(PackPathPolicy.isSafe(""));
        assertFalse(PackPathPolicy.isSafe("."));
        assertFalse(PackPathPolicy.isSafe("models/../"));
    }
}
