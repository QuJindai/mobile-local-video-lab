package com.qujindai.localvideo;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class CheckpointDownloadPolicyTest {
    @Test
    public void resumesOnlyWhenPartialFileIsSmallerThanExpected() {
        long expected = 1_074_370_038L;
        assertEquals(0L, CheckpointDownloadPolicy.resumeOffset(0L, expected));
        assertEquals(123_456L, CheckpointDownloadPolicy.resumeOffset(123_456L, expected));
        assertEquals(0L, CheckpointDownloadPolicy.resumeOffset(expected, expected));
        assertEquals(0L, CheckpointDownloadPolicy.resumeOffset(expected + 1L, expected));
    }

    @Test
    public void percentIsBoundedAndUsesKnownTotal() {
        assertEquals(0, CheckpointDownloadPolicy.percent(0L, 100L));
        assertEquals(50, CheckpointDownloadPolicy.percent(50L, 100L));
        assertEquals(100, CheckpointDownloadPolicy.percent(100L, 100L));
        assertEquals(100, CheckpointDownloadPolicy.percent(150L, 100L));
        assertEquals(0, CheckpointDownloadPolicy.percent(50L, 0L));
    }
}
