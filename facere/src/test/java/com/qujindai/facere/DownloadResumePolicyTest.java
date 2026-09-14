package com.qujindai.facere;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class DownloadResumePolicyTest {
    @Test
    public void rangeResponseAppendsToExistingPartial() {
        assertEquals(DownloadResumePolicy.Action.APPEND,
                DownloadResumePolicy.action(1024L, 206));
    }

    @Test
    public void fullResponseRestartsWhenServerIgnoresRange() {
        assertEquals(DownloadResumePolicy.Action.RESTART,
                DownloadResumePolicy.action(1024L, 200));
    }

    @Test
    public void fullResponseWritesNewFileWhenNoPartialExists() {
        assertEquals(DownloadResumePolicy.Action.RESTART,
                DownloadResumePolicy.action(0L, 200));
    }

    @Test
    public void rangeNotSatisfiableRequiresLocalVerification() {
        assertEquals(DownloadResumePolicy.Action.VERIFY_EXISTING,
                DownloadResumePolicy.action(1024L, 416));
    }
}
