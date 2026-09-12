package com.qujindai.localvideo;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class MobileI2VDownloadSourceTest {
    @Test
    public void officialAndChinaMirrorResolveSamePinnedCheckpoint() {
        MobileI2VDownloadSource official = MobileI2VDownloadSource.official();
        MobileI2VDownloadSource mirror = MobileI2VDownloadSource.chinaMirror();

        assertTrue(official.downloadUrl().startsWith("https://huggingface.co/"));
        assertTrue(mirror.downloadUrl().startsWith("https://hf-mirror.com/"));
        assertNotEquals(official.downloadUrl(), mirror.downloadUrl());
        assertEquals(official.revision, mirror.revision);
        assertEquals(official.fileName, mirror.fileName);
        assertEquals(official.expectedBytes, mirror.expectedBytes);
        assertEquals(official.expectedSha256, mirror.expectedSha256);
        assertEquals("290b2d0dfa93c65388b93e3f7591d7328b335e65", official.revision);
        assertEquals("hybrid_371.pth", official.fileName);
        assertEquals(1074370038L, official.expectedBytes);
        assertEquals("bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d", official.expectedSha256);
    }

    @Test
    public void downloadUrlsUsePinnedRevisionNotMovingMain() {
        assertTrue(MobileI2VDownloadSource.official().downloadUrl().contains(
                "/resolve/290b2d0dfa93c65388b93e3f7591d7328b335e65/hybrid_371.pth"));
        assertTrue(MobileI2VDownloadSource.chinaMirror().downloadUrl().contains(
                "/resolve/290b2d0dfa93c65388b93e3f7591d7328b335e65/hybrid_371.pth"));
    }
}
