package com.qujindai.facere;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

public class FaceModelDownloadCatalogTest {
    @Test
    public void autoUsesChinaMirrorBeforeOfficial() {
        List<FaceModelDownloadCatalog.SourceMode> attempts =
                FaceModelDownloadCatalog.attemptOrder(FaceModelDownloadCatalog.SourceMode.AUTO);
        assertEquals(2, attempts.size());
        assertEquals(FaceModelDownloadCatalog.SourceMode.CHINA_MIRROR, attempts.get(0));
        assertEquals(FaceModelDownloadCatalog.SourceMode.OFFICIAL, attempts.get(1));
    }

    @Test
    public void pinnedHashesMatchReviewedArtifacts() {
        assertEquals("80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
                FaceModelDownloadCatalog.BUFFALO_ZIP_SHA256);
        assertEquals("5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
                FaceModelDownloadCatalog.DETECTOR_SHA256);
        assertEquals("4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
                FaceModelDownloadCatalog.RECOGNIZER_SHA256);
        assertEquals("e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
                FaceModelDownloadCatalog.SWAPPER_SHA256);
    }

    @Test
    public void sourcesAreHttpsAndHostAllowlisted() {
        assertTrue(FaceModelDownloadCatalog.isTrustedUrl(FaceModelDownloadCatalog.OFFICIAL_BUFFALO_URL));
        assertTrue(FaceModelDownloadCatalog.isTrustedUrl(FaceModelDownloadCatalog.OFFICIAL_SWAPPER_URL));
        assertTrue(FaceModelDownloadCatalog.isTrustedUrl(FaceModelDownloadCatalog.MIRROR_DETECTOR_URL));
        assertTrue(FaceModelDownloadCatalog.isTrustedUrl(FaceModelDownloadCatalog.MIRROR_RECOGNIZER_URL));
        assertTrue(FaceModelDownloadCatalog.isTrustedUrl(FaceModelDownloadCatalog.MIRROR_SWAPPER_URL));
        assertFalse(FaceModelDownloadCatalog.isTrustedUrl("http://hf-mirror.com/model.onnx"));
        assertFalse(FaceModelDownloadCatalog.isTrustedUrl("https://example.com/model.onnx"));
    }
}
