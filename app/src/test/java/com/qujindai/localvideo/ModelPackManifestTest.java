package com.qujindai.localvideo;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;

public class ModelPackManifestTest {
    @Test
    public void parsesMobileI2vManifest() throws Exception {
        String text = "format=local-video-model-pack-v1\n"
                + "id=mobilei2v-300m\n"
                + "backend=mobilei2v\n"
                + "version=2026.07\n"
                + "source.repo=hustvl/MobileI2V\n"
                + "source.commit=8d0a253c766b05a43ba408baf5e8f800a36be8b4\n"
                + "license.code=Apache-2.0\n"
                + "license.weights=MIT\n"
                + "min.ram.mb=8192\n"
                + "recommended.ram.mb=12288\n"
                + "files=dit.onnx,vae_encoder.onnx,vae_decoder.onnx,constants.bin\n"
                + "sha256.dit.onnx=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
                + "sha256.vae_encoder.onnx=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
                + "sha256.vae_decoder.onnx=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n"
                + "sha256.constants.bin=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\n";
        ModelPackManifest manifest = ModelPackManifest.parse(new ByteArrayInputStream(
                text.getBytes(StandardCharsets.UTF_8)));
        assertEquals("mobilei2v-300m", manifest.id);
        assertEquals("mobilei2v", manifest.backend);
        assertEquals(4, manifest.files.size());
        assertEquals(12288, manifest.recommendedRamMb);
        assertEquals("MIT", manifest.weightsLicense);
        assertTrue(manifest.isMobileI2V());
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsUnknownFormat() throws Exception {
        String text = "format=other\nid=x\nbackend=mobilei2v\nversion=1\nfiles=a.bin\n"
                + "sha256.a.bin=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n";
        ModelPackManifest.parse(new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsMissingChecksum() throws Exception {
        String text = "format=local-video-model-pack-v1\nid=x\nbackend=mobilei2v\nversion=1\nfiles=a.bin\n";
        ModelPackManifest.parse(new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8)));
    }

    @Test
    public void recognizesNonMobileBackendWithoutMislabeling() throws Exception {
        String text = "format=local-video-model-pack-v1\nid=x\nbackend=custom\nversion=1\nfiles=a.bin\n"
                + "sha256.a.bin=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n";
        ModelPackManifest manifest = ModelPackManifest.parse(new ByteArrayInputStream(
                text.getBytes(StandardCharsets.UTF_8)));
        assertFalse(manifest.isMobileI2V());
    }
}
