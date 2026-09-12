package com.qujindai.localvideo;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;

import org.junit.Test;

public class AcceleratedPackManifestTest {
    private static final String HASH_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    private static final String HASH_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    private static final String HASH_C = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    private static final String HASH_D = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    @Test
    public void v2GpuPackRequiresPinnedExecutionFields() throws Exception {
        AcceleratedPackManifest manifest = AcceleratedPackManifest.parse(stream(validManifest()));

        assertTrue(manifest.isMobileI2VGpuRunnable());
        assertEquals("mnn-opencl", manifest.execution);
        assertEquals("a7666f6198412a58c6eb1eacc28828aa40c7d7ae", manifest.dreamCommit);
        assertEquals("3db3cc904dfea55286972b472b040ad5525aa083", manifest.mnnCommit);
        assertEquals("bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d",
                manifest.checkpointSha256);
        assertEquals(17, manifest.frames);
        assertEquals(1280, manifest.width);
        assertEquals(720, manifest.height);
    }

    @Test
    public void cpuExecutionIsNeverProductionRunnable() throws Exception {
        AcceleratedPackManifest manifest = AcceleratedPackManifest.parse(
                stream(validManifest().replace("execution=mnn-opencl", "execution=cpu")));
        assertFalse(manifest.isMobileI2VGpuRunnable());
    }

    @Test(expected = IllegalArgumentException.class)
    public void wrongCheckpointPinIsRejected() throws Exception {
        AcceleratedPackManifest.parse(stream(validManifest().replace(
                "bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d",
                HASH_A)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void missingRuntimeArtifactIsRejected() throws Exception {
        AcceleratedPackManifest.parse(stream(validManifest().replace(
                "denoiser.mnn,vae_encoder.mnn,vae_decoder.mnn,runtime.properties",
                "denoiser.mnn,vae_encoder.mnn,runtime.properties")));
    }

    @Test(expected = IllegalArgumentException.class)
    public void unsafeArtifactPathIsRejected() throws Exception {
        AcceleratedPackManifest.parse(stream(validManifest().replace(
                "denoiser.mnn,", "../denoiser.mnn,")));
    }

    @Test(expected = IllegalArgumentException.class)
    public void obsoletePromptContractIsRejected() throws Exception {
        AcceleratedPackManifest.parse(stream(validManifest().replace(
                "io.contract=mobilei2v-ltx-explicit-noise-v1\n", "")));
    }

    @Test
    public void unimplementedQnnIsNotRunnable() throws Exception {
        AcceleratedPackManifest manifest = AcceleratedPackManifest.parse(
                stream(validManifest().replace("execution=mnn-opencl", "execution=qnn-htp")));
        assertFalse(manifest.isMobileI2VGpuRunnable());
    }

    private static ByteArrayInputStream stream(String value) {
        return new ByteArrayInputStream(value.getBytes(StandardCharsets.UTF_8));
    }

    private static String validManifest() {
        return "format=local-video-model-pack-v2\n"
                + "id=mobilei2v-300m-gpu\n"
                + "backend=mobilei2v\n"
                + "version=0.7\n"
                + "execution=mnn-opencl\n"
                + "io.contract=mobilei2v-ltx-explicit-noise-v1\n"
                + "source.repo=hustvl/MobileI2V\n"
                + "source.commit=8d0a253c766b05a43ba408baf5e8f800a36be8b4\n"
                + "checkpoint.sha256=bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d\n"
                + "dream.source=xororz/local-dream\n"
                + "dream.commit=a7666f6198412a58c6eb1eacc28828aa40c7d7ae\n"
                + "mnn.commit=3db3cc904dfea55286972b472b040ad5525aa083\n"
                + "frames=17\n"
                + "width=1280\n"
                + "height=720\n"
                + "files=denoiser.mnn,vae_encoder.mnn,vae_decoder.mnn,runtime.properties\n"
                + "sha256.denoiser.mnn=" + HASH_A + "\n"
                + "sha256.vae_encoder.mnn=" + HASH_B + "\n"
                + "sha256.vae_decoder.mnn=" + HASH_C + "\n"
                + "sha256.runtime.properties=" + HASH_D + "\n";
    }
}
