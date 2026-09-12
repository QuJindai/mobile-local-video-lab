package com.qujindai.localvideo;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.regex.Pattern;

/**
 * Strict manifest for real accelerated MobileI2V runtime packs.
 *
 * A downloaded upstream .pth checkpoint is deliberately not represented by this
 * type. This contract is only for Android-executable artifacts whose provenance
 * and acceleration backend can be verified before the app marks MobileI2V READY.
 */
public final class AcceleratedPackManifest {
    public static final String FORMAT = "local-video-model-pack-v2";
    public static final String MOBILE_BACKEND = "mobilei2v";
    public static final String EXECUTION_MNN_OPENCL = "mnn-opencl";
    public static final String EXECUTION_QNN_HTP = "qnn-htp";

    public static final String SOURCE_REPO = "hustvl/MobileI2V";
    public static final String SOURCE_COMMIT = "8d0a253c766b05a43ba408baf5e8f800a36be8b4";
    public static final String CHECKPOINT_SHA256 =
            "bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d";
    public static final String DREAM_SOURCE = "xororz/local-dream";
    public static final String DREAM_COMMIT = "a7666f6198412a58c6eb1eacc28828aa40c7d7ae";
    public static final String MNN_COMMIT = "3db3cc904dfea55286972b472b040ad5525aa083";

    public static final int FRAMES = 17;
    public static final int WIDTH = 1280;
    public static final int HEIGHT = 720;

    private static final Pattern TOKEN = Pattern.compile("[A-Za-z0-9._-]{1,96}");
    private static final Pattern SHA256 = Pattern.compile("[0-9a-fA-F]{64}");
    private static final String[] REQUIRED_RUNTIME_FILES = new String[] {
            "denoiser.mnn",
            "vae_encoder.mnn",
            "vae_decoder.mnn",
            "empty_prompt.f16",
            "empty_prompt_mask.bin"
    };

    public final String id;
    public final String backend;
    public final String version;
    public final String execution;
    public final String sourceRepo;
    public final String sourceCommit;
    public final String checkpointSha256;
    public final String dreamSource;
    public final String dreamCommit;
    public final String mnnCommit;
    public final int frames;
    public final int width;
    public final int height;
    public final List<String> files;
    private final Map<String, String> checksums;

    private AcceleratedPackManifest(
            String id,
            String backend,
            String version,
            String execution,
            String sourceRepo,
            String sourceCommit,
            String checkpointSha256,
            String dreamSource,
            String dreamCommit,
            String mnnCommit,
            int frames,
            int width,
            int height,
            List<String> files,
            Map<String, String> checksums) {
        this.id = id;
        this.backend = backend;
        this.version = version;
        this.execution = execution;
        this.sourceRepo = sourceRepo;
        this.sourceCommit = sourceCommit;
        this.checkpointSha256 = checkpointSha256;
        this.dreamSource = dreamSource;
        this.dreamCommit = dreamCommit;
        this.mnnCommit = mnnCommit;
        this.frames = frames;
        this.width = width;
        this.height = height;
        this.files = Collections.unmodifiableList(new ArrayList<>(files));
        this.checksums = Collections.unmodifiableMap(new LinkedHashMap<>(checksums));
    }

    public static AcceleratedPackManifest parse(InputStream input) throws IOException {
        Properties properties = new Properties();
        properties.load(input);

        requireEquals(properties, "format", FORMAT);
        String id = token(required(properties, "id"), "id");
        String backend = token(required(properties, "backend"), "backend");
        String version = token(required(properties, "version"), "version");
        String execution = token(required(properties, "execution"), "execution");

        String sourceRepo = required(properties, "source.repo");
        String sourceCommit = required(properties, "source.commit").toLowerCase(Locale.US);
        String checkpointSha = required(properties, "checkpoint.sha256").toLowerCase(Locale.US);
        String dreamSource = required(properties, "dream.source");
        String dreamCommit = required(properties, "dream.commit").toLowerCase(Locale.US);
        String mnnCommit = required(properties, "mnn.commit").toLowerCase(Locale.US);

        requirePinned("source.repo", sourceRepo, SOURCE_REPO);
        requirePinned("source.commit", sourceCommit, SOURCE_COMMIT);
        requirePinned("checkpoint.sha256", checkpointSha, CHECKPOINT_SHA256);
        requirePinned("dream.source", dreamSource, DREAM_SOURCE);
        requirePinned("dream.commit", dreamCommit, DREAM_COMMIT);
        requirePinned("mnn.commit", mnnCommit, MNN_COMMIT);

        int frames = positive(properties, "frames");
        int width = positive(properties, "width");
        int height = positive(properties, "height");
        if (frames != FRAMES || width != WIDTH || height != HEIGHT) {
            throw new IllegalArgumentException(
                    "unsupported MobileI2V runtime shape: " + frames + "x" + width + "x" + height);
        }

        ArrayList<String> files = new ArrayList<>();
        LinkedHashMap<String, String> hashes = new LinkedHashMap<>();
        for (String item : required(properties, "files").split(",")) {
            String path = item.trim();
            if (path.isEmpty()) continue;
            if (!PackPathPolicy.isSafe(path)) {
                throw new IllegalArgumentException("unsafe model artifact path: " + path);
            }
            if (files.contains(path)) {
                throw new IllegalArgumentException("duplicate model artifact: " + path);
            }
            String hash = required(properties, "sha256." + path).toLowerCase(Locale.US);
            if (!SHA256.matcher(hash).matches()) {
                throw new IllegalArgumentException("invalid SHA-256 for " + path);
            }
            files.add(path);
            hashes.put(path, hash);
        }
        if (files.isEmpty() || files.size() > 128) {
            throw new IllegalArgumentException("invalid accelerated artifact count: " + files.size());
        }
        for (String required : REQUIRED_RUNTIME_FILES) {
            if (!files.contains(required)) {
                throw new IllegalArgumentException("required accelerated artifact missing: " + required);
            }
        }

        return new AcceleratedPackManifest(
                id, backend, version, execution,
                sourceRepo, sourceCommit, checkpointSha,
                dreamSource, dreamCommit, mnnCommit,
                frames, width, height, files, hashes);
    }

    public boolean isMobileI2VGpuRunnable() {
        if (!MOBILE_BACKEND.equalsIgnoreCase(backend)) return false;
        return EXECUTION_MNN_OPENCL.equalsIgnoreCase(execution)
                || EXECUTION_QNN_HTP.equalsIgnoreCase(execution);
    }

    public String expectedSha256(String path) {
        return checksums.get(path);
    }

    private static void requireEquals(Properties properties, String key, String expected) {
        requirePinned(key, required(properties, key), expected);
    }

    private static void requirePinned(String field, String actual, String expected) {
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException(
                    "unexpected " + field + ": " + actual + " (expected " + expected + ")");
        }
    }

    private static String required(Properties properties, String key) {
        String value = properties.getProperty(key);
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException("missing accelerated pack property: " + key);
        }
        return value.trim();
    }

    private static String token(String value, String field) {
        if (!TOKEN.matcher(value).matches()) {
            throw new IllegalArgumentException("invalid accelerated pack " + field + ": " + value);
        }
        return value;
    }

    private static int positive(Properties properties, String key) {
        try {
            int value = Integer.parseInt(required(properties, key));
            if (value <= 0) throw new NumberFormatException("not positive");
            return value;
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException("invalid accelerated pack " + key, error);
        }
    }
}
