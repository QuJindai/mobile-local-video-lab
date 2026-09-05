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

public final class ModelPackManifest {
    public static final String FORMAT = "local-video-model-pack-v1";
    private static final Pattern TOKEN = Pattern.compile("[A-Za-z0-9._-]{1,96}");
    private static final Pattern SHA256 = Pattern.compile("[0-9a-fA-F]{64}");

    public final String id;
    public final String backend;
    public final String version;
    public final String sourceRepo;
    public final String sourceCommit;
    public final String codeLicense;
    public final String weightsLicense;
    public final int minRamMb;
    public final int recommendedRamMb;
    public final List<String> files;
    private final Map<String, String> checksums;

    private ModelPackManifest(
            String id,
            String backend,
            String version,
            String sourceRepo,
            String sourceCommit,
            String codeLicense,
            String weightsLicense,
            int minRamMb,
            int recommendedRamMb,
            List<String> files,
            Map<String, String> checksums) {
        this.id = id;
        this.backend = backend;
        this.version = version;
        this.sourceRepo = sourceRepo;
        this.sourceCommit = sourceCommit;
        this.codeLicense = codeLicense;
        this.weightsLicense = weightsLicense;
        this.minRamMb = minRamMb;
        this.recommendedRamMb = recommendedRamMb;
        this.files = Collections.unmodifiableList(new ArrayList<>(files));
        this.checksums = Collections.unmodifiableMap(new LinkedHashMap<>(checksums));
    }

    public static ModelPackManifest parse(InputStream input) throws IOException {
        Properties properties = new Properties();
        properties.load(input);

        String format = required(properties, "format");
        if (!FORMAT.equals(format)) {
            throw new IllegalArgumentException("unsupported model pack format: " + format);
        }
        String id = token(required(properties, "id"), "id");
        String backend = token(required(properties, "backend"), "backend");
        String version = token(required(properties, "version"), "version");
        String sourceRepo = properties.getProperty("source.repo", "").trim();
        String sourceCommit = properties.getProperty("source.commit", "").trim();
        String codeLicense = properties.getProperty("license.code", "unknown").trim();
        String weightsLicense = properties.getProperty("license.weights", "unknown").trim();
        int minRam = positiveOrZero(properties.getProperty("min.ram.mb", "0"), "min.ram.mb");
        int recommendedRam = positiveOrZero(
                properties.getProperty("recommended.ram.mb", String.valueOf(minRam)),
                "recommended.ram.mb");
        if (recommendedRam < minRam) {
            throw new IllegalArgumentException("recommended.ram.mb must be >= min.ram.mb");
        }

        String fileList = required(properties, "files");
        ArrayList<String> files = new ArrayList<>();
        LinkedHashMap<String, String> checksums = new LinkedHashMap<>();
        for (String part : fileList.split(",")) {
            String path = part.trim();
            if (path.isEmpty()) continue;
            if (!PackPathPolicy.isSafe(path)) {
                throw new IllegalArgumentException("unsafe model artifact path: " + path);
            }
            if (files.contains(path)) {
                throw new IllegalArgumentException("duplicate model artifact: " + path);
            }
            String checksum = required(properties, "sha256." + path).toLowerCase(Locale.US);
            if (!SHA256.matcher(checksum).matches()) {
                throw new IllegalArgumentException("invalid SHA-256 for " + path);
            }
            files.add(path);
            checksums.put(path, checksum);
        }
        if (files.isEmpty()) throw new IllegalArgumentException("model pack has no artifacts");
        if (files.size() > 128) throw new IllegalArgumentException("model pack has too many artifacts");

        return new ModelPackManifest(
                id, backend, version, sourceRepo, sourceCommit, codeLicense, weightsLicense,
                minRam, recommendedRam, files, checksums);
    }

    public boolean isMobileI2V() {
        return "mobilei2v".equalsIgnoreCase(backend);
    }

    public String expectedSha256(String path) {
        return checksums.get(path);
    }

    private static String required(Properties properties, String key) {
        String value = properties.getProperty(key);
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException("missing model pack property: " + key);
        }
        return value.trim();
    }

    private static String token(String value, String field) {
        if (!TOKEN.matcher(value).matches()) {
            throw new IllegalArgumentException("invalid model pack " + field + ": " + value);
        }
        return value;
    }

    private static int positiveOrZero(String value, String field) {
        try {
            int parsed = Integer.parseInt(value.trim());
            if (parsed < 0) throw new NumberFormatException("negative");
            return parsed;
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException("invalid " + field + ": " + value, error);
        }
    }
}
