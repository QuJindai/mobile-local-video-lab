package com.qujindai.facere;

import java.net.URI;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public final class FaceModelDownloadCatalog {
    public enum SourceMode { AUTO, CHINA_MIRROR, OFFICIAL }

    public static final String BUFFALO_ZIP_SHA256 =
            "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f";
    public static final String DETECTOR_SHA256 =
            "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91";
    public static final String RECOGNIZER_SHA256 =
            "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43";
    public static final String SWAPPER_SHA256 =
            "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af";

    public static final long BUFFALO_ZIP_BYTES = 288_621_354L;
    public static final long SWAPPER_BYTES = 554_253_681L;

    public static final String OFFICIAL_BUFFALO_URL =
            "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip";
    public static final String OFFICIAL_SWAPPER_URL =
            "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx";

    public static final String MIRROR_DETECTOR_URL =
            "https://hf-mirror.com/deepghs/insightface/resolve/main/buffalo_l/det_10g.onnx?download=true";
    public static final String MIRROR_RECOGNIZER_URL =
            "https://hf-mirror.com/deepghs/insightface/resolve/main/buffalo_l/w600k_r50.onnx?download=true";
    public static final String MIRROR_SWAPPER_URL =
            "https://hf-mirror.com/yongliangxie/insightface/resolve/main/inswapper_128.onnx?download=true";

    private FaceModelDownloadCatalog() {}

    public static List<SourceMode> attemptOrder(SourceMode mode) {
        if (mode == null || mode == SourceMode.AUTO) {
            return Arrays.asList(SourceMode.CHINA_MIRROR, SourceMode.OFFICIAL);
        }
        return Collections.singletonList(mode);
    }

    public static boolean isTrustedUrl(String value) {
        if (value == null || value.isEmpty()) return false;
        try {
            URI uri = URI.create(value);
            if (!"https".equalsIgnoreCase(uri.getScheme())) return false;
            String host = uri.getHost();
            if (host == null) return false;
            host = host.toLowerCase(java.util.Locale.US);
            return host.equals("github.com")
                    || host.equals("objects.githubusercontent.com")
                    || host.equals("release-assets.githubusercontent.com")
                    || host.equals("hf-mirror.com")
                    || host.endsWith(".hf-mirror.com");
        } catch (IllegalArgumentException error) {
            return false;
        }
    }
}
