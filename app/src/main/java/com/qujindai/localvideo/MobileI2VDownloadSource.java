package com.qujindai.localvideo;

public final class MobileI2VDownloadSource {
    public enum Kind {
        OFFICIAL,
        CHINA_MIRROR
    }

    private static final String REPO = "hustvl/MobileI2V";
    private static final String REVISION = "290b2d0dfa93c65388b93e3f7591d7328b335e65";
    private static final String FILE_NAME = "hybrid_371.pth";
    private static final long EXPECTED_BYTES = 1_074_370_038L;
    private static final String EXPECTED_SHA256 =
            "bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d";

    public final Kind kind;
    public final String label;
    public final String endpoint;
    public final String revision;
    public final String fileName;
    public final long expectedBytes;
    public final String expectedSha256;

    private MobileI2VDownloadSource(Kind kind, String label, String endpoint) {
        this.kind = kind;
        this.label = label;
        this.endpoint = endpoint;
        this.revision = REVISION;
        this.fileName = FILE_NAME;
        this.expectedBytes = EXPECTED_BYTES;
        this.expectedSha256 = EXPECTED_SHA256;
    }

    public static MobileI2VDownloadSource official() {
        return new MobileI2VDownloadSource(
                Kind.OFFICIAL,
                "Hugging Face 官方原版",
                "https://huggingface.co");
    }

    public static MobileI2VDownloadSource chinaMirror() {
        return new MobileI2VDownloadSource(
                Kind.CHINA_MIRROR,
                "国内 HF-Mirror",
                "https://hf-mirror.com");
    }

    public String downloadUrl() {
        return endpoint + "/" + REPO + "/resolve/" + revision + "/" + fileName + "?download=true";
    }
}
