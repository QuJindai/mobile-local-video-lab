package com.qujindai.localvideo;

import java.nio.charset.StandardCharsets;
import java.util.Base64;

public final class ResultRecord {
    public final String uri;
    public final long createdAtMs;
    public final long durationMs;
    public final int width;
    public final int height;
    public final int frames;
    public final int fps;
    public final String recipe;

    public ResultRecord(String uri, long createdAtMs, long durationMs,
                        int width, int height, int frames, int fps) {
        this(uri, createdAtMs, durationMs, width, height, frames, fps, "");
    }

    public ResultRecord(String uri, long createdAtMs, long durationMs,
                        int width, int height, int frames, int fps, String recipe) {
        if (uri == null || uri.isEmpty()) throw new IllegalArgumentException("uri required");
        this.uri = uri;
        this.createdAtMs = Math.max(0L, createdAtMs);
        this.durationMs = Math.max(0L, durationMs);
        this.width = Math.max(0, width);
        this.height = Math.max(0, height);
        this.frames = Math.max(0, frames);
        this.fps = Math.max(0, fps);
        this.recipe = recipe == null ? "" : recipe;
    }

    public String encode() {
        String encodedUri = Base64.getUrlEncoder().withoutPadding()
                .encodeToString(uri.getBytes(StandardCharsets.UTF_8));
        return createdAtMs + "|" + durationMs + "|" + width + "|" + height + "|"
                + frames + "|" + fps + "|" + encodedUri
                + (recipe.isEmpty() ? "" : "|" + Base64.getUrlEncoder().withoutPadding()
                        .encodeToString(recipe.getBytes(StandardCharsets.UTF_8)));
    }

    public static ResultRecord decode(String encoded) {
        if (encoded == null || encoded.isEmpty()) return null;
        try {
            String[] parts = encoded.split("\\|", -1);
            if (parts.length != 7 && parts.length != 8) return null;
            String uri = new String(Base64.getUrlDecoder().decode(parts[6]), StandardCharsets.UTF_8);
            return new ResultRecord(
                    uri,
                    Long.parseLong(parts[0]),
                    Long.parseLong(parts[1]),
                    Integer.parseInt(parts[2]),
                    Integer.parseInt(parts[3]),
                    Integer.parseInt(parts[4]),
                    Integer.parseInt(parts[5]),
                    parts.length == 8 ? new String(Base64.getUrlDecoder().decode(parts[7]),
                            StandardCharsets.UTF_8) : "");
        } catch (RuntimeException ignored) {
            return null;
        }
    }

    public static ResultRecord fromLegacyUri(String uri) {
        return new ResultRecord(uri, 0L, 0L, 0, 0, 0, 0);
    }
}
