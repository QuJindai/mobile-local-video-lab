package com.qujindai.localvideo;

public final class GenerationPlan {
    public static final int MAX_LONG_EDGE = 720;

    private final int width;
    private final int height;
    private final int frames;
    private final int fps;

    public GenerationPlan(int width, int height, int frames, int fps) {
        if (width < 64 || height < 64 || width % 16 != 0 || height % 16 != 0) {
            throw new IllegalArgumentException("width/height must be >=64 and aligned to 16");
        }
        if (Math.max(width, height) > MAX_LONG_EDGE) {
            throw new IllegalArgumentException("long edge exceeds mobile limit");
        }
        if (frames < 3 || frames % 2 == 0) {
            throw new IllegalArgumentException("frame count must be odd and >=3");
        }
        if (fps < 4 || fps > 30) {
            throw new IllegalArgumentException("fps must be within 4..30");
        }
        this.width = width;
        this.height = height;
        this.frames = frames;
        this.fps = fps;
    }

    public int getWidth() { return width; }
    public int getHeight() { return height; }
    public int getFrames() { return frames; }
    public int getFps() { return fps; }
    public double getDurationSeconds() { return frames / (double) fps; }
}
