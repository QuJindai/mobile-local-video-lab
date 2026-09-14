package com.qujindai.facere;

public final class DownloadResumePolicy {
    public enum Action { APPEND, RESTART, VERIFY_EXISTING }

    private DownloadResumePolicy() {}

    public static Action action(long existingBytes, int httpCode) {
        if (existingBytes > 0L && httpCode == 206) return Action.APPEND;
        if (existingBytes > 0L && httpCode == 416) return Action.VERIFY_EXISTING;
        return Action.RESTART;
    }
}
