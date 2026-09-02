package com.qujindai.localvideo;

public final class UiStatePolicy {
    public enum Phase { IDLE, READY, GENERATING, SUCCESS, ERROR }

    public static final class State {
        public final String primaryLabel;
        public final String secondaryLabel;
        public final String generateLabel;
        public final String modeLabel;
        public final boolean generateEnabled;
        public final boolean clearSecondaryVisible;
        public final boolean openEnabled;
        public final boolean shareEnabled;

        State(String primaryLabel,
              String secondaryLabel,
              String generateLabel,
              String modeLabel,
              boolean generateEnabled,
              boolean clearSecondaryVisible,
              boolean openEnabled,
              boolean shareEnabled) {
            this.primaryLabel = primaryLabel;
            this.secondaryLabel = secondaryLabel;
            this.generateLabel = generateLabel;
            this.modeLabel = modeLabel;
            this.generateEnabled = generateEnabled;
            this.clearSecondaryVisible = clearSecondaryVisible;
            this.openEnabled = openEnabled;
            this.shareEnabled = shareEnabled;
        }
    }

    private UiStatePolicy() {}

    public static State resolve(boolean hasPrimary, boolean hasSecondary, Phase phase, boolean hasResult) {
        boolean generating = phase == Phase.GENERATING;
        String generateLabel;
        if (generating) {
            generateLabel = "生成中…";
        } else if (phase == Phase.SUCCESS && hasResult) {
            generateLabel = "再次生成";
        } else {
            generateLabel = "开始生成";
        }
        return new State(
                hasPrimary ? "更换主图" : "选择主图",
                hasSecondary ? "更换第二张图" : "添加第二张图（可选）",
                generateLabel,
                hasSecondary ? "双图插值" : "单图运动",
                hasPrimary && !generating,
                hasSecondary && !generating,
                hasResult && !generating,
                hasResult && !generating);
    }
}
