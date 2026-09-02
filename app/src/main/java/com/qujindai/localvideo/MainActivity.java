package com.qujindai.localvideo;

import android.app.Activity;
import android.content.ClipData;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Debug;
import android.os.PowerManager;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.MediaController;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.VideoView;

import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int PICK_PRIMARY = 100;
    private static final int PICK_SECONDARY = 101;

    private static final int COLOR_ACCENT = 0xff00796b;
    private static final int COLOR_ACCENT_DARK = 0xff00695c;
    private static final int COLOR_CARD = 0xfff5f7f7;
    private static final int COLOR_MUTED = 0xff66706f;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private Uri primaryUri;
    private Uri secondaryUri;
    private Uri lastVideoUri;
    private UiStatePolicy.Phase phase = UiStatePolicy.Phase.IDLE;
    private ResultHistoryStore historyStore;
    private String lastDiagnostics = "";

    private ImageView primaryPreview;
    private TextView modeLabel;
    private TextView secondaryLabel;
    private TextView statusView;
    private TextView metricsView;
    private ProgressBar progressBar;
    private Button primaryButton;
    private Button secondaryButton;
    private Button clearSecondaryButton;
    private Button generateButton;
    private Button openButton;
    private Button shareButton;
    private Button detailsButton;
    private Button diagnosticsButton;
    private Spinner framesSpinner;
    private Spinner fpsSpinner;
    private LinearLayout resultCard;
    private LinearLayout historyContainer;
    private VideoView resultVideo;
    private TextView resultTitle;
    private boolean detailsExpanded;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        historyStore = new ResultHistoryStore(this);
        setContentView(buildUi());

        lastVideoUri = historyStore.latest();
        if (lastVideoUri != null) {
            phase = UiStatePolicy.Phase.SUCCESS;
            showResult(lastVideoUri, false);
            statusView.setText("已恢复最近生成结果，可直接播放或分享。请选择主图开始新任务。");
        } else {
            statusView.setText("选择一张主图即可开始。第二张图为可选项。");
        }
        refreshHistory();
        updateMetrics("待机");
        applyUiState();
    }

    private View buildUi() {
        int pad = dp(16);
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, dp(18), pad, dp(40));
        scroll.addView(root, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        LinearLayout titleRow = new LinearLayout(this);
        titleRow.setGravity(Gravity.CENTER_VERTICAL);
        titleRow.setOrientation(LinearLayout.HORIZONTAL);
        TextView title = text("Local Video Lab", 25, true);
        titleRow.addView(title, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        TextView badge = text("V0.2", 12, true);
        badge.setTextColor(Color.WHITE);
        badge.setGravity(Gravity.CENTER);
        badge.setBackground(rounded(COLOR_ACCENT, 20));
        badge.setPadding(dp(10), dp(5), dp(10), dp(5));
        titleRow.addView(badge);
        root.addView(titleRow);

        TextView subtitle = text(
                "完全离线 · RIFE v4.6 · ncnn · Vulkan\n单图生成轻微镜头运动，双图生成神经插值视频。", 14, false);
        subtitle.setTextColor(COLOR_MUTED);
        root.addView(subtitle);

        LinearLayout inputCard = card();
        root.addView(inputCard, cardParams());
        inputCard.addView(sectionTitle("1  输入"));

        modeLabel = text("单图运动", 13, true);
        modeLabel.setTextColor(COLOR_ACCENT_DARK);
        inputCard.addView(modeLabel);

        primaryPreview = new ImageView(this);
        primaryPreview.setBackgroundColor(0xffe5e9e8);
        primaryPreview.setScaleType(ImageView.ScaleType.CENTER_CROP);
        LinearLayout.LayoutParams previewParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(220));
        previewParams.topMargin = dp(8);
        inputCard.addView(primaryPreview, previewParams);

        primaryButton = secondaryAction("选择主图");
        primaryButton.setOnClickListener(v -> pickImage(PICK_PRIMARY));
        inputCard.addView(primaryButton);

        secondaryButton = secondaryAction("添加第二张图（可选）");
        secondaryButton.setOnClickListener(v -> pickImage(PICK_SECONDARY));
        inputCard.addView(secondaryButton);

        secondaryLabel = text("第二张：未选择 · 当前为单图运动模式", 13, false);
        secondaryLabel.setTextColor(COLOR_MUTED);
        inputCard.addView(secondaryLabel);

        clearSecondaryButton = textAction("移除第二张图");
        clearSecondaryButton.setOnClickListener(v -> {
            secondaryUri = null;
            secondaryLabel.setText("第二张：未选择 · 当前为单图运动模式");
            if (primaryUri != null) phase = UiStatePolicy.Phase.READY;
            applyUiState();
        });
        inputCard.addView(clearSecondaryButton);

        LinearLayout parametersCard = card();
        root.addView(parametersCard, cardParams());
        parametersCard.addView(sectionTitle("2  生成参数"));

        framesSpinner = spinner(new String[] { "9 帧 · 快速", "17 帧 · 推荐" }, 1);
        fpsSpinner = spinner(new String[] { "6 FPS", "8 FPS · 推荐", "12 FPS" }, 1);
        parametersCard.addView(label("帧数"));
        parametersCard.addView(framesSpinner);
        parametersCard.addView(label("播放速度"));
        parametersCard.addView(fpsSpinner);

        generateButton = primaryAction("开始生成");
        generateButton.setOnClickListener(v -> startGeneration());
        root.addView(generateButton, fullWidthParams(dp(10)));

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(100);
        progressBar.setProgressTintList(android.content.res.ColorStateList.valueOf(COLOR_ACCENT));
        LinearLayout.LayoutParams progressParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(8));
        progressParams.topMargin = dp(10);
        root.addView(progressBar, progressParams);

        statusView = text("", 14, false);
        statusView.setTextColor(COLOR_MUTED);
        root.addView(statusView);

        resultCard = card();
        resultCard.setVisibility(View.GONE);
        root.addView(resultCard, cardParams());
        resultCard.addView(sectionTitle("3  生成结果"));
        resultTitle = text("最近结果", 14, true);
        resultCard.addView(resultTitle);

        resultVideo = new VideoView(this);
        resultVideo.setBackgroundColor(Color.BLACK);
        LinearLayout.LayoutParams videoParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(230));
        videoParams.topMargin = dp(8);
        resultCard.addView(resultVideo, videoParams);

        LinearLayout resultActions = new LinearLayout(this);
        resultActions.setOrientation(LinearLayout.HORIZONTAL);
        resultActions.setPadding(0, dp(8), 0, 0);
        openButton = secondaryAction("打开视频");
        shareButton = primaryAction("分享视频");
        openButton.setOnClickListener(v -> openLastVideo());
        shareButton.setOnClickListener(v -> shareLastVideo());
        resultActions.addView(openButton, weightedButtonParams(1f, 0));
        resultActions.addView(shareButton, weightedButtonParams(1f, dp(8)));
        resultCard.addView(resultActions);

        LinearLayout historyCard = card();
        root.addView(historyCard, cardParams());
        historyCard.addView(sectionTitle("最近生成"));
        TextView historyHint = text("保留最近 5 个结果，重启 App 后仍可打开。", 12, false);
        historyHint.setTextColor(COLOR_MUTED);
        historyCard.addView(historyHint);
        historyContainer = new LinearLayout(this);
        historyContainer.setOrientation(LinearLayout.VERTICAL);
        historyCard.addView(historyContainer);

        detailsButton = textAction("查看详细信息");
        detailsButton.setOnClickListener(v -> {
            detailsExpanded = !detailsExpanded;
            metricsView.setVisibility(detailsExpanded ? View.VISIBLE : View.GONE);
            detailsButton.setText(detailsExpanded ? "收起详细信息" : "查看详细信息");
        });
        root.addView(detailsButton);

        metricsView = text("", 12, false);
        metricsView.setTextColor(COLOR_MUTED);
        metricsView.setTextIsSelectable(true);
        metricsView.setVisibility(View.GONE);
        root.addView(metricsView);

        diagnosticsButton = textAction("导出诊断信息");
        diagnosticsButton.setVisibility(View.GONE);
        diagnosticsButton.setOnClickListener(v -> shareDiagnostics());
        root.addView(diagnosticsButton);
        return scroll;
    }

    private void pickImage(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION |
                Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        startActivityForResult(intent, requestCode);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        try {
            int grant = data.getFlags() & Intent.FLAG_GRANT_READ_URI_PERMISSION;
            if (grant != 0) getContentResolver().takePersistableUriPermission(uri, grant);
        } catch (SecurityException ignored) {
            // The immediate read grant is enough for this generation session.
        }

        if (requestCode == PICK_PRIMARY) {
            primaryUri = uri;
            primaryPreview.setImageURI(uri);
            phase = UiStatePolicy.Phase.READY;
            statusView.setText("主图已就绪。可直接生成，或添加第二张图。 ");
        } else if (requestCode == PICK_SECONDARY) {
            secondaryUri = uri;
            if (primaryUri != null) phase = UiStatePolicy.Phase.READY;
            secondaryLabel.setText("第二张：已选择 · 当前为双图插值模式");
            statusView.setText("双图插值模式已就绪。 ");
        }
        diagnosticsButton.setVisibility(View.GONE);
        applyUiState();
    }

    private void startGeneration() {
        if (primaryUri == null) {
            statusView.setText("请先选择主图。 ");
            return;
        }
        final int frames = framesSpinner.getSelectedItemPosition() == 0 ? 9 : 17;
        final int fps;
        switch (fpsSpinner.getSelectedItemPosition()) {
            case 0: fps = 6; break;
            case 2: fps = 12; break;
            default: fps = 8; break;
        }

        phase = UiStatePolicy.Phase.GENERATING;
        progressBar.setProgress(0);
        statusView.setText("0% · 正在准备本地运行时");
        diagnosticsButton.setVisibility(View.GONE);
        applyUiState();
        final int thermalBefore = thermalStatus();

        worker.submit(() -> {
            try {
                RifeEngine engine = new RifeEngine(this);
                RifeEngine.Result result = engine.generate(primaryUri, secondaryUri, frames, fps,
                        (percent, message) -> runOnUiThread(() -> {
                            progressBar.setProgress(percent);
                            statusView.setText(percent + "% · " + message);
                        }));

                int thermalAfter = thermalStatus();
                lastVideoUri = result.uri;
                historyStore.record(result.uri);
                lastDiagnostics = formatMetrics(result, thermalBefore, thermalAfter);
                runOnUiThread(() -> {
                    phase = UiStatePolicy.Phase.SUCCESS;
                    progressBar.setProgress(100);
                    statusView.setText("100% · 已保存到 Movies/LocalVideoLab");
                    metricsView.setText(lastDiagnostics);
                    showResult(result.uri, true);
                    refreshHistory();
                    applyUiState();
                });
            } catch (Throwable error) {
                String diagnostics = formatError(error);
                runOnUiThread(() -> {
                    phase = UiStatePolicy.Phase.ERROR;
                    statusView.setText("生成失败 · " + safeMessage(error));
                    lastDiagnostics = diagnostics;
                    metricsView.setText(diagnostics);
                    detailsExpanded = true;
                    metricsView.setVisibility(View.VISIBLE);
                    detailsButton.setText("收起详细信息");
                    diagnosticsButton.setVisibility(View.VISIBLE);
                    applyUiState();
                });
            }
        });
    }

    private void showResult(Uri uri, boolean autoPlay) {
        if (uri == null) return;
        lastVideoUri = uri;
        resultCard.setVisibility(View.VISIBLE);
        resultTitle.setText(autoPlay ? "刚刚生成" : "最近结果");

        MediaController controller = new MediaController(this);
        controller.setAnchorView(resultVideo);
        resultVideo.setMediaController(controller);
        resultVideo.setVideoURI(uri);
        resultVideo.setOnPreparedListener(player -> {
            player.setLooping(true);
            if (autoPlay) {
                resultVideo.start();
            } else {
                resultVideo.seekTo(1);
            }
        });
        resultVideo.setOnErrorListener((MediaPlayer mp, int what, int extra) -> {
            statusView.setText("视频预览不可用，可尝试“打开视频”交给系统播放器。 ");
            return true;
        });
        resultVideo.requestFocus();
    }

    private void openLastVideo() {
        if (lastVideoUri == null) return;
        try {
            Intent intent = new Intent(Intent.ACTION_VIEW);
            intent.setDataAndType(lastVideoUri, "video/mp4");
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            intent.setClipData(ClipData.newRawUri("generated-video", lastVideoUri));
            startActivity(Intent.createChooser(intent, "打开视频"));
        } catch (RuntimeException error) {
            statusView.setText("没有可用的视频播放器：" + safeMessage(error));
        }
    }

    private void shareLastVideo() {
        if (lastVideoUri == null) return;
        try {
            Intent intent = new Intent(Intent.ACTION_SEND);
            intent.setType("video/mp4");
            intent.putExtra(Intent.EXTRA_STREAM, lastVideoUri);
            intent.setClipData(ClipData.newRawUri("generated-video", lastVideoUri));
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(Intent.createChooser(intent, "分享生成视频"));
        } catch (RuntimeException error) {
            statusView.setText("无法调用系统分享：" + safeMessage(error));
        }
    }

    private void shareDiagnostics() {
        if (lastDiagnostics == null || lastDiagnostics.isEmpty()) return;
        Intent intent = new Intent(Intent.ACTION_SEND);
        intent.setType("text/plain");
        intent.putExtra(Intent.EXTRA_SUBJECT, "Local Video Lab V0.2 diagnostics");
        intent.putExtra(Intent.EXTRA_TEXT, lastDiagnostics);
        startActivity(Intent.createChooser(intent, "导出诊断信息"));
    }

    private void refreshHistory() {
        historyContainer.removeAllViews();
        List<Uri> recent = historyStore.load();
        if (recent.isEmpty()) {
            TextView empty = text("暂无历史结果", 13, false);
            empty.setTextColor(COLOR_MUTED);
            historyContainer.addView(empty);
            return;
        }
        for (int i = 0; i < recent.size(); i++) {
            Uri uri = recent.get(i);
            Button item = textAction((i == 0 ? "最近一次" : "历史结果 " + (i + 1)) + " · 点击播放");
            item.setGravity(Gravity.START | Gravity.CENTER_VERTICAL);
            item.setOnClickListener(v -> {
                phase = UiStatePolicy.Phase.SUCCESS;
                showResult(uri, true);
                statusView.setText("正在播放已保存结果。 ");
                applyUiState();
            });
            historyContainer.addView(item);
        }
    }

    private void applyUiState() {
        UiStatePolicy.State state = UiStatePolicy.resolve(
                primaryUri != null,
                secondaryUri != null,
                phase,
                lastVideoUri != null);
        primaryButton.setText(state.primaryLabel);
        secondaryButton.setText(state.secondaryLabel);
        generateButton.setText(state.generateLabel);
        modeLabel.setText(state.modeLabel);
        generateButton.setEnabled(state.generateEnabled);
        primaryButton.setEnabled(phase != UiStatePolicy.Phase.GENERATING);
        secondaryButton.setEnabled(phase != UiStatePolicy.Phase.GENERATING);
        clearSecondaryButton.setVisibility(state.clearSecondaryVisible ? View.VISIBLE : View.GONE);
        framesSpinner.setEnabled(phase != UiStatePolicy.Phase.GENERATING);
        fpsSpinner.setEnabled(phase != UiStatePolicy.Phase.GENERATING);
        openButton.setEnabled(state.openEnabled);
        shareButton.setEnabled(state.shareEnabled);
        resultCard.setVisibility(lastVideoUri == null ? View.GONE : View.VISIBLE);
    }

    private String formatMetrics(RifeEngine.Result result, int thermalBefore, int thermalAfter) {
        return String.format(Locale.US,
                "模型: RIFE v4.6 / ncnn / Vulkan\n" +
                        "模式: %s\n" +
                        "输出: %dx%d · %d 帧 · %d FPS\n" +
                        "耗时: %.2f s\n" +
                        "热状态: %s → %s\n" +
                        "Android API: %d\n" +
                        "Java max heap: %.0f MB · native heap: %.0f MB",
                result.singleImageMode ? "单图运动" : "双图插值",
                result.width, result.height, result.frames, result.fps,
                result.elapsedMs / 1000.0,
                thermalName(thermalBefore), thermalName(thermalAfter),
                Build.VERSION.SDK_INT,
                Runtime.getRuntime().maxMemory() / 1048576.0,
                Debug.getNativeHeapAllocatedSize() / 1048576.0);
    }

    private String formatError(Throwable error) {
        return String.format(Locale.US,
                "Local Video Lab V0.2\n" +
                        "状态: 生成失败\n" +
                        "异常: %s\n" +
                        "信息: %s\n" +
                        "模式: %s\n" +
                        "Android API: %d\n" +
                        "热状态: %s\n" +
                        "Java max heap: %.0f MB\n" +
                        "native heap: %.0f MB",
                error.getClass().getName(),
                safeMessage(error),
                secondaryUri == null ? "单图运动" : "双图插值",
                Build.VERSION.SDK_INT,
                thermalName(thermalStatus()),
                Runtime.getRuntime().maxMemory() / 1048576.0,
                Debug.getNativeHeapAllocatedSize() / 1048576.0);
    }

    private void updateMetrics(String state) {
        metricsView.setText(String.format(Locale.US,
                "%s\n设备 API: %d · 热状态: %s\nJava max heap: %.0f MB · native heap: %.0f MB",
                state, Build.VERSION.SDK_INT, thermalName(thermalStatus()),
                Runtime.getRuntime().maxMemory() / 1048576.0,
                Debug.getNativeHeapAllocatedSize() / 1048576.0));
    }

    private int thermalStatus() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return -1;
        PowerManager manager = getSystemService(PowerManager.class);
        return manager == null ? -1 : manager.getCurrentThermalStatus();
    }

    private static String thermalName(int status) {
        switch (status) {
            case PowerManager.THERMAL_STATUS_NONE: return "NONE";
            case PowerManager.THERMAL_STATUS_LIGHT: return "LIGHT";
            case PowerManager.THERMAL_STATUS_MODERATE: return "MODERATE";
            case PowerManager.THERMAL_STATUS_SEVERE: return "SEVERE";
            case PowerManager.THERMAL_STATUS_CRITICAL: return "CRITICAL";
            case PowerManager.THERMAL_STATUS_EMERGENCY: return "EMERGENCY";
            case PowerManager.THERMAL_STATUS_SHUTDOWN: return "SHUTDOWN";
            default: return "N/A";
        }
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty()
                ? error.getClass().getSimpleName()
                : message;
    }

    private Spinner spinner(String[] values, int selected) {
        Spinner spinner = new Spinner(this);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, values);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
        spinner.setSelection(selected);
        spinner.setPadding(dp(6), 0, dp(6), 0);
        return spinner;
    }

    private LinearLayout card() {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(14), dp(12), dp(14), dp(12));
        card.setBackground(rounded(COLOR_CARD, 16));
        return card;
    }

    private LinearLayout.LayoutParams cardParams() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.topMargin = dp(12);
        return params;
    }

    private Button primaryAction(String label) {
        Button button = button(label);
        button.setTextColor(Color.WHITE);
        button.setBackground(rounded(COLOR_ACCENT, 12));
        button.setMinHeight(dp(52));
        return button;
    }

    private Button secondaryAction(String label) {
        Button button = button(label);
        button.setTextColor(0xff263238);
        button.setBackground(rounded(0xffdde3e2, 12));
        button.setMinHeight(dp(50));
        LinearLayout.LayoutParams params = fullWidthParams(dp(8));
        button.setLayoutParams(params);
        return button;
    }

    private Button textAction(String label) {
        Button button = button(label);
        button.setTextColor(COLOR_ACCENT_DARK);
        button.setBackgroundColor(Color.TRANSPARENT);
        button.setGravity(Gravity.CENTER_VERTICAL);
        return button;
    }

    private Button button(String label) {
        Button button = new Button(this);
        button.setText(label);
        button.setTextSize(15);
        button.setAllCaps(false);
        return button;
    }

    private TextView sectionTitle(String value) {
        TextView view = text(value, 18, true);
        view.setTextColor(0xff263238);
        return view;
    }

    private TextView label(String value) {
        TextView view = text(value, 12, true);
        view.setTextColor(COLOR_MUTED);
        view.setPadding(0, dp(7), 0, 0);
        return view;
    }

    private TextView text(String value, int sp, boolean bold) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(sp);
        if (bold) view.setTypeface(view.getTypeface(), Typeface.BOLD);
        view.setPadding(0, dp(5), 0, dp(5));
        return view;
    }

    private GradientDrawable rounded(int color, int radiusDp) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(color);
        drawable.setCornerRadius(dp(radiusDp));
        return drawable;
    }

    private LinearLayout.LayoutParams fullWidthParams(int topMargin) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.topMargin = topMargin;
        return params;
    }

    private LinearLayout.LayoutParams weightedButtonParams(float weight, int leftMargin) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, weight);
        params.leftMargin = leftMargin;
        return params;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    @Override
    protected void onDestroy() {
        try { resultVideo.stopPlayback(); } catch (Exception ignored) {}
        worker.shutdownNow();
        super.onDestroy();
    }
}
