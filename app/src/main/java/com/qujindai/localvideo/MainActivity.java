package com.qujindai.localvideo;

import android.app.Activity;
import android.content.ClipData;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Debug;
import android.os.PowerManager;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int PICK_PRIMARY = 100;
    private static final int PICK_SECONDARY = 101;
    private static final int PICK_MODEL_PACK = 102;

    private static final int COLOR_ACCENT = 0xff00796b;
    private static final int COLOR_ACCENT_DARK = 0xff00695c;
    private static final int COLOR_CARD = 0xfff5f7f7;
    private static final int COLOR_MUTED = 0xff66706f;
    private static final int COLOR_WARNING = 0xff8a5a00;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private Uri primaryUri;
    private Uri secondaryUri;
    private Uri lastVideoUri;
    private ResultRecord lastRecord;
    private UiStatePolicy.Phase phase = UiStatePolicy.Phase.IDLE;
    private ResultHistoryStore historyStore;
    private ModelPackStore modelPackStore;
    private InstalledModelPack mobilePack;
    private DeviceCapabilitySnapshot capabilities;
    private OnnxRuntimeFoundation.Status onnxStatus;
    private String lastDiagnostics = "";
    private boolean modelInstalling;

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
    private Button importModelButton;
    private Button removeModelButton;
    private Spinner framesSpinner;
    private Spinner fpsSpinner;
    private Spinner motionSpinner;
    private Spinner backendSpinner;
    private TextView backendStatusView;
    private TextView modelPackView;
    private TextView deviceView;
    private TextView motionHint;
    private LinearLayout resultCard;
    private LinearLayout historyContainer;
    private ImageView resultThumbnail;
    private TextView resultTitle;
    private TextView resultMeta;
    private TextView resultPlayOverlay;
    private boolean detailsExpanded;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        historyStore = new ResultHistoryStore(this);
        modelPackStore = new ModelPackStore(this);
        capabilities = DeviceCapabilitySnapshot.capture(this);
        onnxStatus = OnnxRuntimeFoundation.probe();
        mobilePack = modelPackStore.loadMobileI2V();
        setContentView(buildUi());

        lastRecord = historyStore.latestRecord();
        if (lastRecord != null) {
            lastVideoUri = Uri.parse(lastRecord.uri);
            phase = UiStatePolicy.Phase.SUCCESS;
            showResult(lastRecord, false);
            statusView.setText("已恢复最近生成结果。点击缩略图可全屏播放。");
        } else {
            statusView.setText("RIFE 后端已就绪。MobileI2V 可在 Model Lab 中安装模型包。 ");
        }
        refreshHistory();
        refreshBackendStatus();
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
        TextView badge = text("V0.4", 12, true);
        badge.setTextColor(Color.WHITE);
        badge.setGravity(Gravity.CENTER);
        badge.setBackground(rounded(COLOR_ACCENT, 20));
        badge.setPadding(dp(10), dp(5), dp(10), dp(5));
        titleRow.addView(badge);
        root.addView(titleRow);

        TextView subtitle = text(
                "完全离线 · 可插拔本地视频模型\n"
                        + "RIFE 为已验证稳定后端；MobileI2V 0.27B 进入独立模型包与 Android runtime 部署链。",
                14, false);
        subtitle.setTextColor(COLOR_MUTED);
        root.addView(subtitle);

        LinearLayout modelCard = card();
        root.addView(modelCard, cardParams());
        modelCard.addView(sectionTitle("0  模型后端 · Model Lab"));
        modelCard.addView(label("选择后端"));
        backendSpinner = spinner(new String[] {
                "RIFE Motion · 已验证",
                "MobileI2V 0.27B · 语义 I2V 实验"
        }, 0);
        modelCard.addView(backendSpinner);

        backendStatusView = text("", 13, true);
        modelCard.addView(backendStatusView);
        modelPackView = text("", 12, false);
        modelPackView.setTextColor(COLOR_MUTED);
        modelCard.addView(modelPackView);

        importModelButton = secondaryAction("导入 MobileI2V 模型包 (.mlvpkg)");
        importModelButton.setOnClickListener(v -> pickModelPack());
        modelCard.addView(importModelButton);
        removeModelButton = textAction("移除已安装的 MobileI2V 模型包");
        removeModelButton.setOnClickListener(v -> removeMobileModelPack());
        modelCard.addView(removeModelButton);

        TextView capabilityTitle = label("设备 / Runtime 能力");
        modelCard.addView(capabilityTitle);
        deviceView = text("", 12, false);
        deviceView.setTextColor(COLOR_MUTED);
        modelCard.addView(deviceView);

        TextView honestyNote = text(
                "MobileI2V 只有在“模型包 + Android 执行 runtime + 设备门槛”全部通过后才允许生成；"
                        + "未就绪时不会回退成 RIFE 冒充语义 I2V。",
                12, false);
        honestyNote.setTextColor(COLOR_WARNING);
        modelCard.addView(honestyNote);

        LinearLayout inputCard = card();
        root.addView(inputCard, cardParams());
        inputCard.addView(sectionTitle("1  输入"));

        modeLabel = text("RIFE · 单图运动", 13, true);
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
        motionSpinner = spinner(new String[] {
                "电影自动 · 推荐",
                "推近",
                "向左平移",
                "向上漂移"
        }, 0);
        parametersCard.addView(label("帧数"));
        parametersCard.addView(framesSpinner);
        parametersCard.addView(label("播放速度"));
        parametersCard.addView(fpsSpinner);
        parametersCard.addView(label("RIFE 单图运动方式"));
        parametersCard.addView(motionSpinner);
        motionHint = text("RIFE 单图模式生效；双图模式直接用两张输入图插值。", 12, false);
        motionHint.setTextColor(COLOR_MUTED);
        parametersCard.addView(motionHint);

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

        FrameLayout previewFrame = new FrameLayout(this);
        LinearLayout.LayoutParams videoParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(230));
        videoParams.topMargin = dp(8);
        resultCard.addView(previewFrame, videoParams);

        resultThumbnail = new ImageView(this);
        resultThumbnail.setBackgroundColor(0xffdfe4e3);
        resultThumbnail.setScaleType(ImageView.ScaleType.CENTER_CROP);
        previewFrame.addView(resultThumbnail, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        resultPlayOverlay = text("▶  全屏播放", 16, true);
        resultPlayOverlay.setTextColor(Color.WHITE);
        resultPlayOverlay.setGravity(Gravity.CENTER);
        resultPlayOverlay.setBackground(rounded(0xaa000000, 24));
        resultPlayOverlay.setPadding(dp(18), dp(10), dp(18), dp(10));
        FrameLayout.LayoutParams overlayParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        overlayParams.gravity = Gravity.CENTER;
        previewFrame.addView(resultPlayOverlay, overlayParams);
        previewFrame.setOnClickListener(v -> openLastVideo());

        resultMeta = text("", 13, false);
        resultMeta.setTextColor(COLOR_MUTED);
        resultCard.addView(resultMeta);

        LinearLayout resultActions = new LinearLayout(this);
        resultActions.setOrientation(LinearLayout.HORIZONTAL);
        resultActions.setPadding(0, dp(8), 0, 0);
        openButton = secondaryAction("全屏播放");
        shareButton = primaryAction("分享视频");
        openButton.setOnClickListener(v -> openLastVideo());
        shareButton.setOnClickListener(v -> shareLastVideo());
        resultActions.addView(openButton, weightedButtonParams(1f, 0));
        resultActions.addView(shareButton, weightedButtonParams(1f, dp(8)));
        resultCard.addView(resultActions);

        LinearLayout historyCard = card();
        root.addView(historyCard, cardParams());
        historyCard.addView(sectionTitle("最近生成"));
        TextView historyHint = text("保留最近 5 个结果，并显示缩略图、时间与视频参数。", 12, false);
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

        backendSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                refreshBackendStatus();
                applyUiState();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // Keep the current backend selection.
            }
        });
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

    private void pickModelPack() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[] {
                "application/zip", "application/octet-stream", "application/x-zip-compressed"
        });
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivityForResult(intent, PICK_MODEL_PACK);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();

        if (requestCode == PICK_MODEL_PACK) {
            installModelPack(uri);
            return;
        }

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
            statusView.setText("主图已就绪。 ");
        } else if (requestCode == PICK_SECONDARY) {
            secondaryUri = uri;
            if (primaryUri != null) phase = UiStatePolicy.Phase.READY;
            secondaryLabel.setText("第二张：已选择 · 当前为 RIFE 双图插值模式");
            statusView.setText("双图插值模式已就绪。 ");
        }
        diagnosticsButton.setVisibility(View.GONE);
        applyUiState();
    }

    private void installModelPack(Uri uri) {
        if (modelInstalling) return;
        modelInstalling = true;
        progressBar.setProgress(0);
        statusView.setText("正在安装并校验 MobileI2V 模型包…");
        diagnosticsButton.setVisibility(View.GONE);
        applyUiState();
        worker.submit(() -> {
            try {
                InstalledModelPack installed = ModelPackInstaller.install(this, uri,
                        (percent, message) -> runOnUiThread(() -> {
                            progressBar.setProgress(percent);
                            statusView.setText(percent + "% · " + message);
                        }));
                if (!installed.manifest.isMobileI2V()) {
                    ModelPackInstaller.deleteRecursively(installed.root);
                    throw new IllegalArgumentException(
                            "模型包 backend=" + installed.manifest.backend + "，不是 MobileI2V");
                }
                modelPackStore.activate(installed);
                runOnUiThread(() -> {
                    mobilePack = installed;
                    modelInstalling = false;
                    progressBar.setProgress(100);
                    statusView.setText("MobileI2V 模型包校验并安装完成；runtime 状态见 Model Lab。 ");
                    refreshBackendStatus();
                    applyUiState();
                });
            } catch (Throwable error) {
                String diagnostics = "Local Video Lab V0.4 · model pack install\n"
                        + error.getClass().getName() + "\n" + safeMessage(error);
                runOnUiThread(() -> {
                    modelInstalling = false;
                    lastDiagnostics = diagnostics;
                    metricsView.setText(diagnostics);
                    diagnosticsButton.setVisibility(View.VISIBLE);
                    statusView.setText("模型包安装失败 · " + safeMessage(error));
                    refreshBackendStatus();
                    applyUiState();
                });
            }
        });
    }

    private void removeMobileModelPack() {
        if (modelInstalling || phase == UiStatePolicy.Phase.GENERATING) return;
        modelPackStore.removeMobileI2V();
        mobilePack = null;
        statusView.setText("MobileI2V 模型包已从 App 私有存储移除。 ");
        refreshBackendStatus();
        applyUiState();
    }

    private void startGeneration() {
        BackendRouter.Decision decision = currentBackendDecision();
        if (selectedBackend() == BackendRouter.Backend.MOBILE_I2V) {
            statusView.setText("MobileI2V 当前不可生成 · " + decision.message);
            refreshBackendStatus();
            applyUiState();
            return;
        }
        if (!decision.ready) {
            statusView.setText("RIFE 后端不可用 · " + decision.message);
            return;
        }
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
        final MotionSpec.Preset motionPreset = selectedMotionPreset();

        phase = UiStatePolicy.Phase.GENERATING;
        progressBar.setProgress(0);
        statusView.setText("0% · 正在准备 RIFE 本地运行时");
        diagnosticsButton.setVisibility(View.GONE);
        applyUiState();
        final int thermalBefore = thermalStatus();

        worker.submit(() -> {
            try {
                RifeEngine engine = new RifeEngine(this);
                RifeEngine.Result result = engine.generate(
                        primaryUri,
                        secondaryUri,
                        frames,
                        fps,
                        motionPreset,
                        (percent, message) -> runOnUiThread(() -> {
                            progressBar.setProgress(percent);
                            statusView.setText(percent + "% · " + message);
                        }));

                int thermalAfter = thermalStatus();
                long durationMs = Math.max(1L, result.frames * 1000L / Math.max(1, result.fps));
                ResultRecord record = new ResultRecord(
                        result.uri.toString(),
                        System.currentTimeMillis(),
                        durationMs,
                        result.width,
                        result.height,
                        result.frames,
                        result.fps);
                lastRecord = record;
                lastVideoUri = result.uri;
                historyStore.record(record);
                lastDiagnostics = formatMetrics(result, thermalBefore, thermalAfter);
                runOnUiThread(() -> {
                    phase = UiStatePolicy.Phase.SUCCESS;
                    progressBar.setProgress(100);
                    statusView.setText("100% · 已保存；点击结果缩略图可全屏播放");
                    metricsView.setText(lastDiagnostics);
                    showResult(record, true);
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

    private BackendRouter.Backend selectedBackend() {
        return backendSpinner != null && backendSpinner.getSelectedItemPosition() == 1
                ? BackendRouter.Backend.MOBILE_I2V
                : BackendRouter.Backend.RIFE_MOTION;
    }

    private BackendRouter.Decision currentBackendDecision() {
        boolean mobileRuntimeReady = onnxStatus != null
                && onnxStatus.jniLoaded
                && onnxStatus.mobileI2vExecutionImplemented;
        return BackendRouter.resolve(
                selectedBackend(),
                true,
                mobilePack != null,
                mobileRuntimeReady,
                capabilities == null ? 0 : capabilities.totalRamMb);
    }

    private void refreshBackendStatus() {
        if (backendStatusView == null) return;
        BackendRouter.Decision decision = currentBackendDecision();
        backendStatusView.setText(decision.ready
                ? "● READY · " + decision.message
                : "● NOT READY · " + decision.message);
        backendStatusView.setTextColor(decision.ready ? COLOR_ACCENT_DARK : COLOR_WARNING);

        if (mobilePack == null) {
            modelPackView.setText(
                    "MobileI2V 模型包：未安装\n"
                            + "官方基线：hustvl/MobileI2V · 0.27B · 17帧 · Apache-2.0 源码\n"
                            + "权重/导出产物采用外部模型包，不把约 1GB checkpoint 塞进 APK。 ");
        } else {
            ModelPackManifest m = mobilePack.manifest;
            modelPackView.setText(String.format(Locale.US,
                    "MobileI2V 模型包：%s · %s\n"
                            + "已验证产物：%d 个 · %.1f MB\n"
                            + "源码：%s @ %s\n"
                            + "许可：code=%s · weights=%s",
                    m.id, m.version, m.files.size(), mobilePack.bytes / 1048576.0,
                    emptyAsUnknown(m.sourceRepo), shortCommit(m.sourceCommit),
                    emptyAsUnknown(m.codeLicense), emptyAsUnknown(m.weightsLicense)));
        }
        removeModelButton.setVisibility(mobilePack == null ? View.GONE : View.VISIBLE);
        importModelButton.setText(mobilePack == null
                ? "导入 MobileI2V 模型包 (.mlvpkg)"
                : "更换 MobileI2V 模型包 (.mlvpkg)");

        String runtime = onnxStatus == null ? "ONNX Runtime: 未探测" : onnxStatus.message;
        deviceView.setText((capabilities == null ? "设备能力：未探测" : capabilities.summary())
                + "\n" + runtime);
    }

    private void showResult(ResultRecord record, boolean newlyGenerated) {
        if (record == null) return;
        lastRecord = record;
        lastVideoUri = Uri.parse(record.uri);
        resultCard.setVisibility(View.VISIBLE);
        resultTitle.setText(newlyGenerated ? "刚刚生成" : "最近结果");
        resultMeta.setText(resultMetadata(record));
        resultThumbnail.setImageDrawable(null);
        resultThumbnail.setBackgroundColor(0xffdfe4e3);
        resultPlayOverlay.setText("读取视频缩略图…");
        loadThumbnailAsync(record, resultThumbnail, true);
    }

    private void loadThumbnailAsync(ResultRecord record, ImageView target, boolean resultPreview) {
        if (record == null || target == null) return;
        String tag = record.uri;
        target.setTag(tag);
        worker.submit(() -> {
            Bitmap bitmap = null;
            try {
                bitmap = VideoThumbnailLoader.load(
                        this,
                        Uri.parse(record.uri),
                        resultPreview ? 720 : 240,
                        resultPreview ? 720 : 180);
                Bitmap ready = bitmap;
                runOnUiThread(() -> {
                    if (!tag.equals(target.getTag())) {
                        if (!ready.isRecycled()) ready.recycle();
                        return;
                    }
                    target.setImageBitmap(ready);
                    if (resultPreview) resultPlayOverlay.setText("▶  全屏播放");
                });
            } catch (Throwable error) {
                if (bitmap != null && !bitmap.isRecycled()) bitmap.recycle();
                runOnUiThread(() -> {
                    if (tag.equals(target.getTag()) && resultPreview) {
                        resultPlayOverlay.setText("▶  缩略图不可用 · 点击播放");
                    }
                });
            }
        });
    }

    private void openLastVideo() {
        openVideo(lastVideoUri);
    }

    private void openVideo(Uri uri) {
        if (uri == null) return;
        try {
            Intent intent = new Intent(Intent.ACTION_VIEW);
            intent.setDataAndType(uri, "video/mp4");
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            intent.setClipData(ClipData.newRawUri("generated-video", uri));
            startActivity(Intent.createChooser(intent, "全屏播放视频"));
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
        intent.putExtra(Intent.EXTRA_SUBJECT, "Local Video Lab V0.4 diagnostics");
        intent.putExtra(Intent.EXTRA_TEXT, lastDiagnostics);
        startActivity(Intent.createChooser(intent, "导出诊断信息"));
    }

    private void refreshHistory() {
        historyContainer.removeAllViews();
        List<ResultRecord> recent = historyStore.loadRecords();
        if (recent.isEmpty()) {
            TextView empty = text("暂无历史结果", 13, false);
            empty.setTextColor(COLOR_MUTED);
            historyContainer.addView(empty);
            return;
        }

        for (int i = 0; i < recent.size(); i++) {
            ResultRecord record = recent.get(i);
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.setPadding(0, dp(8), 0, dp(8));
            row.setClickable(true);
            row.setOnClickListener(v -> {
                lastRecord = record;
                lastVideoUri = Uri.parse(record.uri);
                phase = UiStatePolicy.Phase.SUCCESS;
                showResult(record, false);
                statusView.setText("已载入历史结果。点击缩略图或“全屏播放”。");
                applyUiState();
            });

            ImageView thumbnail = new ImageView(this);
            thumbnail.setBackgroundColor(0xffdfe4e3);
            thumbnail.setScaleType(ImageView.ScaleType.CENTER_CROP);
            row.addView(thumbnail, new LinearLayout.LayoutParams(dp(108), dp(78)));
            loadThumbnailAsync(record, thumbnail, false);

            TextView meta = text(historyMetadata(record, i), 13, false);
            meta.setTextColor(0xff37474f);
            meta.setPadding(dp(12), 0, 0, 0);
            row.addView(meta, new LinearLayout.LayoutParams(
                    0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
            historyContainer.addView(row);
        }
    }

    private void applyUiState() {
        if (generateButton == null) return;
        UiStatePolicy.State state = UiStatePolicy.resolve(
                primaryUri != null,
                secondaryUri != null,
                phase,
                lastVideoUri != null);
        BackendRouter.Backend backend = selectedBackend();
        BackendRouter.Decision decision = currentBackendDecision();
        boolean busy = phase == UiStatePolicy.Phase.GENERATING || modelInstalling;
        boolean rife = backend == BackendRouter.Backend.RIFE_MOTION;

        primaryButton.setText(state.primaryLabel);
        secondaryButton.setText(state.secondaryLabel);
        if (rife) {
            generateButton.setText(state.generateLabel);
            modeLabel.setText(secondaryUri == null
                    ? "RIFE · 单图运动"
                    : "RIFE · 双图神经插值");
        } else {
            generateButton.setText(decision.ready
                    ? "开始 MobileI2V 语义生成"
                    : "MobileI2V 尚未就绪");
            modeLabel.setText("MobileI2V · 单图语义 I2V");
        }

        generateButton.setEnabled(state.generateEnabled && decision.ready && !busy);
        primaryButton.setEnabled(!busy);
        secondaryButton.setEnabled(!busy);
        secondaryButton.setVisibility(rife ? View.VISIBLE : View.GONE);
        secondaryLabel.setVisibility(rife ? View.VISIBLE : View.GONE);
        clearSecondaryButton.setVisibility(
                rife && state.clearSecondaryVisible ? View.VISIBLE : View.GONE);
        framesSpinner.setEnabled(!busy);
        fpsSpinner.setEnabled(!busy);
        motionSpinner.setEnabled(!busy && rife && secondaryUri == null);
        motionSpinner.setVisibility(rife ? View.VISIBLE : View.GONE);
        motionHint.setVisibility(rife ? View.VISIBLE : View.GONE);
        backendSpinner.setEnabled(!busy);
        importModelButton.setEnabled(!busy);
        removeModelButton.setEnabled(!busy);
        openButton.setEnabled(state.openEnabled);
        shareButton.setEnabled(state.shareEnabled);
        resultCard.setVisibility(lastVideoUri == null ? View.GONE : View.VISIBLE);
    }

    private MotionSpec.Preset selectedMotionPreset() {
        switch (motionSpinner.getSelectedItemPosition()) {
            case 1: return MotionSpec.Preset.PUSH_IN;
            case 2: return MotionSpec.Preset.PAN_LEFT;
            case 3: return MotionSpec.Preset.DRIFT_UP;
            default: return MotionSpec.Preset.CINEMATIC_AUTO;
        }
    }

    private static String motionPresetName(MotionSpec.Preset preset) {
        if (preset == null) return "电影自动";
        switch (preset) {
            case PUSH_IN: return "推近";
            case PAN_LEFT: return "向左平移";
            case DRIFT_UP: return "向上漂移";
            case CINEMATIC_AUTO:
            default: return "电影自动";
        }
    }

    private String resultMetadata(ResultRecord record) {
        String time = record.createdAtMs > 0
                ? new SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()).format(new Date(record.createdAtMs))
                : "V0.2 历史结果";
        String duration = record.durationMs > 0
                ? String.format(Locale.US, "%.1f 秒", record.durationMs / 1000.0)
                : "时长未知";
        if (record.width > 0 && record.height > 0) {
            return String.format(Locale.US, "%s · %s\n%d×%d · %d 帧 · %d FPS",
                    time, duration, record.width, record.height, record.frames, record.fps);
        }
        return time + " · " + duration;
    }

    private String historyMetadata(ResultRecord record, int index) {
        String prefix = index == 0 ? "最近一次" : "历史 " + (index + 1);
        String time = record.createdAtMs > 0
                ? new SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()).format(new Date(record.createdAtMs))
                : "旧版记录";
        String duration = record.durationMs > 0
                ? String.format(Locale.US, "%.1fs", record.durationMs / 1000.0)
                : "--";
        if (record.width > 0) {
            return String.format(Locale.US, "%s · %s\n%s · %d×%d · %d帧/%dFPS",
                    prefix, time, duration, record.width, record.height, record.frames, record.fps);
        }
        return prefix + " · " + time + "\n点击载入并全屏播放";
    }

    private String formatMetrics(RifeEngine.Result result, int thermalBefore, int thermalAfter) {
        String mode = result.singleImageMode
                ? "单图运动 · " + motionPresetName(result.motionPreset)
                : "双图插值";
        return String.format(Locale.US,
                "Local Video Lab V0.4\n"
                        + "后端: RIFE v4.6 / ncnn / Vulkan\n"
                        + "模式: %s\n"
                        + "输出: %dx%d · %d 帧 · %d FPS\n"
                        + "耗时: %.2f s\n"
                        + "热状态: %s → %s\n"
                        + "Android API: %d\n"
                        + "Java max heap: %.0f MB · native heap: %.0f MB",
                mode,
                result.width, result.height, result.frames, result.fps,
                result.elapsedMs / 1000.0,
                thermalName(thermalBefore), thermalName(thermalAfter),
                Build.VERSION.SDK_INT,
                Runtime.getRuntime().maxMemory() / 1048576.0,
                Debug.getNativeHeapAllocatedSize() / 1048576.0);
    }

    private String formatError(Throwable error) {
        return String.format(Locale.US,
                "Local Video Lab V0.4\n"
                        + "状态: 生成失败\n"
                        + "后端: %s\n"
                        + "异常: %s\n"
                        + "信息: %s\n"
                        + "模式: %s\n"
                        + "Android API: %d\n"
                        + "热状态: %s\n"
                        + "Java max heap: %.0f MB\n"
                        + "native heap: %.0f MB",
                selectedBackend(),
                error.getClass().getName(),
                safeMessage(error),
                secondaryUri == null ? "单图运动 · " + motionPresetName(selectedMotionPreset()) : "双图插值",
                Build.VERSION.SDK_INT,
                thermalName(thermalStatus()),
                Runtime.getRuntime().maxMemory() / 1048576.0,
                Debug.getNativeHeapAllocatedSize() / 1048576.0);
    }

    private void updateMetrics(String state) {
        String backend = selectedBackend() == BackendRouter.Backend.RIFE_MOTION ? "RIFE" : "MobileI2V";
        metricsView.setText(String.format(Locale.US,
                "%s · backend=%s\n设备 API: %d · 热状态: %s\n"
                        + "Java max heap: %.0f MB · native heap: %.0f MB\n%s",
                state, backend, Build.VERSION.SDK_INT, thermalName(thermalStatus()),
                Runtime.getRuntime().maxMemory() / 1048576.0,
                Debug.getNativeHeapAllocatedSize() / 1048576.0,
                capabilities == null ? "" : capabilities.summary()));
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

    private static String emptyAsUnknown(String value) {
        return value == null || value.trim().isEmpty() ? "unknown" : value.trim();
    }

    private static String shortCommit(String value) {
        String cleaned = emptyAsUnknown(value);
        return cleaned.length() <= 12 ? cleaned : cleaned.substring(0, 12);
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
        worker.shutdownNow();
        super.onDestroy();
    }
}
