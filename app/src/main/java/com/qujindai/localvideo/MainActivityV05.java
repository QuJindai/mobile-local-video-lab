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

/**
 * V0.7 handset workbench.
 *
 * RIFE is the validated baseline, Depth 3D is a genuine second local model path,
 * and MobileI2V remains blocked until its actual Android execution loop exists.
 */
public final class MainActivityV05 extends Activity {
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
    private volatile MobileI2VGpuNative.Probe mobileGpuProbe;
    private MobileI2VMicroscope lastMobileMicroscope;
    private boolean modelInstalling;
    private boolean checkpointDownloading;
    private MobileI2VCheckpointDownloader checkpointDownloader;
    private boolean detailsExpanded;

    private Spinner backendSpinner;
    private Spinner framesSpinner;
    private Spinner fpsSpinner;
    private Spinner rifeMotionSpinner;
    private Spinner depthMotionSpinner;
    private TextView backendStatusView;
    private TextView mobilePackView;
    private TextView checkpointDownloadView;
    private TextView deviceView;
    private TextView modeLabel;
    private TextView secondaryLabel;
    private TextView rifeMotionHint;
    private TextView depthMotionHint;
    private TextView statusView;
    private TextView metricsView;
    private TextView resultTitle;
    private TextView resultMeta;
    private TextView resultPlayOverlay;
    private ImageView primaryPreview;
    private ImageView resultThumbnail;
    private ProgressBar progressBar;
    private Button primaryButton;
    private Button secondaryButton;
    private Button clearSecondaryButton;
    private Button generateButton;
    private Button openButton;
    private Button shareButton;
    private Button importModelButton;
    private Button downloadOfficialButton;
    private Button downloadMirrorButton;
    private Button cancelCheckpointDownloadButton;
    private Button removeModelButton;
    private Button detailsButton;
    private Button diagnosticsButton;
    private LinearLayout resultCard;
    private LinearLayout historyContainer;

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
            statusView.setText("选择后端和主图开始。Depth 3D 已作为第二个真实本地模型后端接入。 ");
        }
        refreshHistory();
        refreshBackendStatus();
        updateMetrics("待机");
        applyUiState();
        if (mobilePack != null) scheduleMobileProbe(mobilePack);
    }

    private View buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(18), dp(16), dp(44));
        scroll.addView(root, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        LinearLayout titleRow = new LinearLayout(this);
        titleRow.setOrientation(LinearLayout.HORIZONTAL);
        titleRow.setGravity(Gravity.CENTER_VERTICAL);
        TextView title = text("Local Video Lab", 25, true);
        titleRow.addView(title, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        TextView badge = text("V0.7", 12, true);
        badge.setTextColor(Color.WHITE);
        badge.setGravity(Gravity.CENTER);
        badge.setBackground(rounded(COLOR_ACCENT, 20));
        badge.setPadding(dp(10), dp(5), dp(10), dp(5));
        titleRow.addView(badge);
        root.addView(titleRow);

        TextView subtitle = text(
                "端侧生成 · 模型支持双源下载\n"
                        + "RIFE · Depth 3D · MobileI2V Adreno GPU / MNN OpenCL",
                14, false);
        subtitle.setTextColor(COLOR_MUTED);
        root.addView(subtitle);

        buildModelLab(root);
        buildInputCard(root);
        buildParameterCard(root);

        generateButton = primaryAction("开始生成");
        generateButton.setOnClickListener(v -> startGeneration());
        root.addView(generateButton, fullWidthParams(dp(12)));

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

        buildResultCard(root);
        buildHistoryCard(root);

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

    private void buildModelLab(LinearLayout root) {
        LinearLayout card = card();
        root.addView(card, cardParams());
        card.addView(sectionTitle("0  模型后端 · Model Lab"));
        card.addView(label("生成后端"));
        backendSpinner = spinner(new String[] {
                "RIFE Motion · 稳定已验收",
                "Depth 3D Motion · Depth Anything V2 Q4 + RIFE",
                "MobileI2V 0.27B · Adreno GPU / MNN OpenCL"
        }, 1);
        card.addView(backendSpinner);

        backendStatusView = text("", 13, true);
        card.addView(backendStatusView);

        TextView depthDescription = text(
                "Depth 3D 会先用 Depth Anything V2 Q4 在手机本地估深，再按深度构造分层视差端点，"
                        + "最后由 RIFE 生成连续中间帧。它是真实第二模型链路，但不是扩散式语义 I2V。",
                12, false);
        depthDescription.setTextColor(COLOR_MUTED);
        card.addView(depthDescription);

        mobilePackView = text("", 12, false);
        mobilePackView.setTextColor(COLOR_MUTED);
        card.addView(mobilePackView);

        card.addView(label("MobileI2V 上游权重下载"));
        checkpointDownloadView = text("", 12, false);
        checkpointDownloadView.setTextColor(COLOR_MUTED);
        card.addView(checkpointDownloadView);

        LinearLayout downloadRow = new LinearLayout(this);
        downloadRow.setOrientation(LinearLayout.HORIZONTAL);
        downloadOfficialButton = secondaryAction("下载 · 官方原版");
        downloadMirrorButton = secondaryAction("下载 · 国内镜像");
        downloadOfficialButton.setOnClickListener(v ->
                startCheckpointDownload(MobileI2VDownloadSource.official()));
        downloadMirrorButton.setOnClickListener(v ->
                startCheckpointDownload(MobileI2VDownloadSource.chinaMirror()));
        downloadRow.addView(downloadOfficialButton, weightedButtonParams(1f, 0));
        downloadRow.addView(downloadMirrorButton, weightedButtonParams(1f, dp(8)));
        card.addView(downloadRow);

        cancelCheckpointDownloadButton = textAction("取消下载 · 保留断点");
        cancelCheckpointDownloadButton.setVisibility(View.GONE);
        cancelCheckpointDownloadButton.setOnClickListener(v -> cancelCheckpointDownload());
        card.addView(cancelCheckpointDownloadButton);

        TextView downloadNote = text(
                "两条路径锁定同一 upstream checkpoint 与 SHA-256。下载的是原始 hybrid_371.pth；"
                        + "它用于后续 Android 导出/转换，不会被冒充为可执行 .mlvpkg。",
                12, false);
        downloadNote.setTextColor(COLOR_WARNING);
        card.addView(downloadNote);

        importModelButton = secondaryAction("导入 MobileI2V 模型包 (.mlvpkg)");
        importModelButton.setOnClickListener(v -> pickModelPack());
        card.addView(importModelButton);

        removeModelButton = textAction("移除 MobileI2V 模型包");
        removeModelButton.setOnClickListener(v -> removeMobileModelPack());
        card.addView(removeModelButton);

        card.addView(label("设备 / Runtime 能力"));
        deviceView = text("", 12, false);
        deviceView.setTextColor(COLOR_MUTED);
        card.addView(deviceView);

        TextView honesty = text(
                "MobileI2V 仅在模型包、Android 执行 runtime 和设备门槛全部通过后才允许生成；"
                        + "当前不会用 RIFE 冒充 MobileI2V。",
                12, false);
        honesty.setTextColor(COLOR_WARNING);
        card.addView(honesty);

        backendSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                refreshBackendStatus();
                applyUiState();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });
    }

    private void buildInputCard(LinearLayout root) {
        LinearLayout card = card();
        root.addView(card, cardParams());
        card.addView(sectionTitle("1  输入"));

        modeLabel = text("Depth 3D · 单图", 13, true);
        modeLabel.setTextColor(COLOR_ACCENT_DARK);
        card.addView(modeLabel);

        primaryPreview = new ImageView(this);
        primaryPreview.setBackgroundColor(0xffe5e9e8);
        primaryPreview.setScaleType(ImageView.ScaleType.CENTER_CROP);
        LinearLayout.LayoutParams previewParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(220));
        previewParams.topMargin = dp(8);
        card.addView(primaryPreview, previewParams);

        primaryButton = secondaryAction("选择主图");
        primaryButton.setOnClickListener(v -> pickImage(PICK_PRIMARY));
        card.addView(primaryButton);

        secondaryButton = secondaryAction("添加第二张图（RIFE 双图）");
        secondaryButton.setOnClickListener(v -> pickImage(PICK_SECONDARY));
        card.addView(secondaryButton);

        secondaryLabel = text("第二张：未选择", 13, false);
        secondaryLabel.setTextColor(COLOR_MUTED);
        card.addView(secondaryLabel);

        clearSecondaryButton = textAction("移除第二张图");
        clearSecondaryButton.setOnClickListener(v -> {
            secondaryUri = null;
            secondaryLabel.setText("第二张：未选择");
            if (primaryUri != null) phase = UiStatePolicy.Phase.READY;
            applyUiState();
        });
        card.addView(clearSecondaryButton);
    }

    private void buildParameterCard(LinearLayout root) {
        LinearLayout card = card();
        root.addView(card, cardParams());
        card.addView(sectionTitle("2  生成参数"));

        framesSpinner = spinner(new String[] { "9 帧 · 快速", "17 帧 · 推荐" }, 1);
        fpsSpinner = spinner(new String[] { "6 FPS", "8 FPS · 推荐", "12 FPS" }, 1);
        rifeMotionSpinner = spinner(new String[] {
                "电影自动 · 推荐",
                "推近",
                "向左平移",
                "向上漂移"
        }, 0);
        depthMotionSpinner = spinner(new String[] {
                "3D 向左视差 · 推荐",
                "3D 向右视差",
                "3D 推近"
        }, 0);

        card.addView(label("帧数"));
        card.addView(framesSpinner);
        card.addView(label("播放速度"));
        card.addView(fpsSpinner);

        card.addView(label("RIFE 二维单图运动"));
        card.addView(rifeMotionSpinner);
        rifeMotionHint = text("仅 RIFE 单图模式生效。", 12, false);
        rifeMotionHint.setTextColor(COLOR_MUTED);
        card.addView(rifeMotionHint);

        card.addView(label("Depth 3D 运动"));
        card.addView(depthMotionSpinner);
        depthMotionHint = text("深度越近的区域位移/缩放越大，形成真实分层视差。", 12, false);
        depthMotionHint.setTextColor(COLOR_MUTED);
        card.addView(depthMotionHint);
    }

    private void buildResultCard(LinearLayout root) {
        resultCard = card();
        resultCard.setVisibility(View.GONE);
        root.addView(resultCard, cardParams());
        resultCard.addView(sectionTitle("3  生成结果"));
        resultTitle = text("最近结果", 14, true);
        resultCard.addView(resultTitle);

        FrameLayout frame = new FrameLayout(this);
        LinearLayout.LayoutParams frameParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(230));
        frameParams.topMargin = dp(8);
        resultCard.addView(frame, frameParams);

        resultThumbnail = new ImageView(this);
        resultThumbnail.setBackgroundColor(0xffdfe4e3);
        resultThumbnail.setScaleType(ImageView.ScaleType.CENTER_CROP);
        frame.addView(resultThumbnail, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        resultPlayOverlay = text("▶  全屏播放", 16, true);
        resultPlayOverlay.setTextColor(Color.WHITE);
        resultPlayOverlay.setGravity(Gravity.CENTER);
        resultPlayOverlay.setBackground(rounded(0xaa000000, 24));
        resultPlayOverlay.setPadding(dp(18), dp(10), dp(18), dp(10));
        FrameLayout.LayoutParams overlay = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        overlay.gravity = Gravity.CENTER;
        frame.addView(resultPlayOverlay, overlay);
        frame.setOnClickListener(v -> openLastVideo());

        resultMeta = text("", 13, false);
        resultMeta.setTextColor(COLOR_MUTED);
        resultCard.addView(resultMeta);

        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        actions.setPadding(0, dp(8), 0, 0);
        openButton = secondaryAction("全屏播放");
        shareButton = primaryAction("分享视频");
        openButton.setOnClickListener(v -> openLastVideo());
        shareButton.setOnClickListener(v -> shareLastVideo());
        actions.addView(openButton, weightedButtonParams(1f, 0));
        actions.addView(shareButton, weightedButtonParams(1f, dp(8)));
        resultCard.addView(actions);
    }

    private void buildHistoryCard(LinearLayout root) {
        LinearLayout card = card();
        root.addView(card, cardParams());
        card.addView(sectionTitle("最近生成"));
        TextView hint = text("保留最近 5 个结果，重启后仍可查看。", 12, false);
        hint.setTextColor(COLOR_MUTED);
        card.addView(hint);
        historyContainer = new LinearLayout(this);
        historyContainer.setOrientation(LinearLayout.VERTICAL);
        card.addView(historyContainer);
    }

    private void startCheckpointDownload(MobileI2VDownloadSource source) {
        if (checkpointDownloading || modelInstalling || phase == UiStatePolicy.Phase.GENERATING) return;
        checkpointDownloading = true;
        checkpointDownloader = new MobileI2VCheckpointDownloader();
        progressBar.setProgress(CheckpointDownloadPolicy.percent(
                MobileI2VCheckpointDownloader.partialBytes(this), source.expectedBytes));
        statusView.setText("准备下载 · " + source.label);
        refreshBackendStatus();
        applyUiState();

        worker.submit(() -> {
            try {
                MobileI2VCheckpointDownloader.Result result = checkpointDownloader.download(
                        this, source, (percent, downloaded, total, bytesPerSecond, message) ->
                                runOnUiThread(() -> {
                                    progressBar.setProgress(percent);
                                    statusView.setText(String.format(Locale.US,
                                            "%d%% · %s · %s / %s · %s/s",
                                            percent, message, formatDownloadBytes(downloaded),
                                            formatDownloadBytes(total), formatDownloadBytes(bytesPerSecond)));
                                    if (checkpointDownloadView != null) {
                                        checkpointDownloadView.setText(String.format(Locale.US,
                                                "正在下载：%s\n%s / %s · %s/s",
                                                source.label, formatDownloadBytes(downloaded),
                                                formatDownloadBytes(total), formatDownloadBytes(bytesPerSecond)));
                                    }
                                }));
                runOnUiThread(() -> {
                    checkpointDownloading = false;
                    progressBar.setProgress(100);
                    statusView.setText((result.fromCache ? "已存在" : "下载完成")
                            + " · SHA-256 PASS · " + result.source.label);
                    refreshBackendStatus();
                    applyUiState();
                });
            } catch (MobileI2VCheckpointDownloader.CancelledException cancelled) {
                runOnUiThread(() -> {
                    checkpointDownloading = false;
                    statusView.setText("下载已取消 · 已保留断点，可从任一路径继续");
                    refreshBackendStatus();
                    applyUiState();
                });
            } catch (Throwable error) {
                String diag = "Local Video Lab V0.7 · MobileI2V checkpoint download\n"
                        + source.label + "\n" + error.getClass().getName() + "\n" + safeMessage(error);
                runOnUiThread(() -> {
                    checkpointDownloading = false;
                    lastDiagnostics = diag;
                    metricsView.setText(diag);
                    diagnosticsButton.setVisibility(View.VISIBLE);
                    statusView.setText("下载失败 · " + source.label + " · " + safeMessage(error));
                    refreshBackendStatus();
                    applyUiState();
                });
            }
        });
    }

    private void cancelCheckpointDownload() {
        if (checkpointDownloader != null) checkpointDownloader.cancel();
    }

    private String formatDownloadBytes(long bytes) {
        if (bytes <= 0L) return "0 B";
        if (bytes >= 1073741824L) return String.format(Locale.US, "%.2f GB", bytes / 1073741824.0);
        if (bytes >= 1048576L) return String.format(Locale.US, "%.1f MB", bytes / 1048576.0);
        if (bytes >= 1024L) return String.format(Locale.US, "%.1f KB", bytes / 1024.0);
        return bytes + " B";
    }

    private void pickImage(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
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
        }

        if (requestCode == PICK_PRIMARY) {
            primaryUri = uri;
            primaryPreview.setImageURI(uri);
            phase = UiStatePolicy.Phase.READY;
            statusView.setText("主图已就绪。 ");
        } else if (requestCode == PICK_SECONDARY) {
            secondaryUri = uri;
            if (primaryUri != null) phase = UiStatePolicy.Phase.READY;
            secondaryLabel.setText("第二张：已选择 · RIFE 双图插值");
            statusView.setText("双图插值输入已就绪。 ");
        }
        diagnosticsButton.setVisibility(View.GONE);
        applyUiState();
    }

    private void installModelPack(Uri uri) {
        if (modelInstalling) return;
        modelInstalling = true;
        progressBar.setProgress(0);
        statusView.setText("正在安装并校验 MobileI2V 模型包…");
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
                    mobileGpuProbe = null;
                    scheduleMobileProbe(installed);
                    modelInstalling = false;
                    progressBar.setProgress(100);
                    statusView.setText("MobileI2V 模型包已完成 SHA-256 校验并安装。 ");
                    refreshBackendStatus();
                    applyUiState();
                });
            } catch (Throwable error) {
                String diag = "Local Video Lab V0.7 · model pack install\n"
                        + error.getClass().getName() + "\n" + safeMessage(error);
                runOnUiThread(() -> {
                    modelInstalling = false;
                    lastDiagnostics = diag;
                    metricsView.setText(diag);
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
        mobileGpuProbe = null;
        statusView.setText("MobileI2V 模型包已移除。 ");
        refreshBackendStatus();
        applyUiState();
    }

    private void startGeneration() {
        BackendRouter.Backend backend = selectedBackend();
        BackendRouter.Decision decision = currentBackendDecision();
        if (!decision.ready) {
            statusView.setText("当前后端不可生成 · " + decision.message);
            refreshBackendStatus();
            applyUiState();
            return;
        }
        if (primaryUri == null) {
            statusView.setText("请先选择主图。 ");
            return;
        }

        final int frames = backend == BackendRouter.Backend.MOBILE_I2V
                ? MobileI2VGpuNative.OUTPUT_FRAMES
                : (framesSpinner.getSelectedItemPosition() == 0 ? 9 : 17);
        final int fps;
        switch (fpsSpinner.getSelectedItemPosition()) {
            case 0: fps = 6; break;
            case 2: fps = 12; break;
            default: fps = 8; break;
        }
        final MotionSpec.Preset rifePreset = selectedRifePreset();
        final DepthMotionSpec.Preset depthPreset = selectedDepthPreset();

        phase = UiStatePolicy.Phase.GENERATING;
        progressBar.setProgress(0);
        statusView.setText("0% · 正在准备 " + backendDisplayName(backend));
        diagnosticsButton.setVisibility(View.GONE);
        applyUiState();
        final int thermalBefore = thermalStatus();

        worker.submit(() -> {
            try {
                if (backend == BackendRouter.Backend.MOBILE_I2V) {
                    MobileI2VGpuEngine mobileEngine = new MobileI2VGpuEngine(this);
                    MobileI2VGpuEngine.Result mobileResult = mobileEngine.generate(
                            primaryUri, mobilePack, fps,
                            (percent, message) -> publishProgress(percent, message));
                    long durationMs = Math.max(1L,
                            mobileResult.frames * 1000L / Math.max(1, mobileResult.fps));
                    ResultRecord record = new ResultRecord(
                            mobileResult.uri.toString(), System.currentTimeMillis(), durationMs,
                            mobileResult.width, mobileResult.height,
                            mobileResult.frames, mobileResult.fps);
                    lastRecord = record;
                    lastVideoUri = mobileResult.uri;
                    historyStore.record(record);
                    lastMobileMicroscope = mobileResult.microscope;
                    lastDiagnostics = mobileResult.microscope.format();
                    runOnUiThread(() -> {
                        phase = UiStatePolicy.Phase.SUCCESS;
                        progressBar.setProgress(100);
                        statusView.setText("100% · MobileI2V GPU 已保存到 Movies/LocalVideoLab");
                        metricsView.setText(lastDiagnostics);
                        showResult(record, true);
                        refreshHistory();
                        applyUiState();
                    });
                    return;
                }

                RifeEngine engine = new RifeEngine(this);
                RifeEngine.Result result;
                if (backend == BackendRouter.Backend.DEPTH_RIFE) {
                    result = engine.generateDepthMotion(
                            primaryUri, frames, fps, depthPreset,
                            (percent, message) -> publishProgress(percent, message));
                } else {
                    result = engine.generate(
                            primaryUri, secondaryUri, frames, fps, rifePreset,
                            (percent, message) -> publishProgress(percent, message));
                }

                int thermalAfter = thermalStatus();
                long durationMs = Math.max(1L, result.frames * 1000L / Math.max(1, result.fps));
                ResultRecord record = new ResultRecord(
                        result.uri.toString(), System.currentTimeMillis(), durationMs,
                        result.width, result.height, result.frames, result.fps);
                lastRecord = record;
                lastVideoUri = result.uri;
                historyStore.record(record);
                lastDiagnostics = formatMetrics(result, thermalBefore, thermalAfter);
                runOnUiThread(() -> {
                    phase = UiStatePolicy.Phase.SUCCESS;
                    progressBar.setProgress(100);
                    statusView.setText("100% · 已保存到 Movies/LocalVideoLab");
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

    private void publishProgress(int percent, String message) {
        runOnUiThread(() -> {
            progressBar.setProgress(percent);
            statusView.setText(percent + "% · " + message);
        });
    }

    private BackendRouter.Backend selectedBackend() {
        int position = backendSpinner == null ? 0 : backendSpinner.getSelectedItemPosition();
        if (position == 1) return BackendRouter.Backend.DEPTH_RIFE;
        if (position == 2) return BackendRouter.Backend.MOBILE_I2V;
        return BackendRouter.Backend.RIFE_MOTION;
    }

    private void scheduleMobileProbe(InstalledModelPack pack) {
        mobileGpuProbe = null;
        if (pack == null || !pack.isAcceleratedMobileI2V()) return;
        worker.submit(() -> {
            MobileI2VGpuNative.Probe probe = MobileI2VGpuNative.probe(pack);
            runOnUiThread(() -> {
                if (mobilePack == pack) {
                    mobileGpuProbe = probe;
                    refreshBackendStatus();
                    applyUiState();
                }
            });
        });
    }

    private BackendRouter.Decision currentBackendDecision() {
        BackendRouter.Backend backend = selectedBackend();
        if (backend == BackendRouter.Backend.MOBILE_I2V) {
            if (mobilePack == null || !mobilePack.isAcceleratedMobileI2V()) {
                return new BackendRouter.Decision(backend, false,
                        BackendRouter.Blocker.MODEL_PACK_MISSING,
                        "未安装可执行 MobileI2V GPU 模型包 (.mlvpkg)");
            }
            long ram = capabilities == null ? 0 : capabilities.totalRamMb;
            if (ram > 0 && ram < 8192) {
                return new BackendRouter.Decision(backend, false,
                        BackendRouter.Blocker.INSUFFICIENT_RAM,
                        "设备内存低于 MobileI2V 最低门槛 8 GB");
            }
            MobileI2VGpuNative.Probe probe = mobileGpuProbe;
            if (probe == null) {
                return new BackendRouter.Decision(backend, false,
                        BackendRouter.Blocker.RUNTIME_PENDING,
                        "MobileI2V GPU runtime 正在探测");
            }
            return new BackendRouter.Decision(backend, probe.openClReady,
                    probe.openClReady ? BackendRouter.Blocker.NONE : BackendRouter.Blocker.RUNTIME_PENDING,
                    probe.openClReady
                            ? "MobileI2V · Adreno GPU · MNN OpenCL 已就绪"
                            : probe.message);
        }
        boolean ortReady = onnxStatus != null && onnxStatus.jniLoaded;
        boolean depthReady = ortReady && DepthRuntimeBundle.isPackaged(this);
        return BackendRouter.resolve(backend, true, depthReady,
                false, false, capabilities == null ? 0 : capabilities.totalRamMb);
    }

    private void refreshBackendStatus() {
        if (backendStatusView == null) return;
        BackendRouter.Decision decision = currentBackendDecision();
        backendStatusView.setText((decision.ready ? "● READY · " : "● NOT READY · ")
                + decision.message);
        backendStatusView.setTextColor(decision.ready ? COLOR_ACCENT_DARK : COLOR_WARNING);

        if (mobilePack == null) {
            mobilePackView.setText(
                    "MobileI2V 模型包：未安装\n"
                            + "固定上游：hustvl/MobileI2V · 0.27B · 17帧\n"
                            + "权重/导出产物采用外部 .mlvpkg，不把约 1GB checkpoint 塞进 APK。 ");
        } else {
            ModelPackManifest m = mobilePack.manifest;
            mobilePackView.setText(String.format(Locale.US,
                    "MobileI2V 模型包：%s · %s\n"
                            + "已验证：%d 个产物 · %.1f MB\n"
                            + "源码：%s @ %s\n"
                            + "许可：code=%s · weights=%s",
                    m.id, m.version, m.files.size(), mobilePack.bytes / 1048576.0,
                    emptyAsUnknown(m.sourceRepo), shortCommit(m.sourceCommit),
                    emptyAsUnknown(m.codeLicense), emptyAsUnknown(m.weightsLicense)));
        }
        if (checkpointDownloadView != null) {
            long partial = MobileI2VCheckpointDownloader.partialBytes(this);
            MobileI2VDownloadSource checkpoint = MobileI2VDownloadSource.official();
            if (MobileI2VCheckpointDownloader.isVerified(this)) {
                checkpointDownloadView.setText(
                        "上游原版权重：已下载 · SHA-256 PASS · "
                                + formatDownloadBytes(checkpoint.expectedBytes)
                                + "\n可离线保留；原始 .pth 仍需 Android 导出 runtime 才能进入生成门槛。");
            } else if (partial > 0L) {
                checkpointDownloadView.setText(
                        "上游原版权重：断点已保存 · " + formatDownloadBytes(partial)
                                + " / " + formatDownloadBytes(checkpoint.expectedBytes)
                                + "\n官方原版与国内镜像可交叉续传，同一 SHA-256 校验。");
            } else {
                checkpointDownloadView.setText(
                        "上游原版权重：未下载 · " + formatDownloadBytes(checkpoint.expectedBytes)
                                + "\n官方：Hugging Face · 国内：HF-Mirror");
            }
        }
        removeModelButton.setVisibility(mobilePack == null ? View.GONE : View.VISIBLE);
        importModelButton.setText(mobilePack == null
                ? "导入 MobileI2V 模型包 (.mlvpkg)"
                : "更换 MobileI2V 模型包 (.mlvpkg)");

        String runtime = onnxStatus == null ? "ONNX Runtime: 未探测" : onnxStatus.message;
        deviceView.setText((capabilities == null ? "设备能力：未探测" : capabilities.summary())
                + "\nDepth Anything V2 Q4 asset: "
                + (DepthRuntimeBundle.isPackaged(this) ? "PACKAGED" : "MISSING")
                + "\n" + runtime
                + "\nMobileI2V GPU: "
                + (mobileGpuProbe == null ? "PENDING" : mobileGpuProbe.message));
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
        boolean busy = phase == UiStatePolicy.Phase.GENERATING || modelInstalling || checkpointDownloading;
        boolean rife = backend == BackendRouter.Backend.RIFE_MOTION;
        boolean depth = backend == BackendRouter.Backend.DEPTH_RIFE;
        boolean mobile = backend == BackendRouter.Backend.MOBILE_I2V;

        primaryButton.setText(primaryUri == null ? "选择主图" : "更换主图");
        secondaryButton.setText(secondaryUri == null ? "添加第二张图（RIFE 双图）" : "更换第二张图");

        if (rife) {
            modeLabel.setText(secondaryUri == null ? "RIFE · 单图运动" : "RIFE · 双图神经插值");
            generateButton.setText(phase == UiStatePolicy.Phase.GENERATING ? "RIFE 生成中…" : "开始 RIFE 本地生成");
        } else if (depth) {
            modeLabel.setText("Depth Anything V2 Q4 + RIFE · 深度 3D 运动");
            generateButton.setText(phase == UiStatePolicy.Phase.GENERATING ? "Depth 3D 生成中…" : "开始 Depth 3D 本地生成");
        } else {
            modeLabel.setText("MobileI2V · Adreno GPU / MNN OpenCL · 17帧");
            generateButton.setText(decision.ready ? "开始 MobileI2V GPU 生成" : "MobileI2V GPU 尚未就绪");
        }

        generateButton.setEnabled(primaryUri != null && decision.ready && !busy);
        primaryButton.setEnabled(!busy);
        backendSpinner.setEnabled(!busy);
        framesSpinner.setEnabled(!busy && !mobile);
        fpsSpinner.setEnabled(!busy);
        importModelButton.setEnabled(!busy);
        downloadOfficialButton.setEnabled(!busy);
        downloadMirrorButton.setEnabled(!busy);
        cancelCheckpointDownloadButton.setVisibility(checkpointDownloading ? View.VISIBLE : View.GONE);
        cancelCheckpointDownloadButton.setEnabled(checkpointDownloading);
        removeModelButton.setEnabled(!busy);

        secondaryButton.setVisibility(rife ? View.VISIBLE : View.GONE);
        secondaryLabel.setVisibility(rife ? View.VISIBLE : View.GONE);
        clearSecondaryButton.setVisibility(rife && secondaryUri != null ? View.VISIBLE : View.GONE);
        secondaryButton.setEnabled(!busy);

        rifeMotionSpinner.setVisibility(rife ? View.VISIBLE : View.GONE);
        rifeMotionHint.setVisibility(rife ? View.VISIBLE : View.GONE);
        rifeMotionSpinner.setEnabled(!busy && secondaryUri == null);

        depthMotionSpinner.setVisibility(depth ? View.VISIBLE : View.GONE);
        depthMotionHint.setVisibility(depth ? View.VISIBLE : View.GONE);
        depthMotionSpinner.setEnabled(!busy);

        openButton.setEnabled(lastVideoUri != null);
        shareButton.setEnabled(lastVideoUri != null);
        resultCard.setVisibility(lastVideoUri == null ? View.GONE : View.VISIBLE);
    }

    private MotionSpec.Preset selectedRifePreset() {
        switch (rifeMotionSpinner.getSelectedItemPosition()) {
            case 1: return MotionSpec.Preset.PUSH_IN;
            case 2: return MotionSpec.Preset.PAN_LEFT;
            case 3: return MotionSpec.Preset.DRIFT_UP;
            default: return MotionSpec.Preset.CINEMATIC_AUTO;
        }
    }

    private DepthMotionSpec.Preset selectedDepthPreset() {
        switch (depthMotionSpinner.getSelectedItemPosition()) {
            case 1: return DepthMotionSpec.Preset.PARALLAX_RIGHT;
            case 2: return DepthMotionSpec.Preset.DOLLY_IN;
            default: return DepthMotionSpec.Preset.PARALLAX_LEFT;
        }
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
                        this, Uri.parse(record.uri),
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
                statusView.setText("已载入历史结果。 ");
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
        intent.putExtra(Intent.EXTRA_SUBJECT, "Local Video Lab V0.7 diagnostics");
        intent.putExtra(Intent.EXTRA_TEXT, lastDiagnostics);
        startActivity(Intent.createChooser(intent, "导出诊断信息"));
    }

    private String formatMetrics(RifeEngine.Result result, int thermalBefore, int thermalAfter) {
        String mode;
        if (result.depthPreset != null) {
            mode = "Depth 3D · " + depthPresetName(result.depthPreset);
        } else if (result.singleImageMode) {
            mode = "RIFE 单图 · " + rifePresetName(result.motionPreset);
        } else {
            mode = "RIFE 双图插值";
        }
        String preprocessing = result.preprocessingMs > 0
                ? String.format(Locale.US, "\n估深预处理: %.2f s", result.preprocessingMs / 1000.0)
                : "";
        return String.format(Locale.US,
                "Local Video Lab V0.7\n"
                        + "后端: %s\n"
                        + "模式: %s\n"
                        + "输出: %dx%d · %d 帧 · %d FPS\n"
                        + "总耗时: %.2f s%s\n"
                        + "热状态: %s → %s\n"
                        + "Android API: %d\n"
                        + "Java max heap: %.0f MB · native heap: %.0f MB",
                result.backendLabel,
                mode,
                result.width, result.height, result.frames, result.fps,
                result.elapsedMs / 1000.0, preprocessing,
                thermalName(thermalBefore), thermalName(thermalAfter),
                Build.VERSION.SDK_INT,
                Runtime.getRuntime().maxMemory() / 1048576.0,
                Debug.getNativeHeapAllocatedSize() / 1048576.0);
    }

    private String formatError(Throwable error) {
        return String.format(Locale.US,
                "Local Video Lab V0.7\n"
                        + "状态: 生成失败\n"
                        + "后端: %s\n"
                        + "异常: %s\n"
                        + "信息: %s\n"
                        + "Android API: %d\n"
                        + "热状态: %s\n"
                        + "Java max heap: %.0f MB\n"
                        + "native heap: %.0f MB",
                selectedBackend(), error.getClass().getName(), safeMessage(error),
                Build.VERSION.SDK_INT, thermalName(thermalStatus()),
                Runtime.getRuntime().maxMemory() / 1048576.0,
                Debug.getNativeHeapAllocatedSize() / 1048576.0);
    }

    private void updateMetrics(String state) {
        metricsView.setText(String.format(Locale.US,
                "%s · backend=%s\n设备 API: %d · 热状态: %s\n"
                        + "Java max heap: %.0f MB · native heap: %.0f MB\n%s",
                state, selectedBackend(), Build.VERSION.SDK_INT, thermalName(thermalStatus()),
                Runtime.getRuntime().maxMemory() / 1048576.0,
                Debug.getNativeHeapAllocatedSize() / 1048576.0,
                capabilities == null ? "" : capabilities.summary()));
    }

    private String resultMetadata(ResultRecord record) {
        String time = record.createdAtMs > 0
                ? new SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()).format(new Date(record.createdAtMs))
                : "旧版结果";
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
        return prefix + " · " + time + "\n点击载入";
    }

    private static String backendDisplayName(BackendRouter.Backend backend) {
        if (backend == BackendRouter.Backend.DEPTH_RIFE) return "Depth 3D";
        if (backend == BackendRouter.Backend.MOBILE_I2V) return "MobileI2V";
        return "RIFE";
    }

    private static String rifePresetName(MotionSpec.Preset preset) {
        if (preset == null) return "电影自动";
        switch (preset) {
            case PUSH_IN: return "推近";
            case PAN_LEFT: return "向左平移";
            case DRIFT_UP: return "向上漂移";
            case CINEMATIC_AUTO:
            default: return "电影自动";
        }
    }

    private static String depthPresetName(DepthMotionSpec.Preset preset) {
        if (preset == null) return "3D 向左视差";
        switch (preset) {
            case PARALLAX_RIGHT: return "3D 向右视差";
            case DOLLY_IN: return "3D 推近";
            case PARALLAX_LEFT:
            default: return "3D 向左视差";
        }
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
        button.setLayoutParams(fullWidthParams(dp(8)));
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
