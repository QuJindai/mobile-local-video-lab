package com.qujindai.facere;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import java.io.File;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class FaceReMainActivity extends Activity {
    private static final int PICK_MODEL = 100;
    private static final int PICK_SOURCE = 101;
    private static final int PICK_TARGET = 102;
    private static final String PREFS = "face-re";
    private static final String PREF_MODEL_LICENSE_ACCEPTED = "insightface-model-license-v1";

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private Uri sourceUri;
    private Uri targetUri;
    private Uri resultUri;
    private Bitmap resultBitmap;
    private boolean busy;
    private boolean modelDownloading;
    private boolean modelDownloadPaused;
    private volatile boolean clearDownloadOnCancel;
    private FaceModelDownloader.CancelToken modelDownloadToken;

    private TextView modelStatus;
    private TextView status;
    private ImageView sourcePreview;
    private ImageView targetPreview;
    private ImageView resultPreview;
    private Button downloadButton;
    private Button cancelDownloadButton;
    private Button importButton;
    private Button sourceButton;
    private Button targetButton;
    private Button swapButton;
    private Button shareButton;
    private Spinner sourceSpinner;
    private ProgressBar progress;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setTitle("Face-RE");
        setContentView(buildUi());
        refreshModelStatus();
        refreshActions();
    }

    private View buildUi() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(18), dp(20), dp(18), dp(40));
        scroll.addView(root, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView title = text("Face-RE", 30, true);
        root.addView(title);
        TextView subtitle = text("S24U 本地静态换脸 · ONNX Runtime\n模型可联网下载；图片与人脸特征不上传", 14, false);
        subtitle.setTextColor(0xff5f6368);
        root.addView(subtitle, marginTop(4));

        root.addView(section("1  模型"), marginTop(22));
        modelStatus = text("", 14, false);
        root.addView(modelStatus, marginTop(8));

        TextView sourceTitle = text("下载源", 13, false);
        sourceTitle.setTextColor(0xff5f6368);
        root.addView(sourceTitle, marginTop(10));
        sourceSpinner = new Spinner(this);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item,
                new String[]{"自动（国内镜像优先）", "国内镜像", "InsightFace 官方源"});
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        sourceSpinner.setAdapter(adapter);
        sourceSpinner.setSelection(0);
        root.addView(sourceSpinner, marginTop(4));

        downloadButton = action("一键下载模型");
        downloadButton.setOnClickListener(v -> startOrPauseModelDownload());
        root.addView(downloadButton, marginTop(8));

        cancelDownloadButton = action("取消并清理下载缓存");
        cancelDownloadButton.setOnClickListener(v -> cancelAndClearModelDownload());
        cancelDownloadButton.setVisibility(View.GONE);
        root.addView(cancelDownloadButton, marginTop(6));

        importButton = action("导入模型包 ZIP");
        importButton.setOnClickListener(v -> pickModel());
        root.addView(importButton, marginTop(6));

        TextView license = text("自动下载使用 InsightFace 官方发布或国内镜像传输，并按固定 SHA-256 验真。模型权重受独立许可约束。", 12, false);
        license.setTextColor(0xff5f6368);
        root.addView(license, marginTop(6));

        root.addView(section("2  源人脸"), marginTop(22));
        sourceButton = action("选择源人脸图片");
        sourceButton.setOnClickListener(v -> pickImage(PICK_SOURCE));
        root.addView(sourceButton, marginTop(8));
        sourcePreview = preview();
        root.addView(sourcePreview, previewParams());

        root.addView(section("3  目标图片"), marginTop(22));
        targetButton = action("选择目标图片");
        targetButton.setOnClickListener(v -> pickImage(PICK_TARGET));
        root.addView(targetButton, marginTop(8));
        targetPreview = preview();
        root.addView(targetPreview, previewParams());

        swapButton = action("开始换脸");
        swapButton.setTextSize(17);
        swapButton.setOnClickListener(v -> startSwap());
        root.addView(swapButton, marginTop(24));

        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progress.setMax(100);
        progress.setProgress(0);
        root.addView(progress, marginTop(10));
        status = text("待机", 14, false);
        status.setTextColor(0xff5f6368);
        root.addView(status, marginTop(8));

        root.addView(section("4  结果"), marginTop(22));
        resultPreview = preview();
        resultPreview.setVisibility(View.GONE);
        root.addView(resultPreview, previewParams());
        shareButton = action("分享结果");
        shareButton.setOnClickListener(v -> shareResult());
        root.addView(shareButton, marginTop(10));
        return scroll;
    }

    private void startOrPauseModelDownload() {
        if (modelDownloading) {
            if (modelDownloadToken != null) modelDownloadToken.cancel();
            downloadButton.setEnabled(false);
            status.setText("正在暂停下载…已完成的分片会保留");
            return;
        }
        if (busy) return;
        SharedPreferences preferences = getSharedPreferences(PREFS, MODE_PRIVATE);
        if (!preferences.getBoolean(PREF_MODEL_LICENSE_ACCEPTED, false)) {
            new AlertDialog.Builder(this)
                    .setTitle("模型许可与隐私")
                    .setMessage("将按你的操作从 InsightFace 官方发布或国内镜像下载预训练模型。InsightFace 说明其预训练模型默认仅限非商业研究用途，商业使用需另行取得相应许可。国内镜像仅作传输加速，文件会按固定 SHA-256 校验。Face-RE 不上传源图片、目标图片或人脸 embedding。")
                    .setNegativeButton("取消", null)
                    .setPositiveButton("同意并下载", (dialog, which) -> {
                        preferences.edit().putBoolean(PREF_MODEL_LICENSE_ACCEPTED, true).apply();
                        beginModelDownload();
                    })
                    .show();
            return;
        }
        beginModelDownload();
    }

    private void beginModelDownload() {
        if (busy || modelDownloading) return;
        final FaceModelDownloadCatalog.SourceMode mode = selectedSourceMode();
        modelDownloadToken = new FaceModelDownloader.CancelToken();
        clearDownloadOnCancel = false;
        modelDownloading = true;
        modelDownloadPaused = false;
        progress.setProgress(0);
        setBusy(true, "准备下载模型…");
        refreshActions();
        worker.submit(() -> {
            try {
                FaceModelFiles files = FaceModelDownloader.downloadAndInstall(this, mode, modelDownloadToken,
                        (percent, message, downloaded, total, bps, source) -> runOnUiThread(() -> {
                            progress.setProgress(Math.max(0, Math.min(100, percent)));
                            String speed = bps > 0L
                                    ? String.format(Locale.US, " · %.1f MB/s", bps / 1048576.0)
                                    : "";
                            status.setText(percent + "% · " + message + speed);
                        }));
                runOnUiThread(() -> {
                    modelDownloading = false;
                    modelDownloadPaused = false;
                    modelDownloadToken = null;
                    progress.setProgress(100);
                    setBusy(false, String.format(Locale.US,
                            "模型下载并安装完成 · %.1f MB · READY", files.totalBytes() / 1048576.0));
                    refreshModelStatus();
                });
            } catch (FaceModelDownloader.CancelledException paused) {
                boolean clear = clearDownloadOnCancel;
                if (clear) clearDownloadCache();
                runOnUiThread(() -> {
                    modelDownloading = false;
                    modelDownloadPaused = !clear;
                    modelDownloadToken = null;
                    clearDownloadOnCancel = false;
                    setBusy(false, clear ? "下载已取消，缓存已清理" : "下载已暂停 · 点击继续可断点续传");
                    refreshActions();
                });
            } catch (Throwable error) {
                runOnUiThread(() -> {
                    modelDownloading = false;
                    modelDownloadPaused = true;
                    modelDownloadToken = null;
                    setBusy(false, "模型下载失败 · " + message(error) + " · 可点击继续重试");
                    refreshActions();
                });
            }
        });
    }

    private void cancelAndClearModelDownload() {
        if (modelDownloading) {
            clearDownloadOnCancel = true;
            if (modelDownloadToken != null) modelDownloadToken.cancel();
            cancelDownloadButton.setEnabled(false);
            downloadButton.setEnabled(false);
            status.setText("正在取消并清理下载缓存…");
            return;
        }
        if (!modelDownloadPaused || busy) return;
        setBusy(true, "正在清理下载缓存…");
        worker.submit(() -> {
            clearDownloadCache();
            runOnUiThread(() -> {
                modelDownloadPaused = false;
                progress.setProgress(0);
                setBusy(false, "下载缓存已清理");
                refreshActions();
            });
        });
    }

    private void clearDownloadCache() {
        FaceModelStore.deleteRecursively(new File(getFilesDir(), "face-re-downloads"));
    }

    private FaceModelDownloadCatalog.SourceMode selectedSourceMode() {
        int position = sourceSpinner == null ? 0 : sourceSpinner.getSelectedItemPosition();
        if (position == 1) return FaceModelDownloadCatalog.SourceMode.CHINA_MIRROR;
        if (position == 2) return FaceModelDownloadCatalog.SourceMode.OFFICIAL;
        return FaceModelDownloadCatalog.SourceMode.AUTO;
    }

    private void pickModel() {
        if (busy) return;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{
                "application/zip", "application/octet-stream", "application/x-zip-compressed"});
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivityForResult(intent, PICK_MODEL);
    }

    private void pickImage(int requestCode) {
        if (busy) return;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        startActivityForResult(intent, requestCode);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        if (requestCode == PICK_MODEL) {
            importModel(uri);
            return;
        }
        try {
            int flags = data.getFlags() & Intent.FLAG_GRANT_READ_URI_PERMISSION;
            if (flags != 0) getContentResolver().takePersistableUriPermission(uri, flags);
        } catch (SecurityException ignored) {
        }
        if (requestCode == PICK_SOURCE) {
            sourceUri = uri;
            loadPreview(sourcePreview, uri);
            status.setText("源人脸已选择");
        } else if (requestCode == PICK_TARGET) {
            targetUri = uri;
            loadPreview(targetPreview, uri);
            status.setText("目标图片已选择");
        }
        refreshActions();
    }

    private void importModel(Uri uri) {
        if (busy) return;
        setBusy(true, "正在校验并导入模型包…");
        worker.submit(() -> {
            try {
                FaceModelFiles files = FaceModelStore.importZip(this, uri,
                        message -> runOnUiThread(() -> status.setText(message)));
                runOnUiThread(() -> {
                    setBusy(false, String.format(Locale.US,
                            "模型就绪 · %.1f MB", files.totalBytes() / 1048576.0));
                    refreshModelStatus();
                });
            } catch (Throwable error) {
                runOnUiThread(() -> setBusy(false, "模型导入失败 · " + message(error)));
            }
        });
    }

    private void startSwap() {
        if (busy || sourceUri == null || targetUri == null || FaceModelStore.active(this) == null) return;
        setBusy(true, "开始本地换脸");
        progress.setProgress(0);
        worker.submit(() -> {
            FaceSwapEngine.Result result = null;
            try {
                result = new FaceSwapEngine(this).swap(sourceUri, targetUri,
                        (percent, message) -> runOnUiThread(() -> {
                            progress.setProgress(percent);
                            status.setText(percent + "% · " + message);
                        }));
                runOnUiThread(() -> status.setText("96% · 保存到 Pictures/FaceRE"));
                Uri published = ImagePublisher.publishJpeg(this, result.bitmap);
                FaceSwapEngine.Result finalResult = result;
                runOnUiThread(() -> {
                    if (resultBitmap != null && resultBitmap != finalResult.bitmap && !resultBitmap.isRecycled()) {
                        resultBitmap.recycle();
                    }
                    resultBitmap = finalResult.bitmap;
                    resultUri = published;
                    resultPreview.setImageBitmap(resultBitmap);
                    resultPreview.setVisibility(View.VISIBLE);
                    progress.setProgress(100);
                    setBusy(false, String.format(Locale.US,
                            "完成 · 源脸 %d · 目标脸 %d · %.2f s · 已保存到 Pictures/FaceRE",
                            finalResult.sourceFaceCount, finalResult.targetFaceCount,
                            finalResult.elapsedMs / 1000.0));
                });
            } catch (Throwable error) {
                if (result != null && result.bitmap != null && !result.bitmap.isRecycled()) result.bitmap.recycle();
                runOnUiThread(() -> setBusy(false, "换脸失败 · " + message(error)));
            }
        });
    }

    private void loadPreview(ImageView view, Uri uri) {
        worker.submit(() -> {
            try {
                Bitmap bitmap = ImageLoader.load(getContentResolver(), uri, 1200);
                runOnUiThread(() -> view.setImageBitmap(bitmap));
            } catch (Throwable ignored) {
                runOnUiThread(() -> view.setImageDrawable(null));
            }
        });
    }

    private void shareResult() {
        if (resultUri == null) return;
        Intent share = new Intent(Intent.ACTION_SEND);
        share.setType("image/jpeg");
        share.putExtra(Intent.EXTRA_STREAM, resultUri);
        share.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivity(Intent.createChooser(share, "分享换脸结果"));
    }

    private void refreshModelStatus() {
        FaceModelFiles files = FaceModelStore.active(this);
        if (files == null) {
            modelStatus.setText("NOT READY · 可一键下载（国内镜像优先）或导入完整模型 ZIP");
            modelStatus.setTextColor(0xffa33a00);
        } else {
            modelStatus.setText(String.format(Locale.US,
                    "READY · 本地模型 %.1f MB · 推理全本地，图片/embedding 不上传",
                    files.totalBytes() / 1048576.0));
            modelStatus.setTextColor(0xff137333);
        }
        refreshActions();
    }

    private void setBusy(boolean value, String message) {
        busy = value;
        status.setText(message);
        refreshActions();
    }

    private void refreshActions() {
        if (swapButton == null || shareButton == null || downloadButton == null) return;
        FaceModelFiles files = FaceModelStore.active(this);
        swapButton.setEnabled(!busy && sourceUri != null && targetUri != null && files != null);
        shareButton.setEnabled(!busy && resultUri != null);
        importButton.setEnabled(!busy);
        sourceButton.setEnabled(!busy);
        targetButton.setEnabled(!busy);
        sourceSpinner.setEnabled(!busy);

        if (modelDownloading) {
            downloadButton.setText("暂停下载");
            downloadButton.setEnabled(true);
            cancelDownloadButton.setVisibility(View.VISIBLE);
            cancelDownloadButton.setEnabled(true);
        } else if (modelDownloadPaused) {
            downloadButton.setText("继续下载模型");
            downloadButton.setEnabled(!busy);
            cancelDownloadButton.setVisibility(View.VISIBLE);
            cancelDownloadButton.setEnabled(!busy);
        } else {
            downloadButton.setText(files == null ? "一键下载模型" : "重新下载模型");
            downloadButton.setEnabled(!busy);
            cancelDownloadButton.setVisibility(View.GONE);
        }
    }

    private TextView section(String value) {
        TextView view = text(value, 18, true);
        view.setTextColor(0xff202124);
        return view;
    }

    private TextView text(String value, int sp, boolean bold) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(sp);
        view.setTextColor(Color.BLACK);
        if (bold) view.setTypeface(view.getTypeface(), android.graphics.Typeface.BOLD);
        return view;
    }

    private Button action(String value) {
        Button button = new Button(this);
        button.setText(value);
        button.setAllCaps(false);
        return button;
    }

    private ImageView preview() {
        ImageView view = new ImageView(this);
        view.setScaleType(ImageView.ScaleType.CENTER_CROP);
        view.setBackgroundColor(0xffeeeeee);
        view.setContentDescription("Face-RE preview");
        return view;
    }

    private LinearLayout.LayoutParams previewParams() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(220));
        params.topMargin = dp(10);
        return params;
    }

    private LinearLayout.LayoutParams marginTop(int dp) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.topMargin = dp(dp);
        return params;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static String message(Throwable error) {
        String value = error.getMessage();
        return value == null || value.trim().isEmpty() ? error.getClass().getSimpleName() : value;
    }

    @Override
    protected void onDestroy() {
        if (modelDownloadToken != null) modelDownloadToken.cancel();
        worker.shutdownNow();
        if (resultBitmap != null && !resultBitmap.isRecycled()) resultBitmap.recycle();
        super.onDestroy();
    }
}
