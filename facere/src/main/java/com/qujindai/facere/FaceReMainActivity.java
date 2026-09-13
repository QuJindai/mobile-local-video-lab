package com.qujindai.facere;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class FaceReMainActivity extends Activity {
    private static final int PICK_MODEL = 100;
    private static final int PICK_SOURCE = 101;
    private static final int PICK_TARGET = 102;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private Uri sourceUri;
    private Uri targetUri;
    private Uri resultUri;
    private Bitmap resultBitmap;
    private boolean busy;

    private TextView modelStatus;
    private TextView status;
    private ImageView sourcePreview;
    private ImageView targetPreview;
    private ImageView resultPreview;
    private Button swapButton;
    private Button shareButton;
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
        TextView subtitle = text("S24U 本地静态换脸 · ONNX Runtime\n仅保留换脸功能，不包含视频/运镜模型", 14, false);
        subtitle.setTextColor(0xff5f6368);
        root.addView(subtitle, marginTop(4));

        root.addView(section("1  模型"), marginTop(22));
        modelStatus = text("", 14, false);
        root.addView(modelStatus, marginTop(8));
        Button modelButton = action("导入模型包 ZIP");
        modelButton.setOnClickListener(v -> pickModel());
        root.addView(modelButton, marginTop(10));

        root.addView(section("2  源人脸"), marginTop(22));
        Button sourceButton = action("选择源人脸图片");
        sourceButton.setOnClickListener(v -> pickImage(PICK_SOURCE));
        root.addView(sourceButton, marginTop(8));
        sourcePreview = preview();
        root.addView(sourcePreview, previewParams());

        root.addView(section("3  目标图片"), marginTop(22));
        Button targetButton = action("选择目标图片");
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

    private void pickModel() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{
                "application/zip", "application/octet-stream", "application/x-zip-compressed"});
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivityForResult(intent, PICK_MODEL);
    }

    private void pickImage(int requestCode) {
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
            modelStatus.setText("NOT READY · 请导入 det_10g.onnx + w600k_r50.onnx + inswapper_128.onnx + emap.bin 的 ZIP");
            modelStatus.setTextColor(0xffa33a00);
        } else {
            modelStatus.setText(String.format(Locale.US, "READY · 本地模型 %.1f MB · 权重不随 APK 分发",
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
        if (swapButton == null || shareButton == null) return;
        swapButton.setEnabled(!busy && sourceUri != null && targetUri != null && FaceModelStore.active(this) != null);
        shareButton.setEnabled(!busy && resultUri != null);
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
        worker.shutdownNow();
        if (resultBitmap != null && !resultBitmap.isRecycled()) resultBitmap.recycle();
        super.onDestroy();
    }
}
