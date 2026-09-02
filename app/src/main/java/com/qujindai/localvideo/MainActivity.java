package com.qujindai.localvideo;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Debug;
import android.os.PowerManager;
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

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int PICK_PRIMARY = 100;
    private static final int PICK_SECONDARY = 101;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private Uri primaryUri;
    private Uri secondaryUri;
    private Uri lastVideoUri;

    private ImageView primaryPreview;
    private TextView secondaryLabel;
    private TextView statusView;
    private TextView metricsView;
    private ProgressBar progressBar;
    private Button generateButton;
    private Button openButton;
    private Spinner framesSpinner;
    private Spinner fpsSpinner;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(buildUi());
        updateMetrics("待机");
    }

    private View buildUi() {
        int pad = dp(16);
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);
        scroll.addView(root, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView title = text("Local Video Lab", 26, true);
        root.addView(title);
        TextView subtitle = text(
                "完全离线 · RIFE v4.6 · ncnn · Vulkan\n单图模式生成轻微镜头运动；双图模式生成神经插值视频。", 15, false);
        subtitle.setTextColor(Color.DKGRAY);
        root.addView(subtitle);

        primaryPreview = new ImageView(this);
        primaryPreview.setBackgroundColor(0xffeeeeee);
        primaryPreview.setScaleType(ImageView.ScaleType.CENTER_CROP);
        LinearLayout.LayoutParams previewParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(220));
        previewParams.topMargin = dp(14);
        root.addView(primaryPreview, previewParams);

        Button primaryButton = button("选择主图");
        primaryButton.setOnClickListener(v -> pickImage(PICK_PRIMARY));
        root.addView(primaryButton);

        Button secondaryButton = button("选择第二张图（可选）");
        secondaryButton.setOnClickListener(v -> pickImage(PICK_SECONDARY));
        root.addView(secondaryButton);

        secondaryLabel = text("第二张：未选择 → 使用单图运动模式", 14, false);
        root.addView(secondaryLabel);

        Button clearSecondary = button("清除第二张，切回单图模式");
        clearSecondary.setOnClickListener(v -> {
            secondaryUri = null;
            secondaryLabel.setText("第二张：未选择 → 使用单图运动模式");
        });
        root.addView(clearSecondary);

        TextView quality = text("生成参数", 17, true);
        LinearLayout.LayoutParams qualityParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        qualityParams.topMargin = dp(12);
        root.addView(quality, qualityParams);

        framesSpinner = spinner(new String[] { "9 帧（快速验收）", "17 帧（推荐）" }, 1);
        fpsSpinner = spinner(new String[] { "6 FPS", "8 FPS", "12 FPS" }, 1);
        root.addView(framesSpinner);
        root.addView(fpsSpinner);

        generateButton = button("开始本地生成 MP4");
        generateButton.setOnClickListener(v -> startGeneration());
        root.addView(generateButton);

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(100);
        root.addView(progressBar, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(18)));

        statusView = text("请选择图片。", 15, false);
        root.addView(statusView);
        metricsView = text("", 13, false);
        metricsView.setTextIsSelectable(true);
        root.addView(metricsView);

        openButton = button("打开生成视频");
        openButton.setEnabled(false);
        openButton.setOnClickListener(v -> openLastVideo());
        root.addView(openButton);
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
            getContentResolver().takePersistableUriPermission(uri,
                    data.getFlags() & Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (SecurityException ignored) {
            // The immediate read grant is still enough for the current generation session.
        }
        if (requestCode == PICK_PRIMARY) {
            primaryUri = uri;
            primaryPreview.setImageURI(uri);
            statusView.setText("主图已选择。可直接生成，或再选择第二张图。");
        } else if (requestCode == PICK_SECONDARY) {
            secondaryUri = uri;
            secondaryLabel.setText("第二张：已选择 → 双图神经插值模式");
        }
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
        generateButton.setEnabled(false);
        openButton.setEnabled(false);
        progressBar.setProgress(0);
        final int thermalBefore = thermalStatus();

        worker.submit(() -> {
            try {
                RifeEngine engine = new RifeEngine(this);
                RifeEngine.Result result = engine.generate(primaryUri, secondaryUri, frames, fps,
                        (percent, message) -> runOnUiThread(() -> {
                            progressBar.setProgress(percent);
                            statusView.setText(percent + "% · " + message);
                        }));
                lastVideoUri = result.uri;
                int thermalAfter = thermalStatus();
                runOnUiThread(() -> {
                    statusView.setText("100% · 已保存到 Movies/LocalVideoLab");
                    metricsView.setText(String.format(Locale.US,
                            "模型: RIFE v4.6 / ncnn / Vulkan\n模式: %s\n输出: %dx%d · %d 帧 · %d FPS\n耗时: %.2f s\n热状态: %s → %s\nJava max heap: %.0f MB · native heap: %.0f MB",
                            result.singleImageMode ? "单图运动" : "双图插值",
                            result.width, result.height, result.frames, result.fps,
                            result.elapsedMs / 1000.0,
                            thermalName(thermalBefore), thermalName(thermalAfter),
                            Runtime.getRuntime().maxMemory() / 1048576.0,
                            Debug.getNativeHeapAllocatedSize() / 1048576.0));
                    generateButton.setEnabled(true);
                    openButton.setEnabled(true);
                });
            } catch (Throwable error) {
                runOnUiThread(() -> {
                    statusView.setText("生成失败: " + error.getMessage());
                    updateMetrics("失败诊断");
                    generateButton.setEnabled(true);
                });
            }
        });
    }

    private void openLastVideo() {
        if (lastVideoUri == null) return;
        Intent intent = new Intent(Intent.ACTION_VIEW);
        intent.setDataAndType(lastVideoUri, "video/mp4");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivity(Intent.createChooser(intent, "打开视频"));
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

    private Spinner spinner(String[] values, int selected) {
        Spinner spinner = new Spinner(this);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, values);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
        spinner.setSelection(selected);
        return spinner;
    }

    private Button button(String label) {
        Button button = new Button(this);
        button.setText(label);
        return button;
    }

    private TextView text(String value, int sp, boolean bold) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(sp);
        if (bold) view.setTypeface(view.getTypeface(), android.graphics.Typeface.BOLD);
        view.setPadding(0, dp(6), 0, dp(6));
        return view;
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
