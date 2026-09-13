package com.qujindai.localvideo;

import android.Manifest;
import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.ClipData;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.DisplayCutout;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.lang.ref.WeakReference;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/** Native, on-device photo workbench. The retained session never owns an Activity. */
public final class FaceSwapActivity extends Activity {
    private static final int PICK_SOURCE = 201;
    private static final int PICK_TARGET = 202;
    private static final int STORAGE_PERMISSION = 203;
    private static final int ACTION_SAVE = 1;
    private static final int ACTION_LOAD = 2;
    private static final int ACTION_VIEW = 3;
    private static final int ACTION_SHARE = 4;
    private static final int ACCENT = 0xff00796b;
    private static final int TEXT = 0xff263238;
    private static final int MUTED = 0xff66706f;
    private static final int CARD = 0xfff5f7f7;
    private static final int ERROR = 0xffa32626;

    private Session session;
    private boolean resumed;
    private ImageView sourcePreview;
    private ImageView targetPreview;
    private ImageView resultPreview;
    private TextView sourceNote;
    private TextView targetNote;
    private TextView modelStatus;
    private TextView status;
    private TextView resultMeta;
    private TextView details;
    private Button sourceButton;
    private Button targetButton;
    private Button prepareButton;
    private Button generateButton;
    private Button cancelButton;
    private Button saveButton;
    private Button viewButton;
    private Button shareButton;
    private Button detailsButton;
    private Button videoButton;
    private ProgressBar progress;
    private LinearLayout resultCard;
    private LinearLayout historyRows;
    private String historyKey = "";

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Object retained = getLastNonConfigurationInstance();
        session = retained instanceof Session ? (Session) retained
                : new Session(getApplicationContext(), savedInstanceState);
        getWindow().setStatusBarColor(Color.WHITE);
        getWindow().setNavigationBarColor(Color.WHITE);
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR | View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR);
        setContentView(buildUi());
        session.attach(this);
        render();
    }

    @Override protected void onResume() {
        super.onResume();
        resumed = true;
        render();
        if (!session.busy()) session.refresh();
    }

    @Override protected void onPause() {
        resumed = false;
        super.onPause();
    }

    @Override public Object onRetainNonConfigurationInstance() {
        return session;
    }

    @Override protected void onSaveInstanceState(Bundle out) {
        if (session.source.uri != null) out.putString("face_source", session.source.uri.toString());
        if (session.target.uri != null) out.putString("face_target", session.target.uri.toString());
        out.putBoolean("face_details", session.detailsExpanded);
        out.putBoolean("face_interrupted", session.work != Work.IDLE || session.fullResult != null);
        out.putInt("face_picker", session.pickerRequest);
        super.onSaveInstanceState(out);
    }

    @Override protected void onDestroy() {
        session.detach(this);
        // Rendered bitmaps are released by dropping references, never recycled while
        // the render thread may still be drawing a previous display list.
        sourcePreview.setImageDrawable(null);
        targetPreview.setImageDrawable(null);
        resultPreview.setImageDrawable(null);
        if (!isChangingConfigurations()) session.close();
        super.onDestroy();
    }

    private View buildUi() {
        FrameLayout viewport = new FrameLayout(this);
        viewport.setBackgroundColor(Color.WHITE);
        viewport.setClipToPadding(true);
        viewport.setOnApplyWindowInsetsListener((view, insets) -> {
            if (Build.VERSION.SDK_INT >= 30) {
                android.graphics.Insets bars = insets.getInsets(
                        WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                viewport.setPadding(bars.left, bars.top, bars.right, bars.bottom);
            } else {
                DisplayCutout cutout = insets.getDisplayCutout();
                viewport.setPadding(Math.max(insets.getSystemWindowInsetLeft(),
                                cutout == null ? 0 : cutout.getSafeInsetLeft()),
                        Math.max(insets.getSystemWindowInsetTop(),
                                cutout == null ? 0 : cutout.getSafeInsetTop()),
                        Math.max(insets.getSystemWindowInsetRight(),
                                cutout == null ? 0 : cutout.getSafeInsetRight()),
                        Math.max(insets.getSystemWindowInsetBottom(),
                                cutout == null ? 0 : cutout.getSafeInsetBottom()));
            }
            return insets;
        });
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        viewport.addView(scroll, new FrameLayout.LayoutParams(-1, -1));
        LinearLayout root = column();
        root.setPadding(dp(16), dp(18), dp(16), dp(32));
        scroll.addView(root, new ScrollView.LayoutParams(-1, -2));
        root.addView(text("FACE-RE", 27, true));
        root.addView(muted("照片换脸 · 在本机完成", 15));
        root.addView(text("参考人脸提供面部身份特征，目标照片决定构图、动作和背景。"
                + "首版每张照片仅支持一张清晰可用的人脸。", 14, false));

        LinearLayout sourceCard = card(root, "1  参考人脸");
        sourceCard.addView(muted("选择清晰的单人照片，作为面部身份参考。", 13));
        sourcePreview = preview("参考人脸预览", 180);
        sourceCard.addView(sourcePreview);
        sourceNote = muted("尚未选择参考人脸", 13);
        sourceCard.addView(sourceNote);
        sourceButton = button("选择参考人脸", false);
        sourceButton.setOnClickListener(v -> pick(PICK_SOURCE));
        sourceCard.addView(sourceButton, full(6));

        LinearLayout targetCard = card(root, "2  目标照片");
        targetCard.addView(muted("选择需要换脸的单人照片，保留这张照片的构图、动作和背景。", 13));
        targetPreview = preview("目标照片预览", 180);
        targetCard.addView(targetPreview);
        targetNote = muted("尚未选择目标照片", 13);
        targetCard.addView(targetNote);
        targetButton = button("选择目标照片", false);
        targetButton.setOnClickListener(v -> pick(PICK_TARGET));
        targetCard.addView(targetButton, full(6));

        LinearLayout modelCard = card(root, "3  准备模型");
        modelStatus = text("正在检查模型状态…", 14, true);
        modelCard.addView(modelStatus);
        modelCard.addView(muted("首次准备需下载约 699 MiB，建议连接 Wi-Fi 并预留足够空间。"
                + "照片处理在本机进行。", 13));
        prepareButton = button("准备换脸模型", false);
        prepareButton.setOnClickListener(v -> session.install());
        modelCard.addView(prepareButton, full(6));

        generateButton = button("开始换脸", true);
        generateButton.setOnClickListener(v -> session.generate());
        root.addView(generateButton, full(14));
        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progress.setMax(100);
        progress.setProgressTintList(ColorStateList.valueOf(ACCENT));
        root.addView(progress, full(8));
        status = text("", 14, false);
        status.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
        root.addView(status);
        cancelButton = button("取消任务", false);
        cancelButton.setOnClickListener(v -> session.cancel());
        root.addView(cancelButton, full(4));

        resultCard = card(root, "最近完成的结果");
        resultPreview = preview("换脸结果预览", 280);
        resultCard.addView(resultPreview);
        resultMeta = muted("", 13);
        resultCard.addView(resultMeta);
        saveButton = button("保存到相册", true);
        saveButton.setOnClickListener(v -> requestAction(ACTION_SAVE, null));
        resultCard.addView(saveButton, full(6));
        viewButton = button("查看已保存图片", false);
        viewButton.setOnClickListener(v -> requestAction(ACTION_VIEW, session.savedResult));
        resultCard.addView(viewButton, full(6));
        shareButton = button("分享已保存图片", false);
        shareButton.setOnClickListener(v -> requestAction(ACTION_SHARE, session.savedResult));
        resultCard.addView(shareButton, full(6));

        LinearLayout historyCard = card(root, "最近保存 · 最多 5 条");
        historyCard.addView(muted("记录只保存在本机。较早记录移出列表后，相册中的图片仍会保留。", 13));
        historyRows = column();
        historyCard.addView(historyRows);

        detailsButton = button("展开详细信息", false);
        detailsButton.setOnClickListener(v -> {
            session.detailsExpanded = !session.detailsExpanded;
            render();
        });
        root.addView(detailsButton, full(14));
        details = muted("", 12);
        details.setTextIsSelectable(true);
        root.addView(details);
        videoButton = button("打开原视频实验", false);
        videoButton.setOnClickListener(v -> {
            try {
                startActivity(new Intent(this, MainActivityV05.class));
            } catch (ActivityNotFoundException error) {
                session.uiError("无法打开视频实验", error);
            }
        });
        root.addView(videoButton, full(12));
        viewport.requestApplyInsets();
        return viewport;
    }

    private void render() {
        boolean idle = !session.busy();
        sourcePreview.setImageBitmap(session.source.preview);
        targetPreview.setImageBitmap(session.target.preview);
        sourcePreview.setVisibility(session.source.preview == null ? View.GONE : View.VISIBLE);
        targetPreview.setVisibility(session.target.preview == null ? View.GONE : View.VISIBLE);
        sourceNote.setText(session.source.note);
        targetNote.setText(session.target.note);
        enable(sourceButton, idle);
        enable(targetButton, idle);
        sourceButton.setText(session.source.uri == null ? "选择参考人脸" : "更换参考人脸");
        targetButton.setText(session.target.uri == null ? "选择目标照片" : "更换目标照片");
        modelStatus.setText(session.ready ? "模型已就绪，可在本机换脸"
                : session.work == Work.INSTALL ? "正在准备模型…" : "模型尚未就绪");
        enable(prepareButton, idle && !session.ready);
        prepareButton.setText(session.ready ? "模型已准备好" : "准备换脸模型");
        enable(generateButton, idle && session.ready && session.source.uri != null
                && session.target.uri != null && session.source.preview != null
                && session.target.preview != null);
        enable(videoButton, idle);
        progress.setVisibility(session.work == Work.IDLE ? View.GONE : View.VISIBLE);
        progress.setIndeterminate(session.work == Work.CHECK || session.work == Work.PREVIEW
                || session.work == Work.LOAD || session.work == Work.OPEN);
        progress.setProgress(session.percent);
        status.setText(session.message);
        status.setTextColor(session.error ? ERROR : MUTED);
        boolean cancellable = session.work == Work.INSTALL || session.work == Work.GENERATE
                || session.work == Work.SAVE;
        cancelButton.setVisibility(cancellable ? View.VISIBLE : View.GONE);
        enable(cancelButton, cancellable && !session.cancelled.get());
        cancelButton.setText(session.cancelled.get() ? "正在取消…" : "取消任务");

        boolean hasResult = session.outputPreview != null || session.savedResult != null;
        resultCard.setVisibility(hasResult ? View.VISIBLE : View.GONE);
        resultPreview.setImageBitmap(session.outputPreview);
        resultMeta.setText(session.resultDescription());
        saveButton.setText(session.savedResult == null ? "保存到相册" : "已保存到相册");
        enable(saveButton, idle && session.fullResult != null && session.savedResult == null);
        enable(viewButton, idle && session.savedResult != null);
        enable(shareButton, idle && session.savedResult != null);
        rebuildHistory(idle);
        detailsButton.setText(session.detailsExpanded ? "收起详细信息" : "展开详细信息");
        details.setVisibility(session.detailsExpanded ? View.VISIBLE : View.GONE);
        details.setText("推理：CPU · ONNX Runtime · 本机运行\n"
                + "模型：3 个 FP32 模型，约 699 MiB\n"
                + "SCRFD 2.5G / ArcFace W600K R50 / INSwapper 128\n"
                + "模型状态：" + session.modelDetail
                + (session.resultModel.isEmpty() ? "" : "\n结果模型：" + session.resultModel)
                + "\n" + session.diagnostic);
        if (resumed && session.externalAction != null && !isFinishing() && !isDestroyed()) {
            PendingAction action = session.externalAction;
            session.externalAction = null;
            openExternal(action);
        }
    }

    private void rebuildHistory(boolean idle) {
        StringBuilder key = new StringBuilder(Boolean.toString(idle));
        for (FacePhotoHistory.Entry entry : session.history) {
            key.append('|').append(entry.uri).append(':').append(entry.createdAtMs);
        }
        if (key.toString().equals(historyKey)) return;
        historyKey = key.toString();
        historyRows.removeAllViews();
        if (session.history.isEmpty()) {
            historyRows.addView(muted("还没有已保存的照片。生成完成后，点击“保存到相册”。", 13));
        }
        for (FacePhotoHistory.Entry entry : session.history) {
            Button row = button(date(entry.createdAtMs) + " · " + entry.width + " × "
                    + entry.height + "\n载入已保存结果", false);
            enable(row, idle);
            row.setOnClickListener(v -> requestAction(ACTION_LOAD, entry));
            historyRows.addView(row, full(6));
        }
    }

    private void pick(int request) {
        if (session.busy()) return;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        intent.putExtra(Intent.EXTRA_LOCAL_ONLY, true);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        session.pickerRequest = request;
        render();
        try {
            startActivityForResult(intent, request);
        } catch (ActivityNotFoundException | SecurityException error) {
            session.pickerRequest = 0;
            session.uiError("无法打开照片选择器", error);
        }
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_SOURCE && requestCode != PICK_TARGET) return;
        session.pickerRequest = 0;
        if (resultCode != RESULT_OK || data == null) {
            render();
            return;
        }
        Uri uri = data.getData();
        if (!FacePhotoHistory.isContentUri(uri)) {
            session.uiError("请选择可读取的本机照片", new IOException("照片选择器未返回有效内容地址"));
            return;
        }
        session.select(requestCode == PICK_SOURCE, uri, data.getFlags());
    }

    private void requestAction(int type, FacePhotoHistory.Entry entry) {
        if (session.busy()) return;
        if (type == ACTION_SAVE && (session.fullResult == null || session.savedResult != null)) return;
        if (type != ACTION_SAVE && entry == null) return;
        PendingAction action = new PendingAction(type, entry);
        boolean needsRead = Build.VERSION.SDK_INT <= 28
                && checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED;
        boolean needsWrite = type == ACTION_SAVE && Build.VERSION.SDK_INT <= 28
                && checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED;
        if (needsRead || needsWrite) {
            session.permissionAction = action;
            session.message = type == ACTION_SAVE ? "保存到系统相册需要存储权限。" : "读取已保存照片需要存储权限。";
            render();
            List<String> permissions = new ArrayList<>();
            if (needsRead) permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE);
            if (needsWrite) permissions.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
            try {
                requestPermissions(permissions.toArray(new String[0]), STORAGE_PERMISSION);
            } catch (RuntimeException error) {
                session.permissionAction = null;
                session.uiError("无法请求存储权限", error);
            }
            return;
        }
        performAction(action);
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode != STORAGE_PERMISSION) return;
        PendingAction action = session.permissionAction;
        session.permissionAction = null;
        if (action == null) { render(); return; }
        boolean granted = results.length > 0;
        for (int result : results) granted &= result == PackageManager.PERMISSION_GRANTED;
        if (granted) {
            performAction(action);
        } else {
            session.uiError("存储权限未授予，操作未完成。可再次点击按钮重试。",
                    new SecurityException("Android 9 需要 READ/WRITE_EXTERNAL_STORAGE 权限；已生成的结果仍可在当前页预览。"));
        }
    }

    private void performAction(PendingAction action) {
        if (action.type == ACTION_SAVE) session.save();
        else if (action.type == ACTION_LOAD) session.load(action.entry);
        else session.open(action);
    }

    private void openExternal(PendingAction action) {
        Uri uri = action.entry.uri;
        Intent intent;
        if (action.type == ACTION_SHARE) {
            intent = new Intent(Intent.ACTION_SEND).setType("image/png");
            intent.putExtra(Intent.EXTRA_STREAM, uri);
        } else {
            intent = new Intent(Intent.ACTION_VIEW).setDataAndType(uri, "image/png");
        }
        intent.setClipData(ClipData.newRawUri("FACE-RE 结果", uri));
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        try {
            Intent chooser = Intent.createChooser(intent, action.type == ACTION_SHARE ? "分享换脸照片" : "查看换脸照片");
            chooser.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(chooser);
        } catch (ActivityNotFoundException | SecurityException error) {
            session.uiError("无法打开图片，请检查图片是否仍存在以及是否安装了可用应用。", error);
        }
    }

    private LinearLayout column() {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        return layout;
    }

    private LinearLayout card(LinearLayout parent, String title) {
        LinearLayout card = column();
        card.setPadding(dp(14), dp(12), dp(14), dp(12));
        card.setBackground(rounded(CARD, 16));
        card.addView(text(title, 18, true));
        parent.addView(card, full(12));
        return card;
    }

    private TextView text(String value, int size, boolean bold) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextColor(TEXT);
        view.setTextSize(size);
        if (bold) view.setTypeface(view.getTypeface(), Typeface.BOLD);
        view.setPadding(0, dp(5), 0, dp(5));
        return view;
    }

    private TextView muted(String value, int size) {
        TextView view = text(value, size, false);
        view.setTextColor(MUTED);
        return view;
    }

    private ImageView preview(String description, int height) {
        ImageView view = new ImageView(this);
        view.setContentDescription(description);
        view.setScaleType(ImageView.ScaleType.FIT_CENTER);
        view.setBackground(rounded(0xffe7edec, 12));
        view.setLayoutParams(new LinearLayout.LayoutParams(-1, dp(height)));
        return view;
    }

    private Button button(String label, boolean primary) {
        Button button = new Button(this);
        button.setAllCaps(false);
        button.setText(label);
        button.setTextSize(15);
        button.setTextColor(primary ? Color.WHITE : TEXT);
        button.setBackgroundTintList(null);
        button.setBackground(rounded(primary ? ACCENT : 0xffdde3e2, 12));
        button.setMinHeight(dp(52));
        button.setPadding(dp(12), dp(8), dp(12), dp(8));
        return button;
    }

    private GradientDrawable rounded(int color, int radius) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(color);
        drawable.setCornerRadius(dp(radius));
        return drawable;
    }

    private LinearLayout.LayoutParams full(int top) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.topMargin = dp(top);
        return params;
    }

    private static void enable(View view, boolean enabled) {
        view.setEnabled(enabled);
        view.setAlpha(enabled ? 1f : .48f);
    }

    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }

    private static String date(long time) {
        return time <= 0 ? "保存时间未知" : new SimpleDateFormat("MM-dd HH:mm", Locale.CHINA).format(new Date(time));
    }

    private enum Work {
        IDLE("待机"), CHECK("正在检查模型和照片"), PREVIEW("正在读取照片"),
        INSTALL("正在准备模型"), GENERATE("正在本机换脸"), SAVE("正在保存到相册"),
        LOAD("正在载入已保存结果"), OPEN("正在检查已保存图片");
        final String label;
        Work(String label) { this.label = label; }
    }

    private static final class Input {
        Uri uri;
        Bitmap preview;
        String note;
        Input(String note) { this.note = note; }
    }

    private static final class PendingAction {
        final int type;
        final FacePhotoHistory.Entry entry;
        PendingAction(int type, FacePhotoHistory.Entry entry) { this.type = type; this.entry = entry; }
    }

    private static final class InputUnavailable extends IOException {
        final boolean source;
        InputUnavailable(boolean source, Throwable cause) {
            super((source ? "参考人脸" : "目标照片") + "无法读取，可能已移动、删除或授权失效，请重新选择。", cause);
            this.source = source;
        }
    }

    /** Mutable UI state is main-thread confined; worker results cross via post(). */
    private static final class Session {
        private static final int PREVIEW_EDGE = 1024;
        final Context app;
        final Handler main = new Handler(Looper.getMainLooper());
        final ExecutorService worker = Executors.newSingleThreadExecutor();
        final AtomicBoolean closed = new AtomicBoolean();
        final Input source = new Input("尚未选择参考人脸");
        final Input target = new Input("尚未选择目标照片");
        private WeakReference<FaceSwapActivity> activity = new WeakReference<>(null);
        // Construct/access these on the worker so model checks and preference reads
        // never block rendering. Neither object is given the Activity context.
        private FaceModelStore models;
        private FacePhotoHistory photos;
        Work work = Work.IDLE;
        AtomicBoolean cancelled = new AtomicBoolean();
        boolean ready;
        boolean error;
        boolean detailsExpanded;
        int pickerRequest;
        int percent;
        String message = "选择参考人脸和目标照片，再准备模型。";
        String modelDetail = "尚未检查";
        String diagnostic = "";
        List<FacePhotoHistory.Entry> history = new ArrayList<>();
        Bitmap fullResult; // Never attached to a View; exclusively borrowed by save().
        Bitmap outputPreview;
        FacePhotoHistory.Entry savedResult;
        long resultTime;
        long resultElapsed;
        int resultWidth;
        int resultHeight;
        String resultModel = "";
        PendingAction permissionAction;
        PendingAction externalAction;

        Session(Context app, Bundle saved) {
            this.app = app;
            if (saved != null) {
                source.uri = restoredUri(saved.getString("face_source"));
                target.uri = restoredUri(saved.getString("face_target"));
                detailsExpanded = saved.getBoolean("face_details");
                pickerRequest = saved.getInt("face_picker", 0);
                if (saved.getBoolean("face_interrupted")) {
                    message = "上次会话已中断，未保存结果需重新生成；已保存图片可从最近记录载入。";
                }
            }
        }

        private static Uri restoredUri(String value) {
            try {
                Uri uri = value == null ? null : Uri.parse(value);
                return FacePhotoHistory.isContentUri(uri) ? uri : null;
            } catch (RuntimeException ignored) { return null; }
        }

        void attach(FaceSwapActivity owner) { activity = new WeakReference<>(owner); }

        void detach(FaceSwapActivity owner) {
            if (activity.get() == owner) activity.clear();
        }

        private void changed() {
            FaceSwapActivity owner = activity.get();
            if (owner != null && !owner.isDestroyed()) owner.render();
        }

        boolean busy() {
            return closed.get() || work != Work.IDLE || pickerRequest != 0
                    || permissionAction != null || externalAction != null;
        }

        private AtomicBoolean begin(Work next) {
            if (busy()) return null;
            work = next;
            cancelled = new AtomicBoolean();
            error = false;
            percent = 0;
            message = next.label + "…";
            changed();
            return cancelled;
        }

        private FaceModelStore models() {
            if (models == null) models = new FaceModelStore(app);
            return models;
        }

        private FacePhotoHistory photos() {
            if (photos == null) photos = new FacePhotoHistory(app);
            return photos;
        }

        private void post(Runnable update, Bitmap... unclaimed) {
            main.post(() -> {
                if (closed.get()) {
                    for (Bitmap bitmap : unclaimed) recycle(bitmap);
                    return;
                }
                update.run();
            });
        }

        void refresh() {
            String previousMessage = message;
            boolean previousError = error;
            AtomicBoolean token = begin(Work.CHECK);
            if (token == null) return;
            final Uri sourceUri = source.uri;
            final Uri targetUri = target.uri;
            final boolean needSource = source.preview == null;
            final boolean needTarget = target.preview == null;
            worker.execute(() -> {
                Bitmap left = null;
                Bitmap right = null;
                try {
                    boolean installed = models().isReady();
                    String modelText = models().status();
                    List<FacePhotoHistory.Entry> entries = photos().load();
                    InputUnavailable sourceError = null;
                    InputUnavailable targetError = null;
                    if (sourceUri != null) {
                        try {
                            requireInput(sourceUri, true);
                            if (needSource) left = decodePreview(sourceUri);
                        } catch (IOException | RuntimeException issue) {
                            sourceError = new InputUnavailable(true, issue);
                        }
                    }
                    if (targetUri != null) {
                        try {
                            requireInput(targetUri, false);
                            if (needTarget) right = decodePreview(targetUri);
                        } catch (IOException | RuntimeException issue) {
                            targetError = new InputUnavailable(false, issue);
                        }
                    }
                    final Bitmap sourceBitmap = left;
                    final Bitmap targetBitmap = right;
                    final InputUnavailable sourceIssue = sourceError;
                    final InputUnavailable targetIssue = targetError;
                    post(() -> {
                        ready = installed;
                        modelDetail = modelText;
                        history = entries;
                        if (sourceBitmap != null) { source.preview = sourceBitmap; source.note = "参考人脸已选择"; }
                        if (targetBitmap != null) { target.preview = targetBitmap; target.note = "目标照片已选择"; }
                        work = Work.IDLE;
                        message = previousMessage;
                        error = previousError;
                        if (sourceIssue != null) invalidate(sourceIssue);
                        if (targetIssue != null) invalidate(targetIssue);
                        if (sourceIssue != null || targetIssue != null) {
                            Throwable issue = sourceIssue != null ? sourceIssue : targetIssue;
                            showFailure("照片访问失效", issue);
                        }
                        changed();
                    }, sourceBitmap, targetBitmap);
                } catch (Exception | OutOfMemoryError | LinkageError issue) {
                    recycle(left);
                    recycle(right);
                    refreshModelAfterFailure();
                    fail("状态检查", issue, token);
                }
            });
        }

        void select(boolean isSource, Uri uri, int flags) {
            AtomicBoolean token = begin(Work.PREVIEW);
            if (token == null) return;
            worker.execute(() -> {
                Bitmap bitmap = null;
                try {
                    boolean persisted = false;
                    if ((flags & Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION) != 0) {
                        try {
                            app.getContentResolver().takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
                            persisted = true;
                        } catch (RuntimeException ignored) { /* Temporary access can still work. */ }
                    }
                    requireReadable(uri);
                    bitmap = decodePreview(uri);
                    final Bitmap preview = bitmap;
                    final boolean persistent = persisted;
                    post(() -> {
                        Input input = isSource ? source : target;
                        input.uri = uri;
                        input.preview = preview;
                        input.note = (isSource ? "参考人脸" : "目标照片") + "已选择"
                                + (persistent ? "" : "；访问授权未能保留，重新打开后可能需要重选");
                        work = Work.IDLE;
                        message = "照片已载入。请确认两张照片各有且仅有一张清晰人脸。";
                        changed();
                        // Also restores the other input after process death while the
                        // document picker was open, and rechecks current model readiness.
                        refresh();
                    }, preview);
                } catch (Exception | OutOfMemoryError issue) {
                    recycle(bitmap);
                    // Preserve the earlier valid selection when its replacement fails.
                    fail("读取照片", issue, token);
                }
            });
        }

        void install() {
            if (ready) return;
            AtomicBoolean token = begin(Work.INSTALL);
            if (token == null) return;
            worker.execute(() -> {
                try {
                    models().install(progress(token, Work.INSTALL), token);
                    boolean installed = models().isReady();
                    String modelText = models().status();
                    if (!installed) throw new IOException("模型安装未通过完整性检查，请重新准备模型。");
                    post(() -> {
                        ready = installed;
                        modelDetail = modelText;
                        work = Work.IDLE;
                        percent = 100;
                        message = "模型已就绪。选择两张单人照片即可开始换脸。";
                        changed();
                    });
                } catch (Exception | OutOfMemoryError | LinkageError issue) {
                    refreshModelAfterFailure();
                    fail("模型准备", issue, token);
                }
            });
        }

        void generate() {
            if (!ready || source.uri == null || target.uri == null
                    || source.preview == null || target.preview == null) return;
            // Snapshot the job inputs before dispatch; no task reads mutable picker state.
            final Uri sourceUri = source.uri;
            final Uri targetUri = target.uri;
            AtomicBoolean token = begin(Work.GENERATE);
            if (token == null) return;
            worker.execute(() -> {
                Bitmap full = null;
                Bitmap small = null;
                try {
                    FacePhotoHistory.checkCancelled(token);
                    if (!models().isReady()) throw new IOException("换脸模型已失效，请重新准备模型。");
                    requireInput(sourceUri, true);
                    requireInput(targetUri, false);
                    FaceSwapEngine.Result result = FaceSwapEngine.generate(app, sourceUri, targetUri,
                            progress(token, Work.GENERATE), token);
                    if (result == null || result.image == null || result.image.isRecycled()) {
                        throw new IOException("换脸未返回有效图片，未生成结果。");
                    }
                    full = result.image;
                    FacePhotoHistory.checkCancelled(token);
                    small = resultPreview(full);
                    FacePhotoHistory.checkCancelled(token);
                    final Bitmap complete = full;
                    final Bitmap preview = small;
                    final long created = System.currentTimeMillis();
                    post(() -> {
                        work = Work.IDLE;
                        if (token.get()) {
                            recycle(complete);
                            recycle(preview);
                            message = "已取消换脸。";
                        } else {
                            recycle(fullResult);
                            fullResult = complete;
                            outputPreview = preview;
                            savedResult = null;
                            resultTime = created;
                            resultElapsed = Math.max(0, result.elapsedMs);
                            resultWidth = complete.getWidth();
                            resultHeight = complete.getHeight();
                            resultModel = result.modelId == null ? "" : result.modelId;
                            percent = 100;
                            message = "换脸完成，请查看结果并保存到相册。";
                        }
                        changed();
                    }, complete, preview);
                } catch (Exception | OutOfMemoryError | LinkageError issue) {
                    recycle(full);
                    recycle(small);
                    refreshModelAfterFailure();
                    fail("换脸", issue, token);
                }
            });
        }

        void save() {
            if (fullResult == null || savedResult != null) return;
            AtomicBoolean token = begin(Work.SAVE);
            if (token == null) return;
            final Bitmap image = fullResult;
            final long created = resultTime;
            final long elapsed = resultElapsed;
            final String model = resultModel;
            worker.execute(() -> {
                FacePhotoHistory.Entry published;
                try {
                    published = photos().save(image, created, elapsed, model, token);
                } catch (Exception | OutOfMemoryError issue) {
                    fail("保存", issue, token);
                    return;
                }
                // Once published, history failures and late cancellation must never
                // report a failed save or re-enable saving this result a second time.
                boolean recorded = false;
                List<FacePhotoHistory.Entry> entries = null;
                try {
                    recorded = photos().record(published);
                    entries = photos().load();
                } catch (RuntimeException | OutOfMemoryError ignored) { /* Photo is already saved. */ }
                final boolean historyWritten = recorded;
                final List<FacePhotoHistory.Entry> recent = entries;
                post(() -> {
                    savedResult = published;
                    if (recent != null) history = recent;
                    recycle(fullResult);
                    fullResult = null;
                    work = Work.IDLE;
                    percent = 100;
                    message = historyWritten ? "已保存到 Pictures/LocalVideoLab，可查看或分享。"
                            : "图片已保存到相册，但最近记录未能写入。仍可查看或分享当前结果。";
                    error = !historyWritten;
                    changed();
                });
            });
        }

        void load(FacePhotoHistory.Entry entry) {
            if (entry == null) return;
            AtomicBoolean token = begin(Work.LOAD);
            if (token == null) return;
            worker.execute(() -> {
                Bitmap small = null;
                try {
                    requireReadable(entry.uri);
                    small = decodePreview(entry.uri);
                    final Bitmap preview = small;
                    post(() -> {
                        recycle(fullResult);
                        fullResult = null;
                        outputPreview = preview;
                        savedResult = entry;
                        resultTime = entry.createdAtMs;
                        resultElapsed = entry.elapsedMs;
                        resultWidth = entry.width;
                        resultHeight = entry.height;
                        resultModel = entry.modelId;
                        work = Work.IDLE;
                        message = "已载入保存结果，可查看或分享。";
                        changed();
                    }, preview);
                } catch (Exception | OutOfMemoryError issue) {
                    recycle(small);
                    fail("载入历史图片（图片可能已删除或访问权限失效）", issue, token);
                }
            });
        }

        void open(PendingAction action) {
            AtomicBoolean token = begin(Work.OPEN);
            if (token == null) return;
            worker.execute(() -> {
                try {
                    requireReadable(action.entry.uri);
                    post(() -> {
                        work = Work.IDLE;
                        externalAction = action;
                        message = "已保存图片可供查看或分享。";
                        changed();
                    });
                } catch (Exception issue) {
                    fail("读取已保存图片（图片可能已删除或访问权限失效）", issue, token);
                }
            });
        }

        private FaceSwapEngine.Progress progress(AtomicBoolean token, Work expected) {
            return new FaceSwapEngine.Progress() {
                private long last;
                @Override public void update(int value, String text) {
                    long now = SystemClock.uptimeMillis();
                    if (value != 100 && now - last < 150) return;
                    last = now;
                    post(() -> {
                        if (cancelled != token || work != expected) return;
                        percent = Math.max(0, Math.min(100, value));
                        diagnostic = text == null ? "" : text;
                        if (!token.get()) message = expected.label + " · " + percent + "%";
                        changed();
                    });
                }
            };
        }

        void cancel() {
            if (work == Work.IDLE) return;
            cancelled.set(true);
            message = "正在取消，请等待当前处理结束…";
            changed();
        }

        private void refreshModelAfterFailure() {
            try {
                boolean installed = models().isReady();
                String modelText = models().status();
                post(() -> { ready = installed; modelDetail = modelText; });
            } catch (Exception | OutOfMemoryError | LinkageError issue) {
                post(() -> { ready = false; modelDetail = detail(issue); });
            }
        }

        private void fail(String stage, Throwable issue, AtomicBoolean token) {
            post(() -> {
                if (cancelled != token) return;
                work = Work.IDLE;
                percent = 0;
                if (issue instanceof CancellationException
                        || (token.get() && !(issue instanceof FacePhotoHistory.CleanupFailure))) {
                    message = "已取消" + stage + "。";
                    error = false;
                    diagnostic = detail(issue);
                } else {
                    if (issue instanceof InputUnavailable) invalidate((InputUnavailable) issue);
                    showFailure(stage, issue);
                }
                changed();
            });
        }

        private void invalidate(InputUnavailable issue) {
            Input input = issue.source ? source : target;
            input.uri = null;
            input.preview = null;
            input.note = issue.getMessage();
        }

        void uiError(String message, Throwable issue) {
            error = true;
            this.message = message;
            diagnostic = detail(issue);
            detailsExpanded = true;
            changed();
        }

        private void showFailure(String stage, Throwable issue) {
            error = true;
            diagnostic = detail(issue);
            detailsExpanded = true;
            String reason = issue instanceof OutOfMemoryError
                    ? "可用内存不足，请关闭其他应用或换较小的照片后重试。" : issue.getMessage();
            if (reason == null || reason.trim().isEmpty()) reason = "请查看下方详细信息后重试。";
            if (reason.length() > 240) reason = reason.substring(0, 240) + "…";
            message = stage + "失败：" + reason;
        }

        String resultDescription() {
            if (outputPreview == null && savedResult == null) return "";
            return date(resultTime) + " · " + resultWidth + " × " + resultHeight
                    + String.format(Locale.CHINA, " · 用时 %.1f 秒", resultElapsed / 1000.0)
                    + (savedResult == null ? "\n尚未保存；离开或替换结果前请先保存。"
                    : "\n已保存到相册 · Pictures/LocalVideoLab");
        }

        private void requireInput(Uri uri, boolean source) throws InputUnavailable {
            try { requireReadable(uri); }
            catch (IOException | RuntimeException issue) { throw new InputUnavailable(source, issue); }
        }

        private void requireReadable(Uri uri) throws IOException {
            if (!FacePhotoHistory.isContentUri(uri)) throw new FileNotFoundException("图片地址无效，请重新选择。");
            try (InputStream stream = app.getContentResolver().openInputStream(uri)) {
                if (stream == null || stream.read() == -1) throw new FileNotFoundException("图片不存在或内容为空。");
            }
        }

        private Bitmap decodePreview(Uri uri) throws IOException {
            return ImageDecoder.decodeBitmap(ImageDecoder.createSource(app.getContentResolver(), uri),
                    (decoder, info, source) -> {
                        int[] size = previewSize(info.getSize().getWidth(), info.getSize().getHeight());
                        decoder.setTargetSize(size[0], size[1]);
                        decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                        decoder.setOnPartialImageListener(error -> false);
                    });
        }

        private static Bitmap resultPreview(Bitmap image) throws IOException {
            int[] size = previewSize(image.getWidth(), image.getHeight());
            Bitmap preview = Bitmap.createScaledBitmap(image, size[0], size[1], true);
            // createScaledBitmap may return its input. Saving and disposing the raw
            // result must never recycle a bitmap held by an ImageView.
            if (preview == image) preview = image.copy(Bitmap.Config.ARGB_8888, false);
            if (preview == null) throw new IOException("无法创建结果预览");
            return preview;
        }

        private static int[] previewSize(int width, int height) {
            if (width <= 0 || height <= 0) throw new IllegalArgumentException("图片尺寸无效");
            double scale = Math.min(1.0, PREVIEW_EDGE / (double) Math.max(width, height));
            return new int[] { Math.max(1, (int) Math.round(width * scale)),
                    Math.max(1, (int) Math.round(height * scale)) };
        }

        private static String detail(Throwable issue) {
            String text = issue.getClass().getSimpleName() + ": " + String.valueOf(issue.getMessage());
            Throwable cause = issue.getCause();
            if (cause != null && cause != issue) text += "\n" + cause.getClass().getSimpleName() + ": " + cause.getMessage();
            return text;
        }

        void close() {
            if (!closed.compareAndSet(false, true)) return;
            activity.clear();
            cancelled.set(true);
            Bitmap raw = fullResult;
            fullResult = null;
            source.preview = null;
            target.preview = null;
            outputPreview = null;
            permissionAction = null;
            externalAction = null;
            // Queue disposal after any save still borrowing raw. Shutdown is graceful:
            // cancellation/finally cleanup completes; the main thread never waits.
            worker.execute(() -> recycle(raw));
            worker.shutdown();
        }

        private static void recycle(Bitmap bitmap) {
            if (bitmap != null && !bitmap.isRecycled()) bitmap.recycle();
        }
    }
}
