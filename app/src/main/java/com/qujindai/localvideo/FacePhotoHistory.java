package com.qujindai.localvideo;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;

/** Local result references only: no input photos, identity vectors, or network access. */
final class FacePhotoHistory {
    private static final String PREFS = "face_re_photo_results";
    private static final String KEY = "recent_results_v1";
    private static final int LIMIT = 5;
    private final Context context;
    private final SharedPreferences preferences;

    static final class CleanupFailure extends IOException {
        CleanupFailure(String message) { super(message); }
        CleanupFailure(String message, Throwable cause) { super(message, cause); }
    }

    FacePhotoHistory(Context context) {
        this.context = context.getApplicationContext();
        preferences = this.context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    static final class Entry {
        final Uri uri;
        final long createdAtMs;
        final long elapsedMs;
        final String modelId;
        final int width;
        final int height;

        Entry(Uri uri, long createdAtMs, long elapsedMs, String modelId, int width, int height) {
            if (!isContentUri(uri)) throw new IllegalArgumentException("结果地址无效");
            this.uri = uri;
            this.createdAtMs = Math.max(0, createdAtMs);
            this.elapsedMs = Math.max(0, elapsedMs);
            this.modelId = modelId == null ? "" : modelId.substring(0, Math.min(256, modelId.length()));
            this.width = Math.max(0, width);
            this.height = Math.max(0, height);
        }
    }

    static boolean isContentUri(Uri uri) {
        return uri != null && "content".equals(uri.getScheme())
                && uri.getAuthority() != null && !uri.getAuthority().isEmpty();
    }

    // All disk work is called from the activity's serial worker.
    List<Entry> load() {
        List<Entry> entries = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        try {
            String encoded = preferences.getString(KEY, "[]");
            if (encoded == null || encoded.length() > 65536) return entries;
            JSONArray rows = new JSONArray(encoded);
            for (int i = 0; i < rows.length() && entries.size() < LIMIT; i++) {
                try {
                    JSONObject row = rows.optJSONObject(i);
                    if (row == null) continue;
                    String value = row.optString("uri", "");
                    if (value.length() > 8192) continue;
                    Uri uri = Uri.parse(value);
                    if (!isContentUri(uri) || !seen.add(value)) continue;
                    entries.add(new Entry(uri, row.optLong("createdAtMs", 0),
                            row.optLong("elapsedMs", 0), row.optString("modelId", ""),
                            row.optInt("width", 0), row.optInt("height", 0)));
                } catch (RuntimeException ignored) {
                    // A malformed row must not hide other usable results.
                }
            }
        } catch (JSONException | RuntimeException ignored) {
            // Also tolerate older preference types and damaged JSON.
        }
        return entries;
    }

    boolean record(Entry entry) {
        if (entry == null) return false;
        try {
            List<Entry> entries = load();
            entries.removeIf(previous -> previous.uri.equals(entry.uri));
            entries.add(0, entry);
            JSONArray rows = new JSONArray();
            for (int i = 0; i < Math.min(LIMIT, entries.size()); i++) {
                Entry item = entries.get(i);
                JSONObject row = new JSONObject();
                row.put("uri", item.uri.toString());
                row.put("createdAtMs", item.createdAtMs);
                row.put("elapsedMs", item.elapsedMs);
                row.put("modelId", item.modelId);
                row.put("width", item.width);
                row.put("height", item.height);
                rows.put(row);
            }
            return preferences.edit().putString(KEY, rows.toString()).commit();
        } catch (JSONException | RuntimeException ignored) {
            return false;
        }
    }

    Entry save(Bitmap image, long createdAtMs, long elapsedMs, String modelId,
               AtomicBoolean cancelled) throws IOException {
        checkCancelled(cancelled);
        if (image == null || image.isRecycled()) throw new IOException("没有可保存的结果");
        String name = "FACE_RE_" + createdAtMs + "_" + UUID.randomUUID() + ".png";
        Uri uri = Build.VERSION.SDK_INT >= 29
                ? saveScoped(image, name, createdAtMs, cancelled)
                : saveLegacy(image, name, createdAtMs, cancelled);
        // Publication is the commit point. A late cancel must not orphan a saved image
        // or make the UI offer a second save of the same completed result.
        return new Entry(uri, createdAtMs, elapsedMs, modelId, image.getWidth(), image.getHeight());
    }

    private Uri saveScoped(Bitmap image, String name, long time, AtomicBoolean cancelled)
            throws IOException {
        ContentResolver resolver = context.getContentResolver();
        ContentValues values = imageValues(image, name, time);
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/LocalVideoLab");
        values.put(MediaStore.Images.Media.IS_PENDING, 1);
        Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        if (uri == null) throw new IOException("无法创建相册记录，请检查剩余空间");
        boolean published = false;
        try {
            try (OutputStream raw = resolver.openOutputStream(uri, "w")) {
                if (raw == null) throw new IOException("无法写入相册");
                writePng(image, raw, cancelled);
            }
            checkCancelled(cancelled);
            ContentValues ready = new ContentValues();
            ready.put(MediaStore.Images.Media.IS_PENDING, 0);
            if (resolver.update(uri, ready, null, null) != 1) {
                throw new IOException("相册发布失败，结果尚未保存");
            }
            published = true;
            return uri;
        } finally {
            if (!published) deleteUnpublished(resolver, uri);
        }
    }

    @SuppressWarnings("deprecation")
    private Uri saveLegacy(Bitmap image, String name, long time, AtomicBoolean cancelled)
            throws IOException {
        // API 28 needs manifest + runtime READ/WRITE_EXTERNAL_STORAGE (maxSdkVersion=28).
        // A fully written file is indexed only after rename; sharing uses MediaStore's
        // content URI, so no FileProvider or file:// exposure is necessary.
        File directory = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES), "LocalVideoLab");
        if (!directory.isDirectory() && !directory.mkdirs()) {
            throw new IOException("无法创建 Pictures/LocalVideoLab，请检查存储权限和剩余空间");
        }
        File staged = File.createTempFile(".face-re-", ".tmp", directory);
        File destination = new File(directory, name);
        Uri uri = null;
        boolean published = false;
        try {
            try (FileOutputStream output = new FileOutputStream(staged)) {
                writePng(image, output, cancelled);
                output.getFD().sync();
            }
            checkCancelled(cancelled);
            if (!staged.renameTo(destination)) throw new IOException("无法完成相册文件写入");
            checkCancelled(cancelled);
            ContentValues values = imageValues(image, name, time);
            values.put(MediaStore.Images.Media.DATA, destination.getAbsolutePath());
            values.put(MediaStore.Images.Media.SIZE, destination.length());
            uri = context.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            if (uri == null) throw new IOException("无法建立相册记录");
            published = true;
            return uri;
        } finally {
            CleanupFailure cleanup = null;
            if (!published) {
                try {
                    if (uri != null) deleteUnpublished(context.getContentResolver(), uri);
                } catch (CleanupFailure failure) { cleanup = failure; }
                if (destination.exists() && !destination.delete()) {
                    cleanup = new CleanupFailure("未完成的相册文件清理失败，请在系统相册中检查", cleanup);
                }
            }
            if (staged.exists() && !staged.delete()) {
                cleanup = new CleanupFailure("临时相册文件清理失败，请检查 Pictures/LocalVideoLab", cleanup);
            }
            if (cleanup != null) throw cleanup;
        }
    }

    private static ContentValues imageValues(Bitmap image, String name, long time) {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, name);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
        values.put(MediaStore.Images.Media.DATE_TAKEN, time);
        values.put(MediaStore.Images.Media.WIDTH, image.getWidth());
        values.put(MediaStore.Images.Media.HEIGHT, image.getHeight());
        return values;
    }

    private static void writePng(Bitmap image, OutputStream output, AtomicBoolean cancelled)
            throws IOException {
        OutputStream checked = new FilterOutputStream(output) {
            @Override public void write(int value) throws IOException {
                checkCancelled(cancelled);
                out.write(value);
            }
            @Override public void write(byte[] bytes, int offset, int count) throws IOException {
                checkCancelled(cancelled);
                out.write(bytes, offset, count);
            }
        };
        checkCancelled(cancelled);
        if (!image.compress(Bitmap.CompressFormat.PNG, 100, checked)) {
            checkCancelled(cancelled);
            throw new IOException("图片编码失败，结果尚未保存");
        }
        checked.flush();
        checkCancelled(cancelled);
    }

    private static void deleteUnpublished(ContentResolver resolver, Uri uri) throws CleanupFailure {
        try {
            if (resolver.delete(uri, null, null) < 1) {
                throw new CleanupFailure("未完成的相册记录清理失败，请在系统相册中检查");
            }
        } catch (RuntimeException error) {
            throw new CleanupFailure("未完成的相册记录清理失败，请在系统相册中检查", error);
        }
    }

    static void checkCancelled(AtomicBoolean cancelled) {
        if (cancelled.get() || Thread.currentThread().isInterrupted()) {
            throw new CancellationException("任务已取消");
        }
    }
}
