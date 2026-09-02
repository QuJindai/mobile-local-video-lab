package com.qujindai.localvideo;

import android.content.Context;
import android.content.SharedPreferences;
import android.net.Uri;

import java.util.ArrayList;
import java.util.List;

public final class ResultHistoryStore {
    private static final String PREFS = "local_video_results";
    private static final String KEY_RECENT_V2 = "recent_uris_v2";
    private static final String KEY_RECENT_V3 = "recent_records_v3";
    private static final int LIMIT = 5;

    private final SharedPreferences preferences;

    public ResultHistoryStore(Context context) {
        preferences = context.getApplicationContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    public void record(ResultRecord record) {
        if (record == null) return;
        List<String> current = RecentResults.decode(preferences.getString(KEY_RECENT_V3, ""));
        List<String> updated = RecentResults.add(current, record.encode(), LIMIT);
        preferences.edit().putString(KEY_RECENT_V3, RecentResults.encode(updated)).apply();
    }

    public void record(Uri uri) {
        if (uri != null) record(ResultRecord.fromLegacyUri(uri.toString()));
    }

    public List<ResultRecord> loadRecords() {
        List<String> stored = RecentResults.decode(preferences.getString(KEY_RECENT_V3, ""));
        ArrayList<ResultRecord> result = new ArrayList<>();
        for (String value : stored) {
            ResultRecord record = ResultRecord.decode(value);
            if (record != null) result.add(record);
        }
        if (!result.isEmpty()) return result;

        // One-time compatibility path for V0.2 URI-only history.
        List<String> legacy = RecentResults.decode(preferences.getString(KEY_RECENT_V2, ""));
        ArrayList<String> migrated = new ArrayList<>();
        for (String value : legacy) {
            try {
                ResultRecord record = ResultRecord.fromLegacyUri(value);
                result.add(record);
                migrated.add(record.encode());
            } catch (RuntimeException ignored) {
                // Drop stale malformed legacy rows rather than breaking the result screen.
            }
            if (result.size() == LIMIT) break;
        }
        if (!migrated.isEmpty()) {
            preferences.edit().putString(KEY_RECENT_V3, RecentResults.encode(migrated)).apply();
        }
        return result;
    }

    public List<Uri> load() {
        ArrayList<Uri> uris = new ArrayList<>();
        for (ResultRecord record : loadRecords()) {
            try {
                uris.add(Uri.parse(record.uri));
            } catch (RuntimeException ignored) {
                // Ignore stale malformed entries.
            }
        }
        return uris;
    }

    public ResultRecord latestRecord() {
        List<ResultRecord> results = loadRecords();
        return results.isEmpty() ? null : results.get(0);
    }

    public Uri latest() {
        ResultRecord record = latestRecord();
        return record == null ? null : Uri.parse(record.uri);
    }
}
