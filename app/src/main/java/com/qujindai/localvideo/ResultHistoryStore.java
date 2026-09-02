package com.qujindai.localvideo;

import android.content.Context;
import android.content.SharedPreferences;
import android.net.Uri;

import java.util.ArrayList;
import java.util.List;

public final class ResultHistoryStore {
    private static final String PREFS = "local_video_results";
    private static final String KEY_RECENT = "recent_uris_v2";
    private static final int LIMIT = 5;

    private final SharedPreferences preferences;

    public ResultHistoryStore(Context context) {
        preferences = context.getApplicationContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    public void record(Uri uri) {
        if (uri == null) return;
        List<String> current = RecentResults.decode(preferences.getString(KEY_RECENT, ""));
        List<String> updated = RecentResults.add(current, uri.toString(), LIMIT);
        preferences.edit().putString(KEY_RECENT, RecentResults.encode(updated)).apply();
    }

    public List<Uri> load() {
        List<String> stored = RecentResults.decode(preferences.getString(KEY_RECENT, ""));
        ArrayList<Uri> result = new ArrayList<>();
        for (String value : stored) {
            try {
                result.add(Uri.parse(value));
            } catch (RuntimeException ignored) {
                // Ignore stale malformed entries instead of breaking the result screen.
            }
        }
        return result;
    }

    public Uri latest() {
        List<Uri> results = load();
        return results.isEmpty() ? null : results.get(0);
    }
}
