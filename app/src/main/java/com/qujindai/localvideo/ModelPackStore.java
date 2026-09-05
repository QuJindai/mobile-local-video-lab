package com.qujindai.localvideo;

import android.content.Context;
import android.content.SharedPreferences;

import java.io.File;
import java.io.IOException;

public final class ModelPackStore {
    private static final String PREFS = "local_video_modelpacks";
    private static final String KEY_MOBILE_I2V_ROOT = "mobilei2v.root.v1";

    private final Context context;
    private final SharedPreferences preferences;

    public ModelPackStore(Context context) {
        this.context = context.getApplicationContext();
        preferences = this.context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    public void activate(InstalledModelPack pack) {
        if (pack == null || !pack.manifest.isMobileI2V()) {
            throw new IllegalArgumentException("only MobileI2V packs can be activated here");
        }
        preferences.edit().putString(KEY_MOBILE_I2V_ROOT, pack.root.getAbsolutePath()).apply();
    }

    public InstalledModelPack loadMobileI2V() {
        String path = preferences.getString(KEY_MOBILE_I2V_ROOT, "");
        if (path == null || path.isEmpty()) return null;
        try {
            InstalledModelPack pack = ModelPackInstaller.inspect(new File(path));
            if (!pack.manifest.isMobileI2V()) return null;
            return pack;
        } catch (IOException | RuntimeException error) {
            return null;
        }
    }

    public void removeMobileI2V() {
        String path = preferences.getString(KEY_MOBILE_I2V_ROOT, "");
        preferences.edit().remove(KEY_MOBILE_I2V_ROOT).apply();
        if (path != null && !path.isEmpty()) {
            File root = new File(path);
            // Only delete roots that are descendants of the app-private modelpacks directory.
            try {
                File safeRoot = new File(context.getFilesDir(), "modelpacks").getCanonicalFile();
                File candidate = root.getCanonicalFile();
                if (candidate.getPath().startsWith(safeRoot.getPath() + File.separator)) {
                    ModelPackInstaller.deleteRecursively(candidate);
                }
            } catch (IOException ignored) {
                // Preference is already cleared; leave any questionable path untouched.
            }
        }
    }
}
