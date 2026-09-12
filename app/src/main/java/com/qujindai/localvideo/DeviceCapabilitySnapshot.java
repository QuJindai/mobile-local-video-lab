package com.qujindai.localvideo;

import android.app.ActivityManager;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.StatFs;
import android.system.Os;
import android.system.OsConstants;

import java.util.Locale;

public final class DeviceCapabilitySnapshot {
    public final int api;
    public final String abi;
    public final String soc;
    public final long totalRamMb;
    public final long freePrivateStorageMb;
    public final long javaMaxHeapMb;
    public final long pageSizeBytes;
    public final boolean vulkan;

    private DeviceCapabilitySnapshot(
            int api, String abi, String soc, long totalRamMb, long freePrivateStorageMb,
            long javaMaxHeapMb, long pageSizeBytes, boolean vulkan) {
        this.api = api;
        this.abi = abi;
        this.soc = soc;
        this.totalRamMb = totalRamMb;
        this.freePrivateStorageMb = freePrivateStorageMb;
        this.javaMaxHeapMb = javaMaxHeapMb;
        this.pageSizeBytes = pageSizeBytes;
        this.vulkan = vulkan;
    }

    public static DeviceCapabilitySnapshot capture(Context context) {
        ActivityManager manager = context.getSystemService(ActivityManager.class);
        ActivityManager.MemoryInfo memory = new ActivityManager.MemoryInfo();
        if (manager != null) manager.getMemoryInfo(memory);
        long ramMb = memory.totalMem > 0 ? memory.totalMem / 1048576L : 0;

        StatFs stat = new StatFs(context.getFilesDir().getAbsolutePath());
        long freeMb = stat.getAvailableBytes() / 1048576L;
        long heapMb = Runtime.getRuntime().maxMemory() / 1048576L;
        long pageSize = 0;
        try {
            pageSize = Os.sysconf(OsConstants._SC_PAGESIZE);
        } catch (RuntimeException ignored) {
            // Keep zero if the platform refuses sysconf.
        }
        String abi = Build.SUPPORTED_ABIS.length == 0 ? "unknown" : Build.SUPPORTED_ABIS[0];
        String soc = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S
                ? Build.SOC_MANUFACTURER + " " + Build.SOC_MODEL
                : Build.HARDWARE;
        PackageManager pm = context.getPackageManager();
        boolean vulkan = pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL)
                || pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION);
        return new DeviceCapabilitySnapshot(
                Build.VERSION.SDK_INT, abi, soc.trim(), ramMb, freeMb, heapMb, pageSize, vulkan);
    }

    public String summary() {
        return String.format(Locale.US,
                "SoC: %s\nABI: %s · Android API %d\nRAM: %.1f GB · Java heap: %d MB\n"
                        + "可用私有存储: %.1f GB\nOS page size: %d KB · Vulkan: %s",
                soc, abi, api, totalRamMb / 1024.0, javaMaxHeapMb,
                freePrivateStorageMb / 1024.0, pageSizeBytes / 1024L,
                vulkan ? "YES" : "NO");
    }
}
