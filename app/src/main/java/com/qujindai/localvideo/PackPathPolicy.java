package com.qujindai.localvideo;

import java.util.ArrayDeque;

public final class PackPathPolicy {
    private PackPathPolicy() {}

    public static boolean isSafe(String raw) {
        if (raw == null) return false;
        String path = raw.trim().replace('\\', '/');
        if (path.isEmpty() || path.equals(".") || path.startsWith("/") || path.matches("^[A-Za-z]:/.*")) {
            return false;
        }
        String[] parts = path.split("/");
        ArrayDeque<String> normalized = new ArrayDeque<>();
        for (String part : parts) {
            if (part.isEmpty() || part.equals(".")) continue;
            if (part.equals("..")) {
                if (normalized.isEmpty()) return false;
                normalized.removeLast();
                continue;
            }
            if (part.indexOf('\0') >= 0) return false;
            normalized.addLast(part);
        }
        if (normalized.isEmpty()) return false;
        // Reject any path that uses parent traversal even if it normalizes back inside the root.
        for (String part : parts) {
            if (part.equals("..")) return false;
        }
        return true;
    }
}
