package com.qujindai.localvideo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class RecentResults {
    private RecentResults() {}

    public static List<String> add(List<String> existing, String value, int limit) {
        if (limit < 1) return Collections.emptyList();
        ArrayList<String> out = new ArrayList<>();
        if (value != null && !value.isEmpty()) out.add(value);
        if (existing != null) {
            for (String item : existing) {
                if (item == null || item.isEmpty() || item.equals(value)) continue;
                out.add(item);
                if (out.size() == limit) break;
            }
        }
        if (out.size() > limit) return new ArrayList<>(out.subList(0, limit));
        return out;
    }

    public static String encode(List<String> values) {
        if (values == null || values.isEmpty()) return "";
        StringBuilder builder = new StringBuilder();
        for (String value : values) {
            if (value == null || value.isEmpty()) continue;
            if (builder.length() > 0) builder.append('\n');
            builder.append(value.replace("\n", ""));
        }
        return builder.toString();
    }

    public static List<String> decode(String encoded) {
        if (encoded == null || encoded.isEmpty()) return Collections.emptyList();
        ArrayList<String> out = new ArrayList<>();
        for (String line : encoded.split("\\n")) {
            if (!line.isEmpty()) out.add(line);
        }
        return out;
    }
}
