package com.qujindai.facere;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

public final class FacePackContract {
    public static final String DETECTOR = "det_10g.onnx";
    public static final String RECOGNIZER = "w600k_r50.onnx";
    public static final String SWAPPER = "inswapper_128.onnx";
    public static final String EMAP = "emap.bin";

    private static final Set<String> REQUIRED;
    static {
        LinkedHashSet<String> names = new LinkedHashSet<>();
        names.add(DETECTOR);
        names.add(RECOGNIZER);
        names.add(SWAPPER);
        names.add(EMAP);
        REQUIRED = Collections.unmodifiableSet(names);
    }

    private FacePackContract() {}

    public static Set<String> requiredFiles() {
        return REQUIRED;
    }

    public static boolean hasRequired(Set<String> names) {
        return names != null && names.containsAll(REQUIRED);
    }

    public static boolean isRequired(String name) {
        return REQUIRED.contains(name);
    }

    public static boolean isSafeEntry(String name) {
        if (name == null || name.isEmpty()) return false;
        if (name.startsWith("/") || name.startsWith("\\")) return false;
        if (name.contains("/") || name.contains("\\") || name.contains("..")) return false;
        return !name.equals(".") && !name.equals("..");
    }
}
