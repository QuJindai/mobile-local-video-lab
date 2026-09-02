package com.qujindai.localvideo;

import java.util.Arrays;
import java.util.List;

public final class RifeCommand {
    private RifeCommand() {}

    public static List<String> build(
            String executable,
            String inputDir,
            String outputDir,
            int frames,
            String modelDir) {
        return Arrays.asList(
                executable,
                "-v",
                "-i", inputDir,
                "-o", outputDir,
                "-n", Integer.toString(frames),
                "-m", modelDir,
                "-j", "1:1:1",
                "-f", "%08d.png");
    }
}
