package com.qujindai.localvideo;

/** Host tests also run in APK CI, without Android or third-party dependencies. */
public final class MotionStudioTest {
    public static void main(String[] args) {
        for (DepthMotionSpec.Preset preset : DepthMotionSpec.Preset.values()) {
            for (DepthMotionSpec.Strength strength : DepthMotionSpec.Strength.values()) {
                for (int time = 0; time <= 100; time++) {
                    DepthMotionSpec.Pose pose = DepthMotionSpec.at(preset, strength, time / 100f);
                    for (float depth : new float[] {0f, .25f, .5f, .75f, 1f}) {
                        float zoom = pose.zoomScale(depth);
                        float x = pose.shiftX(depth, 720);
                        float y = pose.shiftY(depth, 512);
                        // Inverse-warped corners must stay inside the source, even at strong motion.
                        check(Float.isFinite(zoom) && zoom >= 1f, "finite positive zoom");
                        check(359.5f + (0f - x - 359.5f) / zoom >= -.001f, "left crop");
                        check(359.5f + (719f - x - 359.5f) / zoom <= 719.001f, "right crop");
                        check(255.5f + (0f - y - 255.5f) / zoom >= -.001f, "top crop");
                        check(255.5f + (511f - y - 255.5f) / zoom <= 511.001f, "bottom crop");
                    }
                }
            }
        }
        DepthMotionSpec.Pose left = pose(DepthMotionSpec.Preset.PARALLAX_LEFT, 1f);
        DepthMotionSpec.Pose right = pose(DepthMotionSpec.Preset.PARALLAX_RIGHT, 1f);
        check(left.shiftX(1, 720) < 0 && right.shiftX(1, 720) > 0, "direction");
        check(Math.abs(left.shiftX(1, 720)) > Math.abs(left.shiftX(0, 720)), "near layers move more");
        float arcMid = pose(DepthMotionSpec.Preset.ARC_ORBIT, .5f).shiftX(1, 720);
        float arcEnd = pose(DepthMotionSpec.Preset.ARC_ORBIT, 1f).shiftX(1, 720);
        check(arcMid > 10 && Math.abs(arcEnd) < .001f, "arc reverses after midpoint");
        check(pose(DepthMotionSpec.Preset.CRANE_UP, 1f).shiftY(1, 512) < -10, "crane rises");
        check(pose(DepthMotionSpec.Preset.DOLLY_OUT, 0).zoomScale(1)
                > pose(DepthMotionSpec.Preset.DOLLY_OUT, 1).zoomScale(1), "pullback releases crop");
        check(Math.abs(DepthMotionSpec.at(DepthMotionSpec.Preset.PARALLAX_LEFT,
                DepthMotionSpec.Strength.STRONG, 1).shiftX(1, 720)) > Math.abs(left.shiftX(1, 720)), "strength");
        int[] loop = FrameSequence.indices(5, true);
        int[] expected = {0, 1, 2, 3, 4, 3, 2, 1};
        check(java.util.Arrays.equals(expected, loop), "loop excludes repeated turning frames");
        for (int frames : new int[] {9, 17, 33, 49}) {
            for (boolean looping : new boolean[] {false, true}) {
                int[] sequence = FrameSequence.indices(frames, looping);
                check(sequence.length == FrameSequence.outputCount(frames, looping), "duration agrees with frames");
                check(sequence[0] == 0, "first frame");
                for (int i = 1; i < sequence.length; i++) {
                    check(Math.abs(sequence[i] - sequence[i - 1]) == 1, "continuous output order");
                }
                check(!looping || sequence[sequence.length - 1] == 1, "loop boundary has one frame step");
                new GenerationPlan(720, 512, sequence.length, 8);
            }
        }
        check(FrameSequence.outputCount(33, true) == 64, "eight seconds at eight FPS");
        ResultRecord old = ResultRecord.decode("1000|2125|720|512|17|8|Y29udGVudDovL3ZpZGVvLzE");
        check(old != null && old.uri.equals("content://video/1") && old.recipe.isEmpty(), "legacy history");
        ResultRecord source = new ResultRecord("content://video/2", 1, 8000, 720, 512, 64, 8,
                "Depth 弧线环绕 | 标准 · 往返循环");
        ResultRecord restored = ResultRecord.decode(source.encode());
        check(restored != null && restored.recipe.equals(source.recipe)
                && restored.frames == 64 && restored.durationMs == 8000, "history recipe and loop metadata");
        System.out.println("Motion Studio: trajectories, crop bounds, frame order and duration PASS");
    }

    private static DepthMotionSpec.Pose pose(DepthMotionSpec.Preset preset, float time) {
        return DepthMotionSpec.at(preset, DepthMotionSpec.Strength.NORMAL, time);
    }
    private static void check(boolean value, String message) {
        if (!value) throw new AssertionError(message);
    }
}
