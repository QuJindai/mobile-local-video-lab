package com.qujindai.localvideo;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class RifeCommandTest {
    @Test
    public void buildsExactNativeArguments() {
        List<String> command = RifeCommand.build(
                "/native/librife.so",
                "/work/input",
                "/work/frames",
                17,
                "/models/rife-v4.6");

        assertEquals(Arrays.asList(
                "/native/librife.so",
                "-v",
                "-i", "/work/input",
                "-o", "/work/frames",
                "-n", "17",
                "-m", "/models/rife-v4.6",
                "-j", "1:1:1",
                "-f", "%08d.png"), command);
    }
}
