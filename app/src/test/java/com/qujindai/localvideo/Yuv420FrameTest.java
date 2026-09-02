package com.qujindai.localvideo;

import org.junit.Test;

import static org.junit.Assert.*;

public class Yuv420FrameTest {
    @Test public void convertsSolidRedToStableBt601Planes() {
        int[] argb = new int[] {
                0xffff0000, 0xffff0000,
                0xffff0000, 0xffff0000
        };
        Yuv420Frame frame = Yuv420Frame.fromArgb(argb, 2, 2);
        assertEquals(4, frame.y.length);
        assertEquals(1, frame.u.length);
        assertEquals(1, frame.v.length);
        for (byte y : frame.y) assertEquals(82, y & 0xff);
        assertEquals(90, frame.u[0] & 0xff);
        assertEquals(240, frame.v[0] & 0xff);
    }

    @Test public void rejectsOddDimensions() {
        try {
            Yuv420Frame.fromArgb(new int[6], 3, 2);
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("even"));
        }
    }

    @Test public void rejectsWrongPixelCount() {
        try {
            Yuv420Frame.fromArgb(new int[3], 2, 2);
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("pixel"));
        }
    }
}
