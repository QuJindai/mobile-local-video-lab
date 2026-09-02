package com.qujindai.localvideo;

import org.junit.Test;

import java.nio.ByteBuffer;

import static org.junit.Assert.assertArrayEquals;

public class Yuv420PlaneWriterTest {
    @Test public void writesRowsWithPaddingAndPixelStride() {
        ByteBuffer buffer = ByteBuffer.allocate(12);
        for (int i = 0; i < buffer.capacity(); i++) buffer.put(i, (byte) 0x55);

        Yuv420PlaneWriter.write(new byte[] {1, 2, 3, 4}, 2, 2, buffer, 6, 2);

        byte[] actual = new byte[12];
        buffer.position(0);
        buffer.get(actual);
        assertArrayEquals(new byte[] {
                1, 0x55, 2, 0x55, 0x55, 0x55,
                3, 0x55, 4, 0x55, 0x55, 0x55
        }, actual);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsTooSmallTargetPlane() {
        Yuv420PlaneWriter.write(new byte[] {1, 2, 3, 4}, 2, 2,
                ByteBuffer.allocate(4), 4, 2);
    }
}
