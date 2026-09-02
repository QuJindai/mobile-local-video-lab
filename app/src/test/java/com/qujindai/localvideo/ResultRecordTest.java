package com.qujindai.localvideo;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class ResultRecordTest {
    @Test
    public void roundTripPreservesMetadataAndUri() {
        ResultRecord source = new ResultRecord(
                "content://media/external/video/media/123?x=1&y=2",
                1788350000123L,
                2125L,
                720,
                720,
                17,
                8);

        ResultRecord decoded = ResultRecord.decode(source.encode());
        assertNotNull(decoded);
        assertEquals(source.uri, decoded.uri);
        assertEquals(source.createdAtMs, decoded.createdAtMs);
        assertEquals(source.durationMs, decoded.durationMs);
        assertEquals(source.width, decoded.width);
        assertEquals(source.height, decoded.height);
        assertEquals(source.frames, decoded.frames);
        assertEquals(source.fps, decoded.fps);
    }

    @Test
    public void legacyUriBecomesUsableRecord() {
        ResultRecord record = ResultRecord.fromLegacyUri("content://media/external/video/media/456");
        assertEquals("content://media/external/video/media/456", record.uri);
        assertEquals(0L, record.createdAtMs);
        assertEquals(0L, record.durationMs);
    }
}
