package com.qujindai.localvideo;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class RecentResultsTest {
    @Test public void newestResultComesFirstAndDuplicatesMoveToFront() {
        List<String> results = RecentResults.add(Arrays.asList("a", "b", "c"), "b", 5);
        assertEquals(Arrays.asList("b", "a", "c"), results);
    }

    @Test public void trimsToRequestedLimit() {
        List<String> results = RecentResults.add(Arrays.asList("a", "b", "c"), "d", 3);
        assertEquals(Arrays.asList("d", "a", "b"), results);
    }

    @Test public void newlineCodecRoundTrips() {
        List<String> source = Arrays.asList("content://video/3", "content://video/2");
        assertEquals(source, RecentResults.decode(RecentResults.encode(source)));
    }
}
