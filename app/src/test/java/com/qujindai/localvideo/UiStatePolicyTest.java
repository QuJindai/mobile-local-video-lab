package com.qujindai.localvideo;

import org.junit.Test;

import static org.junit.Assert.*;

public class UiStatePolicyTest {
    @Test public void emptyStateRequiresPrimaryImage() {
        UiStatePolicy.State state = UiStatePolicy.resolve(false, false, UiStatePolicy.Phase.IDLE, false);
        assertEquals("选择主图", state.primaryLabel);
        assertFalse(state.generateEnabled);
        assertFalse(state.clearSecondaryVisible);
        assertFalse(state.openEnabled);
        assertFalse(state.shareEnabled);
    }

    @Test public void readySingleImageStateUsesProductLabels() {
        UiStatePolicy.State state = UiStatePolicy.resolve(true, false, UiStatePolicy.Phase.READY, false);
        assertEquals("更换主图", state.primaryLabel);
        assertEquals("开始生成", state.generateLabel);
        assertTrue(state.generateEnabled);
        assertFalse(state.clearSecondaryVisible);
    }

    @Test public void secondaryImageMakesClearActionVisible() {
        UiStatePolicy.State state = UiStatePolicy.resolve(true, true, UiStatePolicy.Phase.READY, false);
        assertTrue(state.clearSecondaryVisible);
        assertEquals("双图插值", state.modeLabel);
    }

    @Test public void successEnablesResultActions() {
        UiStatePolicy.State state = UiStatePolicy.resolve(true, false, UiStatePolicy.Phase.SUCCESS, true);
        assertTrue(state.openEnabled);
        assertTrue(state.shareEnabled);
        assertEquals("再次生成", state.generateLabel);
    }
}
