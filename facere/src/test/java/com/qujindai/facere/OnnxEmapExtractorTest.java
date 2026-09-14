package com.qujindai.facere;

import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.*;

public class OnnxEmapExtractorTest {
    @Test
    public void extractsLast512x512FloatInitializer() throws Exception {
        byte[] raw = new byte[512 * 512 * 4];
        ByteBuffer floats = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 512 * 512; i++) floats.putFloat(i == 0 ? 0.25f : (i == 513 ? 0.5f : 0.0f));

        byte[] tensor = tensorProto(raw);
        ByteArrayOutputStream graph = new ByteArrayOutputStream();
        fieldBytes(graph, 5, tensor);
        ByteArrayOutputStream model = new ByteArrayOutputStream();
        fieldBytes(model, 7, graph.toByteArray());

        File input = File.createTempFile("inswapper-test", ".onnx");
        File output = File.createTempFile("emap-test", ".bin");
        try (FileOutputStream out = new FileOutputStream(input)) {
            out.write(model.toByteArray());
        }

        OnnxEmapExtractor.extract(input, output);
        assertEquals(8L + raw.length, output.length());
        try (DataInputStream in = new DataInputStream(new FileInputStream(output))) {
            byte[] header = new byte[8];
            in.readFully(header);
            ByteBuffer bb = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
            assertEquals(512, bb.getInt());
            assertEquals(512, bb.getInt());
            byte[] extracted = new byte[raw.length];
            in.readFully(extracted);
            assertArrayEquals(raw, extracted);
        } finally {
            input.delete();
            output.delete();
        }
    }

    private static byte[] tensorProto(byte[] raw) throws Exception {
        ByteArrayOutputStream tensor = new ByteArrayOutputStream();
        fieldVarint(tensor, 1, 512);
        fieldVarint(tensor, 1, 512);
        fieldVarint(tensor, 2, 1); // FLOAT
        fieldBytes(tensor, 9, raw);
        return tensor.toByteArray();
    }

    private static void fieldVarint(ByteArrayOutputStream out, int field, long value) {
        writeVarint(out, ((long) field << 3));
        writeVarint(out, value);
    }

    private static void fieldBytes(ByteArrayOutputStream out, int field, byte[] value) {
        writeVarint(out, ((long) field << 3) | 2L);
        writeVarint(out, value.length);
        out.write(value, 0, value.length);
    }

    private static void writeVarint(ByteArrayOutputStream out, long value) {
        while ((value & ~0x7fL) != 0L) {
            out.write((int) ((value & 0x7fL) | 0x80L));
            value >>>= 7;
        }
        out.write((int) value);
    }
}
