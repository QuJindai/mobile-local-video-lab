package com.qujindai.facere;

import java.io.BufferedOutputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Extracts the 512x512 float EMAP initializer from an INSwapper ONNX file without loading it all into RAM. */
public final class OnnxEmapExtractor {
    private static final int SIZE = 512;
    private static final long FLOAT_BYTES = (long) SIZE * SIZE * 4L;

    private OnnxEmapExtractor() {}

    public static void extract(File onnx, File output) throws IOException {
        if (onnx == null || !onnx.isFile() || onnx.length() <= 0L) {
            throw new IOException("inswapper model missing");
        }
        Candidate candidate = null;
        try (RandomAccessFile in = new RandomAccessFile(onnx, "r")) {
            long modelEnd = in.length();
            while (in.getFilePointer() < modelEnd) {
                long tag = readVarint(in, modelEnd);
                int field = (int) (tag >>> 3);
                int wire = (int) (tag & 7L);
                if (field == 7 && wire == 2) { // ModelProto.graph
                    long length = readVarint(in, modelEnd);
                    long start = in.getFilePointer();
                    long end = checkedEnd(start, length, modelEnd);
                    Candidate found = parseGraph(in, end);
                    if (found != null) candidate = found;
                    in.seek(end);
                } else {
                    skipValue(in, wire, modelEnd);
                }
            }

            if (candidate == null) {
                throw new IOException("512x512 float EMAP initializer not found");
            }
            File parent = output.getParentFile();
            if (parent != null && !parent.exists() && !parent.mkdirs()) {
                throw new IOException("cannot create emap output directory");
            }
            try (BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(output))) {
                ByteBuffer header = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
                header.putInt(SIZE).putInt(SIZE);
                out.write(header.array());
                in.seek(candidate.dataOffset);
                byte[] buffer = new byte[1024 * 1024];
                long remaining = FLOAT_BYTES;
                while (remaining > 0L) {
                    int count = (int) Math.min(buffer.length, remaining);
                    int read = in.read(buffer, 0, count);
                    if (read < 0) throw new EOFException("truncated EMAP raw_data");
                    out.write(buffer, 0, read);
                    remaining -= read;
                }
            }
        }
        if (output.length() != 8L + FLOAT_BYTES) {
            throw new IOException("emap output length mismatch");
        }
    }

    private static Candidate parseGraph(RandomAccessFile in, long end) throws IOException {
        Candidate result = null;
        while (in.getFilePointer() < end) {
            long tag = readVarint(in, end);
            int field = (int) (tag >>> 3);
            int wire = (int) (tag & 7L);
            if (field == 5 && wire == 2) { // GraphProto.initializer
                long length = readVarint(in, end);
                long start = in.getFilePointer();
                long tensorEnd = checkedEnd(start, length, end);
                Candidate candidate = parseTensor(in, tensorEnd);
                if (candidate != null) result = candidate; // match Python graph.initializer[-1] semantics
                in.seek(tensorEnd);
            } else {
                skipValue(in, wire, end);
            }
        }
        return result;
    }

    private static Candidate parseTensor(RandomAccessFile in, long end) throws IOException {
        long[] dims = new long[4];
        int dimCount = 0;
        int dataType = 0;
        long rawOffset = -1L;
        long rawLength = -1L;

        while (in.getFilePointer() < end) {
            long tag = readVarint(in, end);
            int field = (int) (tag >>> 3);
            int wire = (int) (tag & 7L);
            if (field == 1 && wire == 0) { // dims, unpacked
                long value = readVarint(in, end);
                if (dimCount < dims.length) dims[dimCount++] = value;
            } else if (field == 1 && wire == 2) { // dims, packed
                long length = readVarint(in, end);
                long packedEnd = checkedEnd(in.getFilePointer(), length, end);
                while (in.getFilePointer() < packedEnd) {
                    long value = readVarint(in, packedEnd);
                    if (dimCount < dims.length) dims[dimCount++] = value;
                }
                in.seek(packedEnd);
            } else if (field == 2 && wire == 0) { // data_type
                dataType = (int) readVarint(in, end);
            } else if (field == 9 && wire == 2) { // raw_data
                long length = readVarint(in, end);
                long start = in.getFilePointer();
                long rawEnd = checkedEnd(start, length, end);
                rawOffset = start;
                rawLength = length;
                in.seek(rawEnd);
            } else {
                skipValue(in, wire, end);
            }
        }

        if (dimCount == 2 && dims[0] == SIZE && dims[1] == SIZE
                && dataType == 1 && rawOffset >= 0L && rawLength == FLOAT_BYTES) {
            return new Candidate(rawOffset);
        }
        return null;
    }

    private static long readVarint(RandomAccessFile in, long end) throws IOException {
        long value = 0L;
        int shift = 0;
        while (shift < 64) {
            if (in.getFilePointer() >= end) throw new EOFException("truncated protobuf varint");
            int b = in.read();
            if (b < 0) throw new EOFException("truncated protobuf varint");
            value |= (long) (b & 0x7f) << shift;
            if ((b & 0x80) == 0) return value;
            shift += 7;
        }
        throw new IOException("invalid protobuf varint");
    }

    private static void skipValue(RandomAccessFile in, int wire, long end) throws IOException {
        switch (wire) {
            case 0:
                readVarint(in, end);
                return;
            case 1:
                seekForward(in, 8L, end);
                return;
            case 2:
                long length = readVarint(in, end);
                seekForward(in, length, end);
                return;
            case 5:
                seekForward(in, 4L, end);
                return;
            default:
                throw new IOException("unsupported protobuf wire type: " + wire);
        }
    }

    private static void seekForward(RandomAccessFile in, long bytes, long end) throws IOException {
        long target = checkedEnd(in.getFilePointer(), bytes, end);
        in.seek(target);
    }

    private static long checkedEnd(long start, long length, long parentEnd) throws IOException {
        if (length < 0L || start > parentEnd - length) throw new IOException("invalid protobuf length");
        return start + length;
    }

    private static final class Candidate {
        final long dataOffset;
        Candidate(long dataOffset) { this.dataOffset = dataOffset; }
    }
}
