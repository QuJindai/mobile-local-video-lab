package com.qujindai.localvideo;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.media.MediaMuxer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

public final class Mp4Encoder {
    public interface ProgressListener {
        void onEncoded(int encoded, int total);
    }

    private Mp4Encoder() {}

    public static void encode(
            List<File> frames,
            File output,
            int width,
            int height,
            int fps,
            ProgressListener progress) throws IOException {
        if (frames.isEmpty()) throw new IOException("no frames to encode");
        if ((width & 1) != 0 || (height & 1) != 0) throw new IOException("encoder needs even dimensions");

        MediaCodec codec = null;
        MediaMuxer muxer = null;
        boolean muxerStarted = false;
        int trackIndex = -1;
        try {
            MediaFormat format = MediaFormat.createVideoFormat("video/avc", width, height);
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            format.setInteger(MediaFormat.KEY_BIT_RATE, Math.max(1_500_000, width * height * 5));
            format.setInteger(MediaFormat.KEY_FRAME_RATE, fps);
            format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);

            codec = MediaCodec.createEncoderByType("video/avc");
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
            codec.start();
            muxer = new MediaMuxer(output.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);

            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            int nextFrame = 0;
            boolean eosQueued = false;
            boolean eosSeen = false;

            while (!eosSeen) {
                if (nextFrame < frames.size()) {
                    int inputIndex = codec.dequeueInputBuffer(10_000);
                    if (inputIndex >= 0) {
                        ByteBuffer input = codec.getInputBuffer(inputIndex);
                        if (input == null) throw new IOException("encoder input buffer unavailable");
                        Bitmap bitmap = BitmapFactory.decodeFile(frames.get(nextFrame).getAbsolutePath());
                        if (bitmap == null) throw new IOException("cannot decode frame " + frames.get(nextFrame));
                        Bitmap scaled = bitmap;
                        if (bitmap.getWidth() != width || bitmap.getHeight() != height) {
                            scaled = Bitmap.createScaledBitmap(bitmap, width, height, true);
                        }
                        byte[] yuv = toI420(scaled);
                        if (scaled != bitmap) scaled.recycle();
                        bitmap.recycle();
                        if (input.capacity() < yuv.length) {
                            throw new IOException("encoder input buffer too small: " + input.capacity());
                        }
                        input.clear();
                        input.put(yuv);
                        long ptsUs = nextFrame * 1_000_000L / fps;
                        codec.queueInputBuffer(inputIndex, 0, yuv.length, ptsUs, 0);
                        nextFrame++;
                        if (progress != null) progress.onEncoded(nextFrame, frames.size());
                    }
                } else if (!eosQueued) {
                    int inputIndex = codec.dequeueInputBuffer(10_000);
                    if (inputIndex >= 0) {
                        long ptsUs = frames.size() * 1_000_000L / fps;
                        codec.queueInputBuffer(inputIndex, 0, 0, ptsUs,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                        eosQueued = true;
                    }
                }

                while (true) {
                    int outputIndex = codec.dequeueOutputBuffer(info, eosQueued ? 10_000 : 0);
                    if (outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER) {
                        break;
                    }
                    if (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        if (muxerStarted) throw new IOException("encoder format changed twice");
                        trackIndex = muxer.addTrack(codec.getOutputFormat());
                        muxer.start();
                        muxerStarted = true;
                        continue;
                    }
                    if (outputIndex >= 0) {
                        ByteBuffer encoded = codec.getOutputBuffer(outputIndex);
                        if (encoded == null) throw new IOException("encoder output buffer unavailable");
                        if ((info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                            info.size = 0;
                        }
                        if (info.size > 0) {
                            if (!muxerStarted) throw new IOException("muxer not started");
                            encoded.position(info.offset);
                            encoded.limit(info.offset + info.size);
                            muxer.writeSampleData(trackIndex, encoded, info);
                        }
                        eosSeen = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                        codec.releaseOutputBuffer(outputIndex, false);
                        if (eosSeen) break;
                    }
                }
            }
        } finally {
            if (codec != null) {
                try { codec.stop(); } catch (Exception ignored) {}
                codec.release();
            }
            if (muxer != null) {
                if (muxerStarted) {
                    try { muxer.stop(); } catch (Exception ignored) {}
                }
                muxer.release();
            }
        }
    }

    static byte[] toI420(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int frameSize = width * height;
        int chromaSize = frameSize / 4;
        byte[] output = new byte[frameSize + chromaSize * 2];
        int[] argb = new int[frameSize];
        bitmap.getPixels(argb, 0, width, 0, 0, width, height);

        int yIndex = 0;
        int uIndex = frameSize;
        int vIndex = frameSize + chromaSize;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int color = argb[y * width + x];
                int r = (color >> 16) & 0xff;
                int g = (color >> 8) & 0xff;
                int b = color & 0xff;
                int yy = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                int uu = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                int vv = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                output[yIndex++] = (byte) clamp(yy);
                if ((y & 1) == 0 && (x & 1) == 0) {
                    output[uIndex++] = (byte) clamp(uu);
                    output[vIndex++] = (byte) clamp(vv);
                }
            }
        }
        return output;
    }

    private static int clamp(int value) {
        return Math.max(0, Math.min(255, value));
    }
}
