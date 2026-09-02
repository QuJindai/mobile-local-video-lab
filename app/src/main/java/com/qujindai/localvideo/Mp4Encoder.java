package com.qujindai.localvideo;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.media.Image;
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
        if ((width & 1) != 0 || (height & 1) != 0) {
            throw new IOException("encoder needs even dimensions");
        }

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
            final String codecName = codec.getName();
            muxer = new MediaMuxer(output.getAbsolutePath(),
                    MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);

            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            int nextFrame = 0;
            boolean eosQueued = false;
            boolean eosSeen = false;
            final int rawFrameSize = width * height * 3 / 2;

            while (!eosSeen) {
                if (nextFrame < frames.size()) {
                    int inputIndex = codec.dequeueInputBuffer(10_000);
                    if (inputIndex >= 0) {
                        Bitmap bitmap = BitmapFactory.decodeFile(
                                frames.get(nextFrame).getAbsolutePath());
                        if (bitmap == null) {
                            throw new IOException("cannot decode frame " + frames.get(nextFrame));
                        }
                        Bitmap scaled = bitmap;
                        if (bitmap.getWidth() != width || bitmap.getHeight() != height) {
                            scaled = Bitmap.createScaledBitmap(bitmap, width, height, true);
                        }
                        int[] argb = new int[width * height];
                        scaled.getPixels(argb, 0, width, 0, 0, width, height);
                        Yuv420Frame yuv = Yuv420Frame.fromArgb(argb, width, height);
                        if (scaled != bitmap) scaled.recycle();
                        bitmap.recycle();

                        Image inputImage = codec.getInputImage(inputIndex);
                        if (inputImage == null) {
                            throw new IOException(
                                    "encoder does not expose YUV_420_888 input image: " + codecName);
                        }
                        if (inputImage.getFormat() != ImageFormat.YUV_420_888) {
                            throw new IOException(
                                    "unexpected encoder input image format " + inputImage.getFormat()
                                            + " from " + codecName);
                        }
                        Image.Plane[] planes = inputImage.getPlanes();
                        if (planes == null || planes.length != 3) {
                            throw new IOException(
                                    "encoder YUV input must expose exactly 3 planes: " + codecName);
                        }

                        writePlane(yuv.y, width, height, planes[0]);
                        writePlane(yuv.u, width / 2, height / 2, planes[1]);
                        writePlane(yuv.v, width / 2, height / 2, planes[2]);

                        long ptsUs = nextFrame * 1_000_000L / fps;
                        // Android CTS queues width*height*3/2 after filling getInputImage().
                        codec.queueInputBuffer(inputIndex, 0, rawFrameSize, ptsUs, 0);
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
                        if (encoded == null) {
                            throw new IOException("encoder output buffer unavailable");
                        }
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

    private static void writePlane(byte[] source, int width, int height, Image.Plane plane) {
        ByteBuffer buffer = plane.getBuffer();
        Yuv420PlaneWriter.write(source, width, height, buffer,
                plane.getRowStride(), plane.getPixelStride());
    }
}
