package com.qujindai.localvideo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.SystemClock;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/** SCRFD -> ArcFace identity -> INSwapper -> inverse warp; CPU is explicit. */
public final class FaceSwapEngine {
    public interface Progress {void update(int percent,String message);}
    public static final class Result {
        public final Bitmap image;
        public final long elapsedMs;
        public final String modelId;
        Result(Bitmap image,long elapsedMs,String modelId){this.image=image;this.elapsedMs=elapsedMs;this.modelId=modelId;}
    }
    private static final ReentrantLock RUN_LOCK=new ReentrantLock();
    private static final int DETECTOR_SIZE=640, MAX_EDGE=1600;
    private FaceSwapEngine(){}

    public static Result generate(Context context,Uri sourceUri,Uri targetUri,Progress progress,AtomicBoolean cancelled) throws Exception {
        if(sourceUri==null||targetUri==null)throw new IOException("请选择参考人脸和目标照片");
        if(!RUN_LOCK.tryLock())throw new IOException("已有换脸任务正在运行");
        Bitmap source=null,target=null;
        long start=SystemClock.elapsedRealtime();
        try {
            Context app=context.getApplicationContext();
            FaceModelStore store=new FaceModelStore(app);
            store.verifyForGeneration(progress,cancelled);
            FaceModelStore.checkCancelled(cancelled);
            progress.update(12,"读取参考人脸和目标照片");
            source=load(app,sourceUri);target=load(app,targetUri);
            int[] sourcePixels=pixels(source),targetPixels=pixels(target);
            int sw=source.getWidth(),sh=source.getHeight(),tw=target.getWidth(),th=target.getHeight();
            source.recycle();source=null;target.recycle();target=null;
            OrtEnvironment env=OrtEnvironment.getEnvironment();
            FaceSwapMath.Detection sourceFace,targetFace;
            try(OrtSession.SessionOptions options=options();OrtSession detector=env.createSession(store.file("scrfd_2.5g.onnx").getAbsolutePath(),options)) {
                checkIo(detector,store.io("detector"));
                progress.update(20,"定位参考人脸");
                sourceFace=singleFace(detect(env,detector,sourcePixels,sw,sh,cancelled),"参考照");
                FaceModelStore.checkCancelled(cancelled);
                progress.update(30,"定位目标人脸");
                targetFace=singleFace(detect(env,detector,targetPixels,tw,th,cancelled),"目标照");
            }
            FaceModelStore.checkCancelled(cancelled);
            double[] sourceTo112=FaceSwapMath.fit(sourceFace.landmarks,112);
            double[] targetTo128=FaceSwapMath.fit(targetFace.landmarks,128);
            float[] sourceRgb=FaceSwapPixels.rgbNchw(FaceSwapPixels.warp(sourcePixels,sw,sh,sourceTo112,112),127.5f,127.5f);
            float[] targetRgb=FaceSwapPixels.rgbNchw(FaceSwapPixels.warp(targetPixels,tw,th,targetTo128,128),0,255);
            float[] identity;
            progress.update(42,"提取参考人脸身份");
            try(OrtSession.SessionOptions options=options();OrtSession recognizer=env.createSession(store.file("arcface_w600k_r50.onnx").getAbsolutePath(),options)) {
                checkIo(recognizer,store.io("recognizer"));
                String input=recognizer.getInputNames().iterator().next();
                try(OnnxTensor tensor=OnnxTensor.createTensor(env,FloatBuffer.wrap(sourceRgb),new long[]{1,3,112,112});
                    OrtSession.RunOptions runOptions=new OrtSession.RunOptions();
                    CancelMonitor monitor=new CancelMonitor(cancelled,runOptions);
                    OrtSession.Result result=recognizer.run(Collections.singletonMap(input,tensor),runOptions)) {
                    identity=flatten(result.get(0).getValue(),512);
                }
            }
            FaceModelStore.checkCancelled(cancelled);
            float[] latent=FaceSwapMath.projectIdentity(identity,store.readProjection());
            float[] generated;
            progress.update(58,"正在本机换脸");
            try(OrtSession.SessionOptions options=options();OrtSession swapper=env.createSession(store.file("inswapper_128.onnx").getAbsolutePath(),options)) {
                checkIo(swapper,store.io("swapper"));
                try(OnnxTensor image=OnnxTensor.createTensor(env,FloatBuffer.wrap(targetRgb),new long[]{1,3,128,128});
                    OnnxTensor face=OnnxTensor.createTensor(env,FloatBuffer.wrap(latent),new long[]{1,512});
                    OrtSession.RunOptions runOptions=new OrtSession.RunOptions();
                    CancelMonitor monitor=new CancelMonitor(cancelled,runOptions)) {
                    Map<String,OnnxTensor> inputs=new LinkedHashMap<>();inputs.put("target",image);inputs.put("source",face);
                    try(OrtSession.Result result=swapper.run(inputs,runOptions)) {
                        generated=flatten(result.get(0).getValue(),3*128*128);
                    }
                }
            }
            FaceModelStore.checkCancelled(cancelled);
            progress.update(90,"融合人脸并还原照片构图");
            int[] fake=FaceSwapPixels.fromRgbNchw(generated,128);
            int[] merged=FaceSwapPixels.composite(targetPixels,tw,th,fake,targetTo128);
            FaceModelStore.checkCancelled(cancelled);
            Bitmap output=Bitmap.createBitmap(merged,tw,th,Bitmap.Config.ARGB_8888);
            if(cancelled.get()){output.recycle();throw new CancellationException("已取消");}
            progress.update(100,"换脸完成，可以保存照片");
            return new Result(output,SystemClock.elapsedRealtime()-start,store.modelId());
        } finally {
            if(source!=null&&!source.isRecycled())source.recycle();
            if(target!=null&&!target.isRecycled())target.recycle();
            RUN_LOCK.unlock();
        }
    }

    private static List<FaceSwapMath.Detection> detect(OrtEnvironment env,OrtSession detector,int[] rgb,int width,int height,AtomicBoolean cancelled) throws Exception {
        float scale=Math.min(1f,Math.min(DETECTOR_SIZE/(float)width,DETECTOR_SIZE/(float)height));
        int rw=Math.max(1,Math.round(width*scale)),rh=Math.max(1,Math.round(height*scale));
        // Independent x/y factors preserve the exact integer resize dimensions.
        Bitmap src=Bitmap.createBitmap(rgb,width,height,Bitmap.Config.ARGB_8888),small=null;
        int[] padded=new int[DETECTOR_SIZE*DETECTOR_SIZE];
        java.util.Arrays.fill(padded,0xff000000);
        try {
            small=Bitmap.createScaledBitmap(src,rw,rh,true);
            small.getPixels(padded,0,DETECTOR_SIZE,0,0,rw,rh);
        } finally {if(small!=null&&small!=src)small.recycle();src.recycle();}
        float[] input=FaceSwapPixels.rgbNchw(padded,127.5f,128f);
        String name=detector.getInputNames().iterator().next();
        float[][] outputs=new float[9][];
        try(OnnxTensor tensor=OnnxTensor.createTensor(env,FloatBuffer.wrap(input),new long[]{1,3,DETECTOR_SIZE,DETECTOR_SIZE});
            OrtSession.RunOptions runOptions=new OrtSession.RunOptions();
            CancelMonitor monitor=new CancelMonitor(cancelled,runOptions);
            OrtSession.Result result=detector.run(Collections.singletonMap(name,tensor),runOptions)) {
            if(result.size()!=9)throw new IOException("检测模型输出不完整");
            for(int i=0;i<9;i++) {
                int stride=new int[]{8,16,32}[i%3];
                int components=i<3?1:(i<6?4:10);
                int expected=2*(DETECTOR_SIZE/stride)*(DETECTOR_SIZE/stride)*components;
                outputs[i]=flatten(result.get(i).getValue(),expected);
            }
        }
        return FaceSwapMath.decodeScrfd(outputs,DETECTOR_SIZE,width,height,rw,rh,.6f);
    }

    private static FaceSwapMath.Detection singleFace(List<FaceSwapMath.Detection> faces,String label) throws IOException {
        if(faces.isEmpty())throw new IOException(label+"未检测到清晰人脸，请换一张光线充足的正面或轻侧面照片");
        if(faces.size()!=1)throw new IOException(label+"检测到多张人脸，请先裁剪到一人后再试");
        FaceSwapMath.Detection face=faces.get(0);
        double eyeDistance=Math.hypot(face.landmarks[0][0]-face.landmarks[1][0],face.landmarks[0][1]-face.landmarks[1][1]);
        if(!Double.isFinite(eyeDistance)||eyeDistance<12 || face.x2-face.x1<40 || face.y2-face.y1<40)
            throw new IOException(label+"人脸过小，请使用更清晰的人脸照片");
        return face;
    }

    private static OrtSession.SessionOptions options() throws Exception {
        OrtSession.SessionOptions options=new OrtSession.SessionOptions();
        try {
            options.setIntraOpNumThreads(Math.max(1,Math.min(4,Runtime.getRuntime().availableProcessors())));
            options.setInterOpNumThreads(1);
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
            return options;
        }catch(Exception error){options.close();throw error;}
    }

    private static Bitmap load(Context context,Uri uri) throws IOException {
        Bitmap bitmap=ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.getContentResolver(),uri),(decoder,info,src)->{
            int w=info.getSize().getWidth(),h=info.getSize().getHeight();
            double scale=Math.min(1d,MAX_EDGE/(double)Math.max(w,h));
            decoder.setTargetSize(Math.max(1,(int)Math.round(w*scale)),Math.max(1,(int)Math.round(h*scale)));
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
        });
        if(bitmap.getConfig()!=Bitmap.Config.ARGB_8888) {
            Bitmap converted=bitmap.copy(Bitmap.Config.ARGB_8888,false);bitmap.recycle();
            if(converted==null)throw new IOException("无法读取照片像素");return converted;
        }
        return bitmap;
    }
    private static int[] pixels(Bitmap bitmap){int w=bitmap.getWidth(),h=bitmap.getHeight();int[] p=new int[w*h];bitmap.getPixels(p,0,w,0,0,w,h);return p;}

    private static void checkIo(OrtSession session,JSONObject spec) throws Exception {
        checkNodes(session.getInputInfo(),spec.getJSONArray("inputs"));
        checkNodes(session.getOutputInfo(),spec.getJSONArray("outputs"));
    }
    private static void checkNodes(Map<String,NodeInfo> nodes,JSONArray expected) throws Exception {
        if(nodes.size()!=expected.length())throw new IOException("换脸模型输入输出数量不匹配");
        List<String> names=new ArrayList<>(nodes.keySet());
        for(int i=0;i<expected.length();i++) {
            JSONObject row=expected.getJSONObject(i);String name=row.getString("name");
            if(!name.equals(names.get(i))||!(nodes.get(name).getInfo() instanceof TensorInfo))throw new IOException("换脸模型张量名称不匹配");
            TensorInfo tensor=(TensorInfo)nodes.get(name).getInfo();
            if(tensor.type!=OnnxJavaType.FLOAT)throw new IOException("换脸模型必须使用 FP32 张量");
            JSONArray shape=row.getJSONArray("shape");long[] actual=tensor.getShape();
            if(actual.length!=shape.length())throw new IOException("换脸模型张量维数不匹配");
            for(int j=0;j<actual.length;j++)if(shape.get(j) instanceof Number&&actual[j]!=shape.getLong(j))throw new IOException("换脸模型张量尺寸不匹配");
        }
    }

    private static float[] flatten(Object value,int expected) throws IOException {
        float[] out=new float[expected];int count=flattenInto(value,out,0);
        if(count!=expected)throw new IOException("换脸模型输出长度错误");
        for(float f:out)if(!Float.isFinite(f))throw new IOException("换脸模型输出包含无效数值");
        return out;
    }
    private static int flattenInto(Object value,float[] out,int offset) throws IOException {
        if(value instanceof float[]) {
            float[] values=(float[])value;
            if(offset+values.length>out.length)throw new IOException("换脸模型输出过长");
            System.arraycopy(values,0,out,offset,values.length);return offset+values.length;
        }
        if(value==null||!value.getClass().isArray())throw new IOException("换脸模型输出类型错误");
        for(int i=0;i<Array.getLength(value);i++)offset=flattenInto(Array.get(value,i),out,offset);
        return offset;
    }

    /** Stops an ongoing native Run via ONNX RunOptions; closes before RunOptions. */
    private static final class CancelMonitor implements AutoCloseable {
        private final Thread thread;
        private volatile boolean closed;
        CancelMonitor(AtomicBoolean cancelled,OrtSession.RunOptions options) {
            thread=new Thread(()->{
                while(!closed) {
                    if(cancelled.get()) {try{options.setTerminate(true);}catch(Exception ignored){}return;}
                    try{Thread.sleep(80);}catch(InterruptedException done){return;}
                }
            },"face-cancel");
            thread.setDaemon(true);thread.start();
        }
        @Override public void close() {
            closed=true;thread.interrupt();
            boolean interrupted=false;
            while(thread.isAlive())try{thread.join();}catch(InterruptedException e){interrupted=true;}
            if(interrupted)Thread.currentThread().interrupt();
        }
    }
}
