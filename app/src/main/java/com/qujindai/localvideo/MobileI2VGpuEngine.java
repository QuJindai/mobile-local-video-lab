package com.qujindai.localvideo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Debug;
import android.os.PowerManager;
import android.os.SystemClock;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Properties;
import java.util.Random;

/** Full MobileI2V production pipeline. Neural inference is MNN OpenCL only. */
public final class MobileI2VGpuEngine {
    public static final int STEPS=28; public static final float CFG_SCALE=4.5f; public static final float FLOW_SCORE=2.0f;
    public interface ProgressListener{void onProgress(int percent,String message);}
    public static final class Result{
        public final Uri uri; public final int frames,width,height,fps; public final long elapsedMs; public final MobileI2VMicroscope microscope;
        Result(Uri uri,int fps,long elapsed,MobileI2VMicroscope m){this.uri=uri;this.frames=17;this.width=1280;this.height=720;this.fps=fps;this.elapsedMs=elapsed;this.microscope=m;}
    }
    private final Context context; public MobileI2VGpuEngine(Context c){context=c.getApplicationContext();}
    public Result generate(Uri imageUri,InstalledModelPack pack,int fps,ProgressListener listener)throws Exception{
        if(imageUri==null)throw new IllegalArgumentException("imageUri is required");
        if(pack==null||!pack.isAcceleratedMobileI2V())throw new IllegalStateException("可执行 MobileI2V GPU 模型包未安装");
        MobileI2VGpuNative.Probe probe=MobileI2VGpuNative.probe(pack);if(!probe.openClReady)throw new IllegalStateException(probe.message);
        long started=SystemClock.elapsedRealtime(),javaBefore=javaHeapMb(),nativeBefore=nativeHeapMb(),javaPeak=javaBefore,nativePeak=nativeBefore;
        int thermalBefore=thermalStatus(); long encodeMs,denoiseMs,decodeMs,mp4Ms;
        File job=new File(context.getCacheDir(),"mobilei2v-"+System.nanoTime()),framesDir=new File(job,"frames");if(!framesDir.mkdirs())throw new IllegalStateException("cannot create MobileI2V frame directory");
        try(MobileI2VGpuNative.Session session=MobileI2VGpuNative.load(pack)){
            progress(listener,2,"读取主图 · 1280×720"); float[] image=loadImageNchw(imageUri),guide=new float[MobileI2VGpuNative.GUIDE_LATENT_FLOATS];
            long stage=SystemClock.elapsedRealtime();progress(listener,6,"GPU VAE encode · MNN OpenCL");session.encode(image,guide);encodeMs=SystemClock.elapsedRealtime()-stage;image=null;javaPeak=Math.max(javaPeak,javaHeapMb());nativePeak=Math.max(nativePeak,nativeHeapMb());
            float[] initial=gaussianLatent(1L);MobileI2VFlowEuler.lockGuideFirstSlice(initial,guide,128,3,23,40);
            stage=SystemClock.elapsedRealtime();progress(listener,12,"MobileI2V · Adreno GPU 去噪 0/"+STEPS);
            float[] sampled=MobileI2VFlowEuler.sample(initial,guide,128,3,23,40,STEPS,CFG_SCALE,(cfg,t,out)->session.runDenoiser(cfg,t,FLOW_SCORE,out),(done,total,t)->progress(listener,12+done*55/total,"MobileI2V · Adreno GPU 去噪 "+done+"/"+total));
            denoiseMs=SystemClock.elapsedRealtime()-stage;initial=null;guide=null;javaPeak=Math.max(javaPeak,javaHeapMb());nativePeak=Math.max(nativePeak,nativeHeapMb());
            stage=SystemClock.elapsedRealtime();progress(listener,70,"GPU VAE decode · 17 帧");session.decode(sampled);decodeMs=SystemClock.elapsedRealtime()-stage;sampled=null;javaPeak=Math.max(javaPeak,javaHeapMb());nativePeak=Math.max(nativePeak,nativeHeapMb());
            progress(listener,77,"逐帧取回 GPU 解码结果");List<File> frameFiles=new ArrayList<>(17);int[] argb=new int[MobileI2VGpuNative.FRAME_ARGB_PIXELS];
            for(int frame=0;frame<17;frame++){session.copyDecodedFrameArgb(frame,argb);Bitmap bitmap=Bitmap.createBitmap(argb,1280,720,Bitmap.Config.ARGB_8888);File file=new File(framesDir,String.format(Locale.US,"%03d.png",frame));try(FileOutputStream out=new FileOutputStream(file)){if(!bitmap.compress(Bitmap.CompressFormat.PNG,100,out))throw new IllegalStateException("frame PNG encode failed: "+frame);}finally{bitmap.recycle();}frameFiles.add(file);progress(listener,77+(frame+1)*7/17,"帧缓存 "+(frame+1)+"/17");}
            session.clearDecoded();stage=SystemClock.elapsedRealtime();progress(listener,85,"H.264 / MediaCodec 输出 MP4");File mp4=new File(job,"mobilei2v-gpu.mp4");Mp4Encoder.encode(frameFiles,mp4,1280,720,fps,(encoded,total)->progress(listener,85+encoded*10/Math.max(1,total),"MP4 "+encoded+"/"+total));mp4Ms=SystemClock.elapsedRealtime()-stage;
            progress(listener,96,"写入 Movies/LocalVideoLab");String stamp=new SimpleDateFormat("yyyyMMdd_HHmmss",Locale.US).format(new Date());Uri uri=MediaStorePublisher.publish(context,mp4,"mobilei2v_gpu_"+stamp+".mp4");
            javaPeak=Math.max(javaPeak,javaHeapMb());nativePeak=Math.max(nativePeak,nativeHeapMb());AcceleratedPackManifest a=pack.acceleratedManifest;
            MobileI2VMicroscope microscope=new MobileI2VMicroscope(a.id,a.version,a.sourceCommit,a.checkpointSha256,a.dreamCommit,a.mnnCommit,MobileI2VRuntimeProbe.Backend.MNN_OPENCL,MobileI2VRuntimeProbe.Backend.MNN_OPENCL,true,probe.tuningCacheEnabled,vaeImpl(pack),STEPS,17,1280,720,encodeMs,denoiseMs,decodeMs,mp4Ms,javaBefore,javaPeak,nativeBefore,nativePeak,thermalBefore,thermalStatus(),false);
            if(!microscope.acceleratedEvidenceValid())throw new IllegalStateException("MobileI2V 加速证据无效");progress(listener,100,"MobileI2V GPU 完成");return new Result(uri,fps,SystemClock.elapsedRealtime()-started,microscope);
        }finally{deleteRecursively(job);}
    }
    private float[] loadImageNchw(Uri uri)throws Exception{Bitmap raw;try(InputStream in=context.getContentResolver().openInputStream(uri)){if(in==null)throw new IllegalStateException("cannot open input image");raw=BitmapFactory.decodeStream(in);}if(raw==null)throw new IllegalStateException("cannot decode input image");Bitmap scaled=raw.getWidth()==1280&&raw.getHeight()==720?raw:Bitmap.createScaledBitmap(raw,1280,720,true);if(scaled!=raw)raw.recycle();int pixels=1280*720;int[] argb=new int[pixels];scaled.getPixels(argb,0,1280,0,0,1280,720);scaled.recycle();float[] out=new float[3*pixels];for(int i=0;i<pixels;i++){int p=argb[i];out[i]=(((p>>>16)&255)/127.5f)-1f;out[pixels+i]=(((p>>>8)&255)/127.5f)-1f;out[2*pixels+i]=((p&255)/127.5f)-1f;}return out;}
    private static float[] gaussianLatent(long seed){float[] l=new float[MobileI2VGpuNative.SINGLE_LATENT_FLOATS];Random r=new Random(seed);for(int i=0;i<l.length;i++)l[i]=(float)r.nextGaussian();return l;}
    private static String vaeImpl(InstalledModelPack pack){File f=pack.artifact("runtime.properties");if(!f.isFile())return "LTX-Video";Properties p=new Properties();try(FileInputStream in=new FileInputStream(f)){p.load(in);return p.getProperty("vae.impl","LTX-Video");}catch(Exception e){return "LTX-Video";}}
    private int thermalStatus(){try{PowerManager p=context.getSystemService(PowerManager.class);return p==null?-1:p.getCurrentThermalStatus();}catch(Throwable e){return -1;}}
    private static long javaHeapMb(){Runtime r=Runtime.getRuntime();return Math.max(0L,(r.totalMemory()-r.freeMemory())/1048576L);} private static long nativeHeapMb(){return Math.max(0L,Debug.getNativeHeapAllocatedSize()/1048576L);}
    private static void progress(ProgressListener l,int p,String m){if(l!=null)l.onProgress(p,m);} private static void deleteRecursively(File f){if(f==null||!f.exists())return;if(f.isDirectory()){File[] cs=f.listFiles();if(cs!=null)for(File c:cs)deleteRecursively(c);}f.delete();}
}
