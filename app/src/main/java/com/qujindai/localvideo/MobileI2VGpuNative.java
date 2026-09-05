package com.qujindai.localvideo;

import java.io.File;

/** JNI bridge for the production MobileI2V Adreno/MNN OpenCL runtime. */
public final class MobileI2VGpuNative {
    public static final int WIDTH = 1280;
    public static final int HEIGHT = 720;
    public static final int OUTPUT_FRAMES = 17;
    public static final int LATENT_CHANNELS = 128;
    public static final int LATENT_FRAMES = 3;
    public static final int LATENT_HEIGHT = 23;
    public static final int LATENT_WIDTH = 40;
    public static final int SINGLE_LATENT_FLOATS = LATENT_CHANNELS * LATENT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH;
    public static final int GUIDE_LATENT_FLOATS = LATENT_CHANNELS * LATENT_HEIGHT * LATENT_WIDTH;
    public static final int LATENT_CFG_FLOATS = 2 * SINGLE_LATENT_FLOATS;
    public static final int ENCODER_INPUT_FLOATS = 3 * HEIGHT * WIDTH;
    public static final int FRAME_ARGB_PIXELS = HEIGHT * WIDTH;
    private static final boolean NATIVE_LOADED;
    private static final String NATIVE_ERROR;
    static {
        boolean loaded=false; String error="";
        try { System.loadLibrary("mobilei2v_gpu"); loaded=true; }
        catch(Throwable f){ error=f.getClass().getSimpleName()+(f.getMessage()==null?"":": "+f.getMessage()); }
        NATIVE_LOADED=loaded; NATIVE_ERROR=error;
    }
    private MobileI2VGpuNative() {}
    public static boolean isNativeLoaded(){return NATIVE_LOADED;}
    public static String nativeError(){return NATIVE_ERROR;}
    public static Probe probe(InstalledModelPack pack){
        if(pack==null||!pack.isAcceleratedMobileI2V()) return new Probe(false,false,"加速 MobileI2V 模型包未安装");
        if(!NATIVE_LOADED) return new Probe(false,false,"GPU native runtime 加载失败 · "+NATIVE_ERROR);
        for(String name:new String[]{"denoiser.mnn","vae_encoder.mnn","vae_decoder.mnn","empty_prompt.f16","empty_prompt_mask.bin"}){
            if(!pack.artifact(name).isFile()) return new Probe(false,false,name+" 缺失");
        }
        try{
            String result=nativeProbe(pack.root.getAbsolutePath());
            boolean ready=result!=null&&result.startsWith("MNN_OPENCL_READY:");
            return new Probe(ready,ready&&result.contains("cache-enabled"),result==null?"native probe returned null":result);
        }catch(Throwable e){return new Probe(false,false,"MNN OpenCL 探测异常 · "+e.getClass().getSimpleName());}
    }
    public static Session load(InstalledModelPack pack){
        Probe p=probe(pack); if(!p.openClReady) throw new IllegalStateException(p.message);
        long h=nativeLoad(pack.root.getAbsolutePath()); if(h==0L) throw new IllegalStateException("MNN OpenCL MobileI2V session load failed");
        return new Session(h);
    }
    public static final class Probe{
        public final boolean openClReady,tuningCacheEnabled; public final String message;
        Probe(boolean r,boolean c,String m){openClReady=r;tuningCacheEnabled=c;message=m;}
    }
    public static final class Session implements AutoCloseable{
        private long handle; Session(long h){handle=h;}
        public synchronized void encode(float[] image,float[] guide){requireOpen();requireLength(image,ENCODER_INPUT_FLOATS,"imageNchw");requireLength(guide,GUIDE_LATENT_FLOATS,"guideLatent");int c=nativeEncode(handle,image,guide);if(c!=0)throw new IllegalStateException("GPU VAE encoder failed · code="+c);}
        public synchronized void runDenoiser(float[] latent,float timestep,float flowScore,float[] output){requireOpen();requireLength(latent,LATENT_CFG_FLOATS,"latentCfg2");requireLength(output,LATENT_CFG_FLOATS,"outputCfg2");int c=nativeRunDenoiser(handle,latent,timestep,new float[]{flowScore,flowScore},output);if(c!=0)throw new IllegalStateException("GPU denoiser failed · code="+c);}
        public synchronized void decode(float[] latent){requireOpen();requireLength(latent,SINGLE_LATENT_FLOATS,"latent");int c=nativeDecode(handle,latent);if(c!=0)throw new IllegalStateException("GPU VAE decoder failed · code="+c);}
        public synchronized void copyDecodedFrameArgb(int frame,int[] argb){requireOpen();if(frame<0||frame>=OUTPUT_FRAMES)throw new IllegalArgumentException("invalid frame "+frame);if(argb==null||argb.length!=FRAME_ARGB_PIXELS)throw new IllegalArgumentException("argb must contain "+FRAME_ARGB_PIXELS+" pixels");int c=nativeCopyDecodedFrameArgb(handle,frame,argb);if(c!=0)throw new IllegalStateException("decoded frame copy failed · code="+c);}
        public synchronized void clearDecoded(){if(handle!=0L)nativeClearDecoded(handle);}
        private void requireOpen(){if(handle==0L)throw new IllegalStateException("GPU session already closed");}
        @Override public synchronized void close(){if(handle!=0L){nativeRelease(handle);handle=0L;}}
    }
    private static void requireLength(float[] v,int e,String n){if(v==null||v.length!=e)throw new IllegalArgumentException(n+" must contain "+e+" floats");}
    private static native String nativeProbe(String modelDir);
    private static native long nativeLoad(String modelDir);
    private static native int nativeEncode(long handle,float[] imageNchw,float[] guideLatent);
    private static native int nativeRunDenoiser(long handle,float[] latentCfg2,float timestep,float[] flowScoreCfg2,float[] outputCfg2);
    private static native int nativeDecode(long handle,float[] latent);
    private static native int nativeCopyDecodedFrameArgb(long handle,int frame,int[] argb);
    private static native void nativeClearDecoded(long handle);
    private static native void nativeRelease(long handle);
}
