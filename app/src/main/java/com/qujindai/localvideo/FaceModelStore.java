package com.qujindai.localvideo;

import android.content.Context;
import android.os.StatFs;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;

/** App-private, pinned photo-swap weights; unrelated to MobileI2V downloads. */
public final class FaceModelStore {
    private static final Object INSTALL_LOCK = new Object();
    private static final String ASSET = "models/face-swap/models.json";
    private static final String MODEL_ID = "inswapper128-fp32-v1";
    private final Context context;
    private final File directory;
    private final List<Model> models = new ArrayList<>();
    private JSONObject manifest;
    private String manifestHash;
    private Exception manifestError;

    private static final class Model {
        final String name, url, hash;
        final long bytes;
        Model(JSONObject row) throws Exception {
            name=row.getString("name");url=row.getString("url");hash=row.getString("sha256");
            bytes=row.getLong("bytes");
            if (!hash.matches("[a-f0-9]{64}") || bytes <= 0 || bytes > 600_000_000L
                    || !url.equals("https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/"+name))
                throw new IOException("换脸模型清单无效");
        }
    }

    public FaceModelStore(Context context) {
        this.context=context.getApplicationContext();
        directory=new File(this.context.getFilesDir(),"face-models/"+MODEL_ID);
        try (InputStream in=this.context.getAssets().open(ASSET)) {
            byte[] bytes=readSmall(in,64*1024);
            manifestHash=hex(MessageDigest.getInstance("SHA-256").digest(bytes));
            manifest=new JSONObject(new String(bytes, StandardCharsets.UTF_8));
            if (manifest.getInt("schema_version")!=1 || !MODEL_ID.equals(manifest.getString("model_id")))
                throw new IOException("不兼容的换脸模型版本");
            HashSet<String> names=new HashSet<>(Arrays.asList("scrfd_2.5g.onnx","arcface_w600k_r50.onnx","inswapper_128.onnx"));
            JSONArray rows=manifest.getJSONArray("models");
            if (rows.length()!=3) throw new IOException("换脸模型清单不完整");
            for(int i=0;i<rows.length();i++) {
                Model model=new Model(rows.getJSONObject(i));
                if (!names.remove(model.name)) throw new IOException("未知或重复的换脸模型");
                models.add(model);
            }
            if (!manifest.getString("emap_sha256").matches("[a-f0-9]{64}")
                    || manifest.getLong("emap_bytes")!=512L*512*4) throw new IOException("身份投影清单无效");
        } catch(Exception error) {manifestError=error;models.clear();}
    }

    public boolean isReady() {
        if(manifestError!=null) return false;
        try {
            File marker=new File(directory,"verified.txt");
            if (!marker.isFile() || marker.length()!=64) return false;
            try(InputStream in=new FileInputStream(marker)) {
                if(!manifestHash.equals(new String(readSmall(in,64),StandardCharsets.US_ASCII)))return false;
            }
            for(Model model:models) {
                File file=new File(directory,model.name);
                if(!file.isFile() || file.length()!=model.bytes)return false;
            }
            return true;
        } catch(Exception error) {return false;}
    }

    public String status() {
        if(manifestError!=null)return "此安装包缺少有效的换脸模型清单";
        return isReady()?"换脸模型已安装 · 生成前校验":"换脸模型未准备好 · 需下载约 699 MiB";
    }

    public void install(FaceSwapEngine.Progress progress,AtomicBoolean cancelled) throws Exception {
        synchronized(INSTALL_LOCK) {
            checkManifest();
            if(!directory.isDirectory()&&!directory.mkdirs())throw new IOException("无法创建换脸模型目录");
            long needed=0;
            for(Model model:models) {
                File full=new File(directory,model.name),partial=new File(directory,model.name+".partial");
                if(!full.isFile() || full.length()!=model.bytes)
                    needed+=Math.max(0,model.bytes-(partial.isFile()?Math.min(partial.length(),model.bytes):0));
            }
            if(new StatFs(directory.getAbsolutePath()).getAvailableBytes()<needed+128L*1024*1024)
                throw new IOException("剩余存储不足，准备换脸模型需要约 830 MiB 可用空间");
            long completed=0,total=totalBytes();
            for(Model model:models) {
                checkCancelled(cancelled);
                File file=new File(directory,model.name);
                progress.update((int)(completed*90/total),"校验 "+model.name);
                if(!valid(file,model,cancelled)) {
                    download(model,file,completed,total,progress,cancelled);
                }
                completed+=model.bytes;
            }
            checkCancelled(cancelled);
            // The marker is written only after all downloaded files pass SHA-256.
            File temporary=new File(directory,"verified.txt.partial");
            try(FileOutputStream out=new FileOutputStream(temporary)) {
                out.write(manifestHash.getBytes(StandardCharsets.US_ASCII));out.getFD().sync();
            }
            File marker=new File(directory,"verified.txt");
            if(!temporary.renameTo(marker))throw new IOException("换脸模型激活失败");
            progress.update(100,"换脸模型已准备好");
        }
    }

    public void verifyForGeneration(FaceSwapEngine.Progress progress,AtomicBoolean cancelled) throws Exception {
        checkManifest();
        if(!isReady())throw new IOException("请先准备换脸模型");
        int i=0;
        for(Model model:models) {
            progress.update(2+i*3,"校验换脸模型 "+(++i)+"/3");
            if(!valid(new File(directory,model.name),model,cancelled)) {
                new File(directory,"verified.txt").delete();
                throw new IOException("换脸模型校验失败，请重新准备："+model.name);
            }
        }
    }

    public File file(String name) throws IOException {
        for(Model model:models)if(model.name.equals(name))return new File(directory,name);
        throw new IOException("未知换脸模型");
    }

    public JSONObject io(String role) throws Exception {checkManifest();return manifest.getJSONObject("io").getJSONObject(role);}
    public String modelId(){return MODEL_ID;}

    public float[] readProjection() throws Exception {
        checkManifest();
        byte[] bytes;
        try(InputStream in=context.getAssets().open("models/face-swap/emap.bin")) {
            bytes=readSmall(in,512*512*4);
        }
        if(bytes.length!=512*512*4 || !hex(MessageDigest.getInstance("SHA-256").digest(bytes)).equals(manifest.getString("emap_sha256")))
            throw new IOException("身份投影校验失败");
        float[] result=new float[512*512];
        java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(result);
        return result;
    }

    private void download(Model model,File file,long completed,long total,FaceSwapEngine.Progress progress,AtomicBoolean cancelled) throws Exception {
        File partial=new File(directory,model.name+".partial");
        for(int attempt=0;attempt<3;attempt++) {
            checkCancelled(cancelled);
            if(partial.length()>model.bytes && !partial.delete())throw new IOException("无法重建下载缓存");
            if(partial.length()==model.bytes) {
                if(valid(partial,model,cancelled)) {publish(partial,file);return;}
                if(!partial.delete())throw new IOException("无法删除损坏的下载缓存");
            }
            long offset=partial.isFile()?partial.length():0;
            if(new StatFs(directory.getAbsolutePath()).getAvailableBytes()<model.bytes-offset+32L*1024*1024)
                throw new IOException("剩余存储不足，无法完整下载或修复换脸模型");
            HttpURLConnection connection=(HttpURLConnection)new URL(model.url).openConnection();
            connection.setConnectTimeout(20_000);connection.setReadTimeout(20_000);
            connection.setInstanceFollowRedirects(true);
            connection.setRequestProperty("Accept-Encoding","identity");
            if(offset>0)connection.setRequestProperty("Range","bytes="+offset+"-");
            try {
                int code=connection.getResponseCode();
                boolean append=offset>0 && code==206;
                if(append) {
                    String range=connection.getHeaderField("Content-Range");
                    if(range==null || !range.startsWith("bytes "+offset+"-"))throw new IOException("模型下载续传位置错误");
                } else if(code==200)offset=0;
                else throw new IOException("模型下载 HTTP "+code);
                byte[] buffer=new byte[256*1024];
                try(InputStream in=connection.getInputStream();FileOutputStream out=new FileOutputStream(partial,append)) {
                    int count;long received=offset;
                    while((count=in.read(buffer))!=-1) {
                        checkCancelled(cancelled);
                        if(received+count>model.bytes)throw new IOException("模型下载大小异常");
                        out.write(buffer,0,count);received+=count;
                        progress.update((int)((completed+received)*90/total),"下载换脸模型 "+((completed+received)/(1024*1024))+" / "+(total/(1024*1024))+" MiB");
                    }
                    out.getFD().sync();
                }
                checkCancelled(cancelled);
                if(!valid(partial,model,cancelled)) {
                    partial.delete();throw new IOException("模型下载校验未通过");
                }
                publish(partial,file);return;
            } catch(IOException error) {
                if(attempt==2)throw error;
                progress.update((int)(completed*90/total),"连接中断，正在续传换脸模型");
            } finally {connection.disconnect();}
        }
    }

    private static void publish(File partial,File full) throws IOException {
        if(!partial.renameTo(full))throw new IOException("无法保存已校验的换脸模型");
    }

    private static boolean valid(File file,Model model,AtomicBoolean cancelled) throws Exception {
        if(!file.isFile()||file.length()!=model.bytes)return false;
        MessageDigest digest=MessageDigest.getInstance("SHA-256");
        try(InputStream in=new FileInputStream(file)) {
            byte[] buffer=new byte[1024*1024];int n;
            while((n=in.read(buffer))!=-1){checkCancelled(cancelled);digest.update(buffer,0,n);}
        }
        return model.hash.equals(hex(digest.digest()));
    }

    private long totalBytes(){long total=0;for(Model model:models)total+=model.bytes;return total;}
    private void checkManifest() throws IOException {if(manifestError!=null)throw new IOException("换脸模型清单不可用",manifestError);}
    static void checkCancelled(AtomicBoolean cancelled){if(cancelled.get()||Thread.currentThread().isInterrupted())throw new CancellationException("已取消");}
    private static String hex(byte[] bytes){StringBuilder s=new StringBuilder();for(byte b:bytes)s.append(String.format(java.util.Locale.ROOT,"%02x",b&255));return s.toString();}
    private static byte[] readSmall(InputStream in,int limit) throws IOException {
        java.io.ByteArrayOutputStream out=new java.io.ByteArrayOutputStream();byte[] buffer=new byte[8192];int n;
        while((n=in.read(buffer))!=-1){if(out.size()+n>limit)throw new IOException("模型清单或投影大小异常");out.write(buffer,0,n);}
        return out.toByteArray();
    }
}
