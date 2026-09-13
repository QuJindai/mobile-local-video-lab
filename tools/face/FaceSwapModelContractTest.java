package com.qujindai.localvideo;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;

/** Compare the Android Java core against actual model outputs and OpenCV references. */
public final class FaceSwapModelContractTest {
    public static void main(String[] args) throws Exception {
        Path root=Path.of(args[0]);
        for(int fixture=0;fixture<2;fixture++) {
            Path p=root.resolve("contract/"+fixture);
            int[] dims=ints(p.resolve("dimensions.i32"));
            float[][] raw=new float[9][];
            for(int i=0;i<9;i++)raw[i]=floats(p.resolve("detector-"+i+".f32"));
            List<FaceSwapMath.Detection> faces=FaceSwapMath.decodeScrfd(raw,640,dims[0],dims[1],dims[2],dims[3],.6f);
            if(faces.isEmpty())throw new AssertionError("Actual detector outputs produced no faces");
            FaceSwapMath.Detection face=faces.stream().max(Comparator.comparingDouble(f->(f.x2-f.x1)*(f.y2-f.y1))).orElseThrow();
            float[] expected=floats(p.resolve("landmarks.f32"));
            double landmarkError=0;
            for(int i=0;i<5;i++)for(int j=0;j<2;j++)landmarkError=Math.max(landmarkError,Math.abs(face.landmarks[i][j]-expected[2*i+j]));
            if(landmarkError>2e-4)throw new AssertionError("Detector decode/scale mismatch "+landmarkError);
            int size=fixture==0?112:128;
            double[] affine=FaceSwapMath.fit(face.landmarks,size);
            double[] reference=doubles(p.resolve("affine.f64"));
            for(int i=0;i<6;i++)if(Math.abs(affine[i]-reference[i])>2e-4)throw new AssertionError("Affine convention mismatch "+i);
            int[] warped=FaceSwapPixels.warp(ints(p.resolve("image.argb")),dims[0],dims[1],affine,size);
            int[] aligned=ints(p.resolve("aligned.argb"));
            double difference=0;
            for(int i=0;i<warped.length;i++)for(int shift:new int[]{16,8,0})difference+=Math.abs(((warped[i]>>>shift)&255)-((aligned[i]>>>shift)&255));
            difference/=3.0*warped.length;
            // OpenCV uses a 1/32 interpolation table; Java computes continuous bilinear.
            if(difference>1.0)throw new AssertionError("RGB warp differs from OpenCV: "+difference);
            System.out.println("Actual-model fixture "+fixture+" PASS faces="+faces.size()+" landmarkMax="+landmarkError+" warpRgbMae="+difference);
        }
        float[] actual=FaceSwapMath.projectIdentity(floats(root.resolve("contract/embedding.f32")),floats(root.resolve("emap.bin")));
        float[] expected=floats(root.resolve("contract/projected.f32"));
        double error=0;
        for(int i=0;i<actual.length;i++)error=Math.max(error,Math.abs(actual[i]-expected[i]));
        if(error>2e-6)throw new AssertionError("Identity projection mismatch "+error);
        System.out.println("Actual-model identity projection PASS maxAbs="+error);
    }
    private static ByteBuffer bytes(Path path)throws Exception{return ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN);}
    private static int[] ints(Path p)throws Exception{ByteBuffer b=bytes(p);int[] out=new int[b.remaining()/4];b.asIntBuffer().get(out);return out;}
    private static float[] floats(Path p)throws Exception{ByteBuffer b=bytes(p);float[] out=new float[b.remaining()/4];b.asFloatBuffer().get(out);return out;}
    private static double[] doubles(Path p)throws Exception{ByteBuffer b=bytes(p);double[] out=new double[b.remaining()/8];b.asDoubleBuffer().get(out);return out;}
}
