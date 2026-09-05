#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mnn_runtime.hpp"

namespace {
constexpr const char* kTag = "MobileI2VGpu";
constexpr size_t kWidth = 1280u;
constexpr size_t kHeight = 720u;
constexpr size_t kVideoFrames = 17u;
constexpr size_t kSingleLatent = 128u * 3u * 23u * 40u;
constexpr size_t kGuideLatent = 128u * 23u * 40u;
constexpr size_t kLatentCfg = 2u * kSingleLatent;
constexpr size_t kEncoderInput = 3u * kHeight * kWidth;
constexpr size_t kPromptCfg = 2u * 1u * 300u * 896u;
constexpr size_t kCondMask = 3u * 23u * 40u;
constexpr size_t kGuidePlane = 23u * 40u;
constexpr size_t kTextMask = 300u;
constexpr size_t kFlowScore = 2u;
constexpr size_t kDecodedVideo = 3u * kVideoFrames * kHeight * kWidth;
constexpr size_t kFramePixels = kWidth * kHeight;

struct RuntimeHandle {
    mobilei2v::OpenClSession encoder;
    mobilei2v::OpenClSession denoiser;
    mobilei2v::OpenClSession decoder;
    std::vector<float> promptCfg;
    std::vector<uint8_t> textMask;
    std::vector<float> condMask;
    std::vector<float> decoded;
};

std::mutex gMutex;
std::unordered_map<jlong, std::unique_ptr<RuntimeHandle>> gHandles;
jlong gNextHandle = 1;

std::string toString(JNIEnv* env, jstring value) {
    if (!value) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

jstring newString(JNIEnv* env, const std::string& value) { return env->NewStringUTF(value.c_str()); }

float halfToFloat(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16u;
    uint32_t exponent = (value >> 10u) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1u; ++shift; }
            mantissa &= 0x03ffu;
            const uint32_t exp32 = static_cast<uint32_t>(127 - 15 - shift + 1);
            bits = sign | (exp32 << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 0x1fu) bits = sign | 0x7f800000u | (mantissa << 13u);
    else bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    float out; std::memcpy(&out, &bits, sizeof(out)); return out;
}

bool readPrompt(const std::string& path, std::vector<float>& output) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamsize>(kPromptCfg * sizeof(uint16_t))) return false;
    stream.seekg(0, std::ios::beg);
    std::vector<uint16_t> half(kPromptCfg);
    if (!stream.read(reinterpret_cast<char*>(half.data()), static_cast<std::streamsize>(half.size()*2u))) return false;
    output.resize(kPromptCfg);
    for (size_t i=0;i<kPromptCfg;i++) { output[i]=halfToFloat(half[i]); if(!std::isfinite(output[i])) return false; }
    return true;
}

bool readMask(const std::string& path, std::vector<uint8_t>& output) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamsize>(kTextMask)) return false;
    stream.seekg(0, std::ios::beg); output.resize(kTextMask);
    return static_cast<bool>(stream.read(reinterpret_cast<char*>(output.data()), kTextMask));
}

std::vector<float> buildCondMask() {
    std::vector<float> v(kCondMask, 0.0f);
    std::fill(v.begin(), v.begin() + kGuidePlane, 1.0f);
    return v;
}

bool expected(MNN::Interpreter* i, MNN::Session* s, const char* name, size_t count, bool output=false) {
    if (!i || !s) return false;
    auto* t = output ? i->getSessionOutput(s, name) : i->getSessionInput(s, name);
    return t && static_cast<size_t>(t->elementSize()) == count;
}

bool validate(const RuntimeHandle& h) {
    return h.encoder && h.denoiser && h.decoder
        && expected(h.encoder.interpreter.get(), h.encoder.session, "image", kEncoderInput)
        && expected(h.encoder.interpreter.get(), h.encoder.session, "latent", kGuideLatent, true)
        && expected(h.denoiser.interpreter.get(), h.denoiser.session, "latent", kLatentCfg)
        && expected(h.denoiser.interpreter.get(), h.denoiser.session, "timestep", 2u)
        && expected(h.denoiser.interpreter.get(), h.denoiser.session, "prompt", kPromptCfg)
        && expected(h.denoiser.interpreter.get(), h.denoiser.session, "cond_mask", kCondMask)
        && expected(h.denoiser.interpreter.get(), h.denoiser.session, "text_mask", kTextMask)
        && expected(h.denoiser.interpreter.get(), h.denoiser.session, "flow_score", kFlowScore)
        && expected(h.denoiser.interpreter.get(), h.denoiser.session, "output", kLatentCfg, true)
        && expected(h.decoder.interpreter.get(), h.decoder.session, "latent", kSingleLatent)
        && expected(h.decoder.interpreter.get(), h.decoder.session, "video", kDecodedVideo, true);
}

std::unique_ptr<RuntimeHandle> createHandle(const std::string& dir) {
    auto h=std::make_unique<RuntimeHandle>();
    if (!readPrompt(dir+"/empty_prompt.f16",h->promptCfg)) return nullptr;
    if (!readMask(dir+"/empty_prompt_mask.bin",h->textMask)) return nullptr;
    h->condMask=buildCondMask();
    h->encoder=mobilei2v::createOpenClSession(dir+"/vae_encoder.mnn",dir,"vae-encoder-opencl.cache");
    h->denoiser=mobilei2v::createOpenClSession(dir+"/denoiser.mnn",dir,"denoiser-opencl.cache");
    h->decoder=mobilei2v::createOpenClSession(dir+"/vae_decoder.mnn",dir,"vae-decoder-opencl.cache");
    if (!validate(*h)) return nullptr;
    return h;
}

RuntimeHandle* get(jlong id) {
    auto it=gHandles.find(id); return it==gHandles.end()?nullptr:it->second.get();
}
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeProbe(JNIEnv* env,jclass,jstring dirValue) {
    const std::string dir=toString(env,dirValue); if(dir.empty()) return newString(env,"NOT_READY:empty model directory");
    auto h=createHandle(dir); if(!h) return newString(env,"NOT_READY:MNN OpenCL full graph contract failed");
    bool cache=!h->encoder.cacheFile.empty()&&!h->denoiser.cacheFile.empty()&&!h->decoder.cacheFile.empty();
    return newString(env,cache?"MNN_OPENCL_READY:cache-enabled":"MNN_OPENCL_READY:cache-disabled");
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeLoad(JNIEnv* env,jclass,jstring dirValue) {
    const std::string dir=toString(env,dirValue); if(dir.empty()) return 0;
    auto h=createHandle(dir); if(!h) return 0;
    std::lock_guard<std::mutex> guard(gMutex); const jlong id=gNextHandle++; gHandles.emplace(id,std::move(h)); return id;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeEncode(JNIEnv* env,jclass,jlong id,jfloatArray image,jfloatArray guide) {
    std::lock_guard<std::mutex> guard(gMutex); RuntimeHandle* h=get(id); if(!h) return -1;
    if(!image||env->GetArrayLength(image)!=static_cast<jsize>(kEncoderInput)) return -2;
    if(!guide||env->GetArrayLength(guide)!=static_cast<jsize>(kGuideLatent)) return -3;
    jfloat* in=env->GetFloatArrayElements(image,nullptr); if(!in) return -4;
    bool ok=mobilei2v::copyFloatInput(h->encoder.interpreter.get(),h->encoder.session,"image",in,kEncoderInput);
    env->ReleaseFloatArrayElements(image,in,JNI_ABORT); if(!ok) return -5;
    int code=h->encoder.interpreter->runSession(h->encoder.session); if(code!=0) return -6;
    jfloat* out=env->GetFloatArrayElements(guide,nullptr); if(!out) return -7;
    ok=mobilei2v::copyFloatOutput(h->encoder.interpreter.get(),h->encoder.session,"latent",out,kGuideLatent);
    env->ReleaseFloatArrayElements(guide,out,ok?0:JNI_ABORT); return ok?0:-8;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeRunDenoiser(JNIEnv* env,jclass,jlong id,jfloatArray latentValue,jfloat timestepValue,jfloatArray flowValue,jfloatArray outputValue) {
    std::lock_guard<std::mutex> guard(gMutex); RuntimeHandle* h=get(id); if(!h) return -1;
    if(!latentValue||env->GetArrayLength(latentValue)!=static_cast<jsize>(kLatentCfg)) return -2;
    if(!flowValue||env->GetArrayLength(flowValue)!=2) return -3;
    if(!outputValue||env->GetArrayLength(outputValue)!=static_cast<jsize>(kLatentCfg)) return -4;
    jfloat* latent=env->GetFloatArrayElements(latentValue,nullptr); jfloat* flow=env->GetFloatArrayElements(flowValue,nullptr);
    if(!latent||!flow){if(latent)env->ReleaseFloatArrayElements(latentValue,latent,JNI_ABORT);if(flow)env->ReleaseFloatArrayElements(flowValue,flow,JNI_ABORT);return -5;}
    const float timestep[2]={timestepValue,timestepValue};
    bool ok=mobilei2v::copyFloatInput(h->denoiser.interpreter.get(),h->denoiser.session,"latent",latent,kLatentCfg)
      && mobilei2v::copyFloatInput(h->denoiser.interpreter.get(),h->denoiser.session,"timestep",timestep,2u)
      && mobilei2v::copyFloatInput(h->denoiser.interpreter.get(),h->denoiser.session,"prompt",h->promptCfg.data(),kPromptCfg)
      && mobilei2v::copyFloatInput(h->denoiser.interpreter.get(),h->denoiser.session,"cond_mask",h->condMask.data(),kCondMask)
      && mobilei2v::copyUint8Input(h->denoiser.interpreter.get(),h->denoiser.session,"text_mask",h->textMask.data(),kTextMask)
      && mobilei2v::copyFloatInput(h->denoiser.interpreter.get(),h->denoiser.session,"flow_score",flow,kFlowScore);
    env->ReleaseFloatArrayElements(latentValue,latent,JNI_ABORT);env->ReleaseFloatArrayElements(flowValue,flow,JNI_ABORT); if(!ok)return -6;
    int code=h->denoiser.interpreter->runSession(h->denoiser.session); if(code!=0){__android_log_print(ANDROID_LOG_ERROR,kTag,"denoiser runSession failed %d",code);return -7;}
    jfloat* out=env->GetFloatArrayElements(outputValue,nullptr);if(!out)return -8;
    ok=mobilei2v::copyFloatOutput(h->denoiser.interpreter.get(),h->denoiser.session,"output",out,kLatentCfg);
    env->ReleaseFloatArrayElements(outputValue,out,ok?0:JNI_ABORT);return ok?0:-9;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeDecode(JNIEnv* env,jclass,jlong id,jfloatArray latentValue) {
    std::lock_guard<std::mutex> guard(gMutex); RuntimeHandle* h=get(id); if(!h)return -1;
    if(!latentValue||env->GetArrayLength(latentValue)!=static_cast<jsize>(kSingleLatent))return -2;
    jfloat* latent=env->GetFloatArrayElements(latentValue,nullptr);if(!latent)return -3;
    bool ok=mobilei2v::copyFloatInput(h->decoder.interpreter.get(),h->decoder.session,"latent",latent,kSingleLatent);
    env->ReleaseFloatArrayElements(latentValue,latent,JNI_ABORT);if(!ok)return -4;
    int code=h->decoder.interpreter->runSession(h->decoder.session);if(code!=0)return -5;
    h->decoded.assign(kDecodedVideo,0.0f);
    ok=mobilei2v::copyFloatOutput(h->decoder.interpreter.get(),h->decoder.session,"video",h->decoded.data(),kDecodedVideo);
    if(!ok){h->decoded.clear();return -6;} return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeCopyDecodedFrameArgb(JNIEnv* env,jclass,jlong id,jint frame,jintArray argbValue) {
    std::lock_guard<std::mutex> guard(gMutex); RuntimeHandle* h=get(id);if(!h)return -1;
    if(h->decoded.size()!=kDecodedVideo)return -2; if(frame<0||frame>=static_cast<jint>(kVideoFrames))return -3;
    if(!argbValue||env->GetArrayLength(argbValue)!=static_cast<jsize>(kFramePixels))return -4;
    jint* argb=env->GetIntArrayElements(argbValue,nullptr);if(!argb)return -5;
    const size_t t=static_cast<size_t>(frame); const size_t plane=kHeight*kWidth;
    for(size_t i=0;i<plane;i++){
        int rgb[3];
        for(size_t c=0;c<3;c++){
            const size_t idx=(c*kVideoFrames+t)*plane+i;
            float v=std::max(-1.0f,std::min(1.0f,h->decoded[idx]));
            rgb[c]=static_cast<int>(std::lround((v+1.0f)*127.5f));
        }
        argb[i]=static_cast<jint>(0xff000000u | (static_cast<uint32_t>(rgb[0])<<16u) | (static_cast<uint32_t>(rgb[1])<<8u) | static_cast<uint32_t>(rgb[2]));
    }
    env->ReleaseIntArrayElements(argbValue,argb,0);return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeClearDecoded(JNIEnv*,jclass,jlong id){std::lock_guard<std::mutex> guard(gMutex);if(auto* h=get(id)){std::vector<float>().swap(h->decoded);}}

extern "C" JNIEXPORT void JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeRelease(JNIEnv*,jclass,jlong id){std::lock_guard<std::mutex> guard(gMutex);gHandles.erase(id);}
