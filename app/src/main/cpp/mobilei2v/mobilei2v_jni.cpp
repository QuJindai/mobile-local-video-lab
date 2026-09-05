#include <jni.h>
#include <android/log.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "mnn_runtime.hpp"

namespace {
constexpr const char* kTag = "MobileI2VGpu";
constexpr size_t kLatentCfgFloats = 2u * 128u * 3u * 23u * 40u;
constexpr size_t kPromptCfgFloats = 2u * 1u * 300u * 896u;
constexpr size_t kPromptMaskBytes = 2u * 300u;
constexpr size_t kFlowScoreFloats = 2u;

struct RuntimeHandle {
    mobilei2v::OpenClSession denoiser;
    std::string modelDir;
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

jstring newString(JNIEnv* env, const std::string& value) {
    return env->NewStringUTF(value.c_str());
}

bool hasExpectedDenoiserIo(mobilei2v::OpenClSession& runtime) {
    if (!runtime) return false;
    auto* i = runtime.interpreter.get();
    auto* s = runtime.session;
    auto* latent = i->getSessionInput(s, "latent");
    auto* timestep = i->getSessionInput(s, "timestep");
    auto* prompt = i->getSessionInput(s, "prompt");
    auto* mask = i->getSessionInput(s, "mask");
    auto* flow = i->getSessionInput(s, "flow_score");
    auto* output = i->getSessionOutput(s, "output");
    return latent && timestep && prompt && mask && flow && output
            && static_cast<size_t>(latent->elementSize()) == kLatentCfgFloats
            && static_cast<size_t>(prompt->elementSize()) == kPromptCfgFloats
            && static_cast<size_t>(mask->elementSize()) == kPromptMaskBytes
            && static_cast<size_t>(flow->elementSize()) == kFlowScoreFloats
            && static_cast<size_t>(output->elementSize()) == kLatentCfgFloats;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeProbe(
        JNIEnv* env, jclass, jstring modelDirValue) {
    const std::string modelDir = toString(env, modelDirValue);
    if (modelDir.empty()) return newString(env, "NOT_READY:empty model directory");
    const std::string denoiser = modelDir + "/denoiser.mnn";
    auto runtime = mobilei2v::createOpenClSession(
            denoiser, modelDir, "denoiser-opencl.cache");
    if (!runtime) return newString(env, "NOT_READY:MNN OpenCL session creation failed");
    if (!hasExpectedDenoiserIo(runtime)) {
        return newString(env, "NOT_READY:denoiser tensor contract mismatch");
    }
    return newString(env, runtime.cacheFile.empty()
            ? "MNN_OPENCL_READY:cache-disabled"
            : "MNN_OPENCL_READY:cache-enabled");
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeLoad(
        JNIEnv* env, jclass, jstring modelDirValue) {
    const std::string modelDir = toString(env, modelDirValue);
    if (modelDir.empty()) return 0;
    auto handle = std::make_unique<RuntimeHandle>();
    handle->modelDir = modelDir;
    handle->denoiser = mobilei2v::createOpenClSession(
            modelDir + "/denoiser.mnn", modelDir, "denoiser-opencl.cache");
    if (!handle->denoiser || !hasExpectedDenoiserIo(handle->denoiser)) return 0;

    std::lock_guard<std::mutex> guard(gMutex);
    const jlong id = gNextHandle++;
    gHandles.emplace(id, std::move(handle));
    return id;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeRunDenoiser(
        JNIEnv* env,
        jclass,
        jlong handleId,
        jfloatArray latentValue,
        jfloat timestepValue,
        jfloatArray promptValue,
        jbyteArray maskValue,
        jfloatArray flowScoreValue,
        jfloatArray outputValue) {
    std::lock_guard<std::mutex> guard(gMutex);
    auto found = gHandles.find(handleId);
    if (found == gHandles.end()) return -1;
    RuntimeHandle* handle = found->second.get();
    auto& runtime = handle->denoiser;
    if (!runtime) return -2;

    if (!latentValue || env->GetArrayLength(latentValue) != static_cast<jsize>(kLatentCfgFloats)) return -3;
    if (!promptValue || env->GetArrayLength(promptValue) != static_cast<jsize>(kPromptCfgFloats)) return -4;
    if (!maskValue || env->GetArrayLength(maskValue) != static_cast<jsize>(kPromptMaskBytes)) return -5;
    if (!flowScoreValue || env->GetArrayLength(flowScoreValue) != static_cast<jsize>(kFlowScoreFloats)) return -6;
    if (!outputValue || env->GetArrayLength(outputValue) != static_cast<jsize>(kLatentCfgFloats)) return -7;

    jfloat* latent = env->GetFloatArrayElements(latentValue, nullptr);
    jfloat* prompt = env->GetFloatArrayElements(promptValue, nullptr);
    jbyte* mask = env->GetByteArrayElements(maskValue, nullptr);
    jfloat* flow = env->GetFloatArrayElements(flowScoreValue, nullptr);
    if (!latent || !prompt || !mask || !flow) {
        if (latent) env->ReleaseFloatArrayElements(latentValue, latent, JNI_ABORT);
        if (prompt) env->ReleaseFloatArrayElements(promptValue, prompt, JNI_ABORT);
        if (mask) env->ReleaseByteArrayElements(maskValue, mask, JNI_ABORT);
        if (flow) env->ReleaseFloatArrayElements(flowScoreValue, flow, JNI_ABORT);
        return -8;
    }

    const float timesteps[2] = {timestepValue, timestepValue};
    bool ok = mobilei2v::copyFloatInput(runtime.interpreter.get(), runtime.session,
            "latent", latent, kLatentCfgFloats);
    ok = ok && mobilei2v::copyFloatInput(runtime.interpreter.get(), runtime.session,
            "timestep", timesteps, 2);
    ok = ok && mobilei2v::copyFloatInput(runtime.interpreter.get(), runtime.session,
            "prompt", prompt, kPromptCfgFloats);
    ok = ok && mobilei2v::copyUint8Input(runtime.interpreter.get(), runtime.session,
            "mask", reinterpret_cast<uint8_t*>(mask), kPromptMaskBytes);
    ok = ok && mobilei2v::copyFloatInput(runtime.interpreter.get(), runtime.session,
            "flow_score", flow, kFlowScoreFloats);

    env->ReleaseFloatArrayElements(latentValue, latent, JNI_ABORT);
    env->ReleaseFloatArrayElements(promptValue, prompt, JNI_ABORT);
    env->ReleaseByteArrayElements(maskValue, mask, JNI_ABORT);
    env->ReleaseFloatArrayElements(flowScoreValue, flow, JNI_ABORT);
    if (!ok) return -9;

    const int code = runtime.interpreter->runSession(runtime.session);
    if (code != 0) {
        __android_log_print(ANDROID_LOG_ERROR, kTag, "MNN OpenCL runSession failed: %d", code);
        return -10;
    }

    jfloat* output = env->GetFloatArrayElements(outputValue, nullptr);
    if (!output) return -11;
    ok = mobilei2v::copyFloatOutput(runtime.interpreter.get(), runtime.session,
            "output", output, kLatentCfgFloats);
    env->ReleaseFloatArrayElements(outputValue, output, ok ? 0 : JNI_ABORT);
    return ok ? 0 : -12;
}

extern "C" JNIEXPORT void JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeRelease(
        JNIEnv*, jclass, jlong handleId) {
    std::lock_guard<std::mutex> guard(gMutex);
    gHandles.erase(handleId);
}
