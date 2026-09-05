#include <jni.h>
#include <android/log.h>

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
constexpr size_t kSingleLatentFloats = 128u * 3u * 23u * 40u;
constexpr size_t kLatentCfgFloats = 2u * kSingleLatentFloats;
constexpr size_t kPromptCfgFloats = 2u * 1u * 300u * 896u;
constexpr size_t kCondMaskFloats = 3u * 23u * 40u;
constexpr size_t kGuideMaskFloats = 23u * 40u;
constexpr size_t kTextMaskBytes = 300u;
constexpr size_t kFlowScoreFloats = 2u;

struct RuntimeHandle {
    mobilei2v::OpenClSession denoiser;
    std::string modelDir;
    std::vector<float> promptCfg;
    std::vector<uint8_t> textMask;
    std::vector<float> condMask;
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

float halfToFloat(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16u;
    uint32_t exponent = (value >> 10u) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1u;
                ++shift;
            }
            mantissa &= 0x03ffu;
            const uint32_t exp32 = static_cast<uint32_t>(127 - 15 - shift + 1);
            bits = sign | (exp32 << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        const uint32_t exp32 = exponent + (127u - 15u);
        bits = sign | (exp32 << 23u) | (mantissa << 13u);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

bool readPromptF16(const std::string& path, std::vector<float>& output) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return false;
    const std::streamsize bytes = stream.tellg();
    if (bytes != static_cast<std::streamsize>(kPromptCfgFloats * sizeof(uint16_t))) return false;
    stream.seekg(0, std::ios::beg);
    std::vector<uint16_t> half(kPromptCfgFloats);
    if (!stream.read(reinterpret_cast<char*>(half.data()), bytes)) return false;
    output.resize(kPromptCfgFloats);
    for (size_t i = 0; i < kPromptCfgFloats; ++i) {
        output[i] = halfToFloat(half[i]);
        if (!std::isfinite(output[i])) return false;
    }
    return true;
}

bool readTextMask(const std::string& path, std::vector<uint8_t>& output) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return false;
    const std::streamsize bytes = stream.tellg();
    if (bytes != static_cast<std::streamsize>(kTextMaskBytes)) return false;
    stream.seekg(0, std::ios::beg);
    output.resize(kTextMaskBytes);
    return static_cast<bool>(stream.read(reinterpret_cast<char*>(output.data()), bytes));
}

std::vector<float> buildCondMask() {
    std::vector<float> result(kCondMaskFloats, 0.0f);
    std::fill(result.begin(), result.begin() + kGuideMaskFloats, 1.0f);
    return result;
}

bool hasExpectedDenoiserIo(mobilei2v::OpenClSession& runtime) {
    if (!runtime) return false;
    auto* i = runtime.interpreter.get();
    auto* s = runtime.session;
    auto* latent = i->getSessionInput(s, "latent");
    auto* timestep = i->getSessionInput(s, "timestep");
    auto* prompt = i->getSessionInput(s, "prompt");
    auto* condMask = i->getSessionInput(s, "cond_mask");
    auto* textMask = i->getSessionInput(s, "text_mask");
    auto* flow = i->getSessionInput(s, "flow_score");
    auto* output = i->getSessionOutput(s, "output");
    return latent && timestep && prompt && condMask && textMask && flow && output
            && static_cast<size_t>(latent->elementSize()) == kLatentCfgFloats
            && static_cast<size_t>(timestep->elementSize()) == 2u
            && static_cast<size_t>(prompt->elementSize()) == kPromptCfgFloats
            && static_cast<size_t>(condMask->elementSize()) == kCondMaskFloats
            && static_cast<size_t>(textMask->elementSize()) == kTextMaskBytes
            && static_cast<size_t>(flow->elementSize()) == kFlowScoreFloats
            && static_cast<size_t>(output->elementSize()) == kLatentCfgFloats;
}

std::unique_ptr<RuntimeHandle> createHandle(const std::string& modelDir) {
    auto handle = std::make_unique<RuntimeHandle>();
    handle->modelDir = modelDir;
    if (!readPromptF16(modelDir + "/empty_prompt.f16", handle->promptCfg)) return nullptr;
    if (!readTextMask(modelDir + "/empty_prompt_mask.bin", handle->textMask)) return nullptr;
    handle->condMask = buildCondMask();
    handle->denoiser = mobilei2v::createOpenClSession(
            modelDir + "/denoiser.mnn", modelDir, "denoiser-opencl.cache");
    if (!handle->denoiser || !hasExpectedDenoiserIo(handle->denoiser)) return nullptr;
    return handle;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeProbe(
        JNIEnv* env, jclass, jstring modelDirValue) {
    const std::string modelDir = toString(env, modelDirValue);
    if (modelDir.empty()) return newString(env, "NOT_READY:empty model directory");
    auto handle = createHandle(modelDir);
    if (!handle) return newString(env, "NOT_READY:MNN OpenCL model/conditioning contract failed");
    return newString(env, handle->denoiser.cacheFile.empty()
            ? "MNN_OPENCL_READY:cache-disabled"
            : "MNN_OPENCL_READY:cache-enabled");
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeLoad(
        JNIEnv* env, jclass, jstring modelDirValue) {
    const std::string modelDir = toString(env, modelDirValue);
    if (modelDir.empty()) return 0;
    auto handle = createHandle(modelDir);
    if (!handle) return 0;

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
        jfloatArray flowScoreValue,
        jfloatArray outputValue) {
    std::lock_guard<std::mutex> guard(gMutex);
    auto found = gHandles.find(handleId);
    if (found == gHandles.end()) return -1;
    RuntimeHandle* handle = found->second.get();
    auto& runtime = handle->denoiser;
    if (!runtime) return -2;

    if (!latentValue || env->GetArrayLength(latentValue) != static_cast<jsize>(kLatentCfgFloats)) return -3;
    if (!flowScoreValue || env->GetArrayLength(flowScoreValue) != static_cast<jsize>(kFlowScoreFloats)) return -4;
    if (!outputValue || env->GetArrayLength(outputValue) != static_cast<jsize>(kLatentCfgFloats)) return -5;

    jfloat* latent = env->GetFloatArrayElements(latentValue, nullptr);
    jfloat* flow = env->GetFloatArrayElements(flowScoreValue, nullptr);
    if (!latent || !flow) {
        if (latent) env->ReleaseFloatArrayElements(latentValue, latent, JNI_ABORT);
        if (flow) env->ReleaseFloatArrayElements(flowScoreValue, flow, JNI_ABORT);
        return -6;
    }

    const float timesteps[2] = {timestepValue, timestepValue};
    bool ok = mobilei2v::copyFloatInput(runtime.interpreter.get(), runtime.session,
            "latent", latent, kLatentCfgFloats);
    ok = ok && mobilei2v::copyFloatInput(runtime.interpreter.get(), runtime.session,
            "timestep", timesteps, 2);
    ok = ok && mobilei2v::copyFloatInput(runtime.interpreter.get(), runtime.session,
            "prompt", handle->promptCfg.data(), kPromptCfgFloats);
    ok = ok && mobilei2v::copyFloatInput(runtime.interpreter.get(), runtime.session,
            "cond_mask", handle->condMask.data(), kCondMaskFloats);
    ok = ok && mobilei2v::copyUint8Input(runtime.interpreter.get(), runtime.session,
            "text_mask", handle->textMask.data(), kTextMaskBytes);
    ok = ok && mobilei2v::copyFloatInput(runtime.interpreter.get(), runtime.session,
            "flow_score", flow, kFlowScoreFloats);

    env->ReleaseFloatArrayElements(latentValue, latent, JNI_ABORT);
    env->ReleaseFloatArrayElements(flowScoreValue, flow, JNI_ABORT);
    if (!ok) return -7;

    const int code = runtime.interpreter->runSession(runtime.session);
    if (code != 0) {
        __android_log_print(ANDROID_LOG_ERROR, kTag, "MNN OpenCL runSession failed: %d", code);
        return -8;
    }

    jfloat* output = env->GetFloatArrayElements(outputValue, nullptr);
    if (!output) return -9;
    ok = mobilei2v::copyFloatOutput(runtime.interpreter.get(), runtime.session,
            "output", output, kLatentCfgFloats);
    env->ReleaseFloatArrayElements(outputValue, output, ok ? 0 : JNI_ABORT);
    return ok ? 0 : -10;
}

extern "C" JNIEXPORT void JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeRelease(
        JNIEnv*, jclass, jlong handleId) {
    std::lock_guard<std::mutex> guard(gMutex);
    gHandles.erase(handleId);
}
