#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "mnn_runtime.hpp"

#if !defined(__cpp_exceptions)
#error "MobileI2V JNI requires C++ exceptions for its native exception boundaries"
#endif

namespace {
constexpr const char* kTag = "MobileI2VGpu";
constexpr size_t kW = 1280, kH = 720, kT = 17;
constexpr size_t kSingle = 128 * 3 * 23 * 40, kGuide = 128 * 23 * 40;
constexpr size_t kCfg = 2 * kSingle, kImage = 3 * kH * kW;
constexpr size_t kCond = 2760, kGuidePlane = 23 * 40;
constexpr size_t kDecoded = 3 * kT * kH * kW, kPixels = kH * kW;
using mobilei2v::FloatPort;
constexpr FloatPort kEncoderImage{"image", {1, 3, 720, 1280}, 4, kImage};
constexpr FloatPort kEncoderEpsilon{"posterior_epsilon", {1, 128, 1, 23, 40}, 5, kGuide};
constexpr FloatPort kEncoderOutput{"guide", {1, 128, 1, 23, 40}, 5, kGuide};
constexpr FloatPort kDenoiserLatent{"latent", {2, 128, 3, 23, 40}, 5, kCfg};
constexpr FloatPort kTimestep{"timestep", {2}, 1, 2};
constexpr FloatPort kCondMask{"cond_mask", {1, 2760}, 2, kCond};
constexpr FloatPort kFlowScore{"flow_score", {2}, 1, 2};
constexpr FloatPort kDenoiserOutput{"output", {2, 128, 3, 23, 40}, 5, kCfg};
constexpr FloatPort kDecoderLatent{"latent", {1, 128, 3, 23, 40}, 5, kSingle};
constexpr FloatPort kDecoderOutput{"video", {1, 3, 17, 720, 1280}, 5, kDecoded};
static_assert(sizeof(jfloat) == sizeof(float) && sizeof(float) == 4, "FP32 JNI required");
static_assert(kDecoded <= static_cast<size_t>(std::numeric_limits<jsize>::max()), "JNI array size");

struct RuntimeHandle {
    mobilei2v::OpenClSession encoder, denoiser, decoder;
    std::vector<float> cond, decoded;
    bool usable = true;
};
std::mutex gMutex;
std::unordered_map<jlong, std::unique_ptr<RuntimeHandle>> gHandles;
jlong gNext = 1;
thread_local char gLastError[2048]{};

void recordError(const char* operation, const char* message) noexcept {
    std::snprintf(gLastError, sizeof(gLastError), "%s: %s", operation, message);
    __android_log_print(ANDROID_LOG_ERROR, kTag, "%s", gLastError);
}
jint failure(jint code, const char* operation, const std::string& error) {
    recordError(operation, error.c_str());
    return code;
}

template <typename T, typename F>
T boundary(JNIEnv* env, const char* operation, T failed, F&& action) noexcept {
    if (env->ExceptionCheck()) return failed;
    gLastError[0] = '\0';
    try {
        return action();
    } catch (const std::exception& ex) {
        recordError(operation, ex.what());
    } catch (...) {
        recordError(operation, "unknown C++ exception");
    }
    return failed;
}

std::string modelDir(JNIEnv* env, jstring value) {
    if (!value) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};
    struct UtfChars {
        JNIEnv* env;
        jstring value;
        const char* chars;
        ~UtfChars() { env->ReleaseStringUTFChars(value, chars); }
    } release{env, value, chars};
    return std::string(chars);
}

class FloatElements {
public:
    FloatElements(JNIEnv* env, jfloatArray array)
        : env_(env), array_(array), values_(env->GetFloatArrayElements(array, nullptr)) {}
    ~FloatElements() {
        if (values_) env_->ReleaseFloatArrayElements(array_, values_, JNI_ABORT);
    }
    FloatElements(const FloatElements&) = delete;
    FloatElements& operator=(const FloatElements&) = delete;
    const float* data() const { return values_; }
private:
    JNIEnv* env_;
    jfloatArray array_;
    jfloat* values_;
};

bool arrayLength(JNIEnv* env, jarray array, size_t count) {
    if (!array || env->ExceptionCheck() || count > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
        return false;
    }
    const auto length = env->GetArrayLength(array);
    return !env->ExceptionCheck() && length == static_cast<jsize>(count);
}

bool validatePorts(mobilei2v::OpenClSession& graph,
        std::initializer_list<FloatPort> inputs, const FloatPort& output, std::string& error) {
    if (!mobilei2v::validateOpenClSession(graph, error)) return false;
    const auto& actualInputs = graph.interpreter->getSessionInputAll(graph.session);
    const auto& actualOutputs = graph.interpreter->getSessionOutputAll(graph.session);
    if (actualInputs.size() != inputs.size() || actualOutputs.size() != 1) {
        return mobilei2v::fail(error, "unexpected graph input/output ports");
    }
    for (const auto& port : inputs) {
        const auto found = actualInputs.find(port.name);
        auto* tensor = found == actualInputs.end() ? nullptr : found->second;
        if (!mobilei2v::validateFloatTensor(tensor, port, error) ||
                !mobilei2v::checkDeviceTensor(graph, tensor, port.name, error, true)) return false;
    }
    const auto found = actualOutputs.find(output.name);
    auto* tensor = found == actualOutputs.end() ? nullptr : found->second;
    return mobilei2v::validateFloatTensor(tensor, output, error) &&
            mobilei2v::checkDeviceTensor(graph, tensor, output.name, error);
}

bool encoderPorts(RuntimeHandle& handle, std::string& error) {
    return validatePorts(handle.encoder, {kEncoderImage, kEncoderEpsilon}, kEncoderOutput, error);
}
bool denoiserPorts(RuntimeHandle& handle, std::string& error) {
    return validatePorts(handle.denoiser,
            {kDenoiserLatent, kTimestep, kCondMask, kFlowScore}, kDenoiserOutput, error);
}
bool decoderPorts(RuntimeHandle& handle, std::string& error) {
    return validatePorts(handle.decoder, {kDecoderLatent}, kDecoderOutput, error);
}

std::unique_ptr<RuntimeHandle> create(const std::string& dir, std::string& error) {
    if (dir.empty()) {
        mobilei2v::fail(error, "empty model directory");
        return nullptr;
    }
    auto handle = std::make_unique<RuntimeHandle>();
    handle->cond.assign(kCond, 0.f);
    std::fill_n(handle->cond.begin(), kGuidePlane, 1.f);
    handle->encoder = mobilei2v::createOpenClSession(
            dir + "/vae_encoder.mnn", dir, "vae-encoder-fp32-opencl.cache", error);
    if (!handle->encoder || !encoderPorts(*handle, error) ||
            !mobilei2v::auditOpenClExecutions(handle->encoder, error)) {
        error = "encoder: " + error;
        return nullptr;
    }
    handle->denoiser = mobilei2v::createOpenClSession(
            dir + "/denoiser.mnn", dir, "denoiser-fp32-opencl.cache", error);
    if (!handle->denoiser || !denoiserPorts(*handle, error) ||
            !mobilei2v::auditOpenClExecutions(handle->denoiser, error)) {
        error = "denoiser: " + error;
        return nullptr;
    }
    handle->decoder = mobilei2v::createOpenClSession(
            dir + "/vae_decoder.mnn", dir, "vae-decoder-fp32-opencl.cache", error);
    if (!handle->decoder || !decoderPorts(*handle, error) ||
            !mobilei2v::auditOpenClExecutions(handle->decoder, error)) {
        error = "decoder: " + error;
        return nullptr;
    }
    return handle;
}

RuntimeHandle* get(jlong id) {
    const auto found = gHandles.find(id);
    return found == gHandles.end() ? nullptr : found->second.get();
}

// The global mutex protects both a borrowed handle and all its MNN resources.
// A failed/throwing inference poisons the handle; release is still always safe.
struct InferenceGuard {
    RuntimeHandle& handle;
    bool success = false;
    ~InferenceGuard() {
        if (!success) {
            handle.usable = false;
            handle.decoded.clear();
        }
    }
};
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeLastError(JNIEnv* env, jclass) {
    if (env->ExceptionCheck()) return nullptr;
    return env->NewStringUTF(gLastError);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeProbe(JNIEnv* env, jclass, jstring dirValue) {
    return boundary<jstring>(env, "probe", nullptr, [&]() -> jstring {
        std::lock_guard<std::mutex> lock(gMutex);
        const auto dir = modelDir(env, dirValue);
        if (env->ExceptionCheck()) return nullptr;
        std::string error;
        auto handle = create(dir, error);
        if (!handle) {
            recordError("probe", error.c_str());
            return env->NewStringUTF(("NOT_READY:" + error).c_str());
        }
        // This is a runtime contract/backend audit, never numerical qualification.
        return env->NewStringUTF("MNN_OPENCL_CONTRACT_VALID:FP32;execution-backends-audited;"
                "numerical-qualification-unverified");
    });
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeLoad(JNIEnv* env, jclass, jstring dirValue) {
    return boundary<jlong>(env, "load", 0, [&]() -> jlong {
        std::lock_guard<std::mutex> lock(gMutex);
        const auto dir = modelDir(env, dirValue);
        if (env->ExceptionCheck()) return 0;
        std::string error;
        auto handle = create(dir, error);  // No dependence on a prior probe.
        if (!handle) {
            recordError("load", error.c_str());
            return 0;
        }
        if (gNext <= 0) {
            recordError("load", "native handle IDs exhausted");
            return 0;
        }
        const jlong id = gNext;
        gNext = id == std::numeric_limits<jlong>::max() ? 0 : id + 1;
        if (!gHandles.emplace(id, std::move(handle)).second) {
            recordError("load", "native handle collision");
            return 0;
        }
        return id;
    });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeEncode(
        JNIEnv* env, jclass, jlong id, jfloatArray image, jfloatArray epsilon, jfloatArray guide) {
    return boundary<jint>(env, "encode", -100, [&]() -> jint {
        std::lock_guard<std::mutex> lock(gMutex);
        auto* handle = get(id);
        if (!handle || !handle->usable) return failure(-1, "encode", "invalid/failed handle");
        if (!arrayLength(env, image, kImage) || !arrayLength(env, epsilon, kGuide) ||
                !arrayLength(env, guide, kGuide)) return failure(-2, "encode", "array length mismatch");
        std::string error;
        InferenceGuard guard{*handle};
        if (!encoderPorts(*handle, error)) return failure(-4, "encode", error);
        {
            FloatElements input(env, image);
            if (!input.data()) return failure(-3, "encode", "cannot access image array");
            FloatElements noise(env, epsilon);
            if (!noise.data()) return failure(-3, "encode", "cannot access posterior_epsilon array");
            if (!mobilei2v::copyFloatInput(handle->encoder, kEncoderImage, input.data(), kImage, error) ||
                    !mobilei2v::copyFloatInput(handle->encoder, kEncoderEpsilon, noise.data(), kGuide, error)) {
                return failure(-4, "encode", error);
            }
        }
        if (!mobilei2v::runOpenClSession(handle->encoder, error)) return failure(-4, "encode", error);
        std::vector<float> output(kGuide);
        if (!mobilei2v::copyFloatOutput(handle->encoder, kEncoderOutput, output.data(), output.size(), error)) {
            return failure(-6, "encode", error);
        }
        env->SetFloatArrayRegion(guide, 0, static_cast<jsize>(kGuide), output.data());
        if (env->ExceptionCheck()) return failure(-5, "encode", "cannot write guide array");
        guard.success = true;
        return 0;
    });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeRunDenoiser(
        JNIEnv* env, jclass, jlong id, jfloatArray latent, jfloat timestep,
        jfloatArray flow, jfloatArray outputArray) {
    return boundary<jint>(env, "denoiser", -100, [&]() -> jint {
        std::lock_guard<std::mutex> lock(gMutex);
        auto* handle = get(id);
        if (!handle || !handle->usable) return failure(-1, "denoiser", "invalid/failed handle");
        if (!arrayLength(env, latent, kCfg) || !arrayLength(env, flow, 2) ||
                !arrayLength(env, outputArray, kCfg) || !std::isfinite(timestep)) {
            return failure(-2, "denoiser", "array length mismatch/non-finite timestep");
        }
        std::string error;
        InferenceGuard guard{*handle};
        if (!denoiserPorts(*handle, error)) return failure(-4, "denoiser", error);
        {
            FloatElements input(env, latent);
            if (!input.data()) return failure(-3, "denoiser", "cannot access latent array");
            FloatElements score(env, flow);
            if (!score.data()) return failure(-3, "denoiser", "cannot access flow_score array");
            const float time[2] = {timestep, timestep};
            if (!mobilei2v::copyFloatInput(handle->denoiser, kDenoiserLatent, input.data(), kCfg, error) ||
                    !mobilei2v::copyFloatInput(handle->denoiser, kTimestep, time, 2, error) ||
                    !mobilei2v::copyFloatInput(handle->denoiser, kCondMask, handle->cond.data(), kCond, error) ||
                    !mobilei2v::copyFloatInput(handle->denoiser, kFlowScore, score.data(), 2, error)) {
                return failure(-4, "denoiser", error);
            }
        }
        if (!mobilei2v::runOpenClSession(handle->denoiser, error)) return failure(-5, "denoiser", error);
        std::vector<float> output(kCfg);
        if (!mobilei2v::copyFloatOutput(handle->denoiser, kDenoiserOutput, output.data(), output.size(), error)) {
            return failure(-7, "denoiser", error);
        }
        env->SetFloatArrayRegion(outputArray, 0, static_cast<jsize>(kCfg), output.data());
        if (env->ExceptionCheck()) return failure(-6, "denoiser", "cannot write output array");
        guard.success = true;
        return 0;
    });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeDecode(
        JNIEnv* env, jclass, jlong id, jfloatArray latent) {
    return boundary<jint>(env, "decode", -100, [&]() -> jint {
        std::lock_guard<std::mutex> lock(gMutex);
        auto* handle = get(id);
        if (!handle || !handle->usable) return failure(-1, "decode", "invalid/failed handle");
        handle->decoded.clear();  // Never expose an older video's frames on failure.
        if (!arrayLength(env, latent, kSingle)) return failure(-2, "decode", "array length mismatch");
        std::string error;
        InferenceGuard guard{*handle};
        if (!decoderPorts(*handle, error)) return failure(-4, "decode", error);
        {
            FloatElements input(env, latent);
            if (!input.data()) return failure(-3, "decode", "cannot access latent array");
            if (!mobilei2v::copyFloatInput(handle->decoder, kDecoderLatent, input.data(), kSingle, error)) {
                return failure(-4, "decode", error);
            }
        }
        if (!mobilei2v::runOpenClSession(handle->decoder, error)) return failure(-4, "decode", error);
        handle->decoded.resize(kDecoded);
        if (!mobilei2v::copyFloatOutput(handle->decoder, kDecoderOutput,
                handle->decoded.data(), handle->decoded.size(), error)) return failure(-5, "decode", error);
        guard.success = true;
        return 0;
    });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeCopyDecodedFrameArgb(
        JNIEnv* env, jclass, jlong id, jint frame, jintArray pixels) {
    return boundary<jint>(env, "frame", -100, [&]() -> jint {
        std::lock_guard<std::mutex> lock(gMutex);
        auto* handle = get(id);
        if (!handle || !handle->usable || handle->decoded.size() != kDecoded) {
            return failure(-1, "frame", "no valid decoded video");
        }
        if (frame < 0 || frame >= static_cast<jint>(kT) || !arrayLength(env, pixels, kPixels)) {
            return failure(-2, "frame", "invalid frame/ARGB array length");
        }
        std::vector<jint> argb(kPixels);
        const size_t time = static_cast<size_t>(frame);
        for (size_t i = 0; i < kPixels; ++i) {
            uint32_t pixel = 0xff000000u;
            for (size_t c = 0; c < 3; ++c) {
                // Export already top-crops to [1,3,17,720,1280]. No native crop.
                const float value = handle->decoded[(c * kT + time) * kPixels + i];
                if (!std::isfinite(value)) {
                    handle->decoded.clear();
                    handle->usable = false;
                    return failure(-4, "frame", "non-finite decoded sample");
                }
                const auto channel = static_cast<uint32_t>(
                        std::lround((std::clamp(value, -1.f, 1.f) + 1.f) * 127.5f));
                pixel |= channel << (16u - static_cast<unsigned>(c) * 8u);
            }
            argb[i] = static_cast<jint>(pixel);
        }
        env->SetIntArrayRegion(pixels, 0, static_cast<jsize>(kPixels), argb.data());
        if (env->ExceptionCheck()) return failure(-3, "frame", "cannot write ARGB array");
        return 0;
    });
}

extern "C" JNIEXPORT void JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeClearDecoded(JNIEnv* env, jclass, jlong id) {
    boundary<jint>(env, "clear decoded", -100, [&]() -> jint {
        std::lock_guard<std::mutex> lock(gMutex);
        if (auto* handle = get(id)) std::vector<float>().swap(handle->decoded);
        return 0;
    });
}

extern "C" JNIEXPORT void JNICALL
Java_com_qujindai_localvideo_MobileI2VGpuNative_nativeRelease(JNIEnv* env, jclass, jlong id) {
    boundary<jint>(env, "release", -100, [&]() -> jint {
        std::lock_guard<std::mutex> lock(gMutex);
        gHandles.erase(id);
        return 0;
    });
}
