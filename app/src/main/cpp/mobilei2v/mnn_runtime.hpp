#pragma once

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>

namespace mobilei2v {

// Dream-derived behavior: load large model files through mmap so the source
// pages remain file-backed/reclaimable instead of creating a second Java/native
// anonymous copy during startup.
inline std::unique_ptr<MNN::Interpreter> createInterpreterMmap(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return nullptr;
    }
    const size_t size = static_cast<size_t>(st.st_size);
    void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) return nullptr;
    madvise(mapped, size, MADV_SEQUENTIAL);
    auto* raw = MNN::Interpreter::createFromBuffer(mapped, size);
    munmap(mapped, size);
    if (!raw) return nullptr;
    raw->setExternalFile((path + ".weight").c_str());
    return std::unique_ptr<MNN::Interpreter>(raw);
}

inline std::string ensureCacheDir(const std::string& modelDir) {
    if (modelDir.empty()) return {};
    std::error_code ec;
    std::filesystem::path cache = std::filesystem::path(modelDir) / "cache";
    std::filesystem::create_directories(cache, ec);
    return ec ? std::string{} : cache.string();
}

struct OpenClSession {
    std::unique_ptr<MNN::Interpreter> interpreter;
    MNN::Session* session = nullptr;
    std::string cacheFile;

    explicit operator bool() const { return interpreter && session; }
};

// Production MobileI2V creates accelerator sessions only. If the OpenCL
// backend cannot be created, readiness fails and Java keeps the model NOT READY.
inline OpenClSession createOpenClSession(
        const std::string& modelPath,
        const std::string& modelDir,
        const std::string& cacheName) {
    OpenClSession result;
    result.interpreter = createInterpreterMmap(modelPath);
    if (!result.interpreter) return result;

    const std::string cacheDir = ensureCacheDir(modelDir);
    if (!cacheDir.empty()) {
        result.cacheFile = (std::filesystem::path(cacheDir) / cacheName).string();
        result.interpreter->setCacheFile(result.cacheFile.c_str());
    }

    MNN::ScheduleConfig config;
    MNN::BackendConfig backend;
    config.type = MNN_FORWARD_OPENCL;
    config.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST;
    backend.precision = MNN::BackendConfig::Precision_Low;
    backend.power = MNN::BackendConfig::Power_High;
    backend.memory = MNN::BackendConfig::Memory_Low;
    config.backendConfig = &backend;

    result.session = result.interpreter->createSession(config);
    if (!result.session) {
        result.interpreter.reset();
        return result;
    }
    // Session owns compiled graph resources; release raw model buffer after
    // successful compilation to reduce peak memory on the 12-GB handset.
    result.interpreter->releaseModel();
    return result;
}

inline bool copyFloatInput(
        MNN::Interpreter* interpreter,
        MNN::Session* session,
        const char* name,
        const float* values,
        size_t count) {
    if (!interpreter || !session || !values) return false;
    MNN::Tensor* device = interpreter->getSessionInput(session, name);
    if (!device || static_cast<size_t>(device->elementSize()) != count) return false;
    std::unique_ptr<MNN::Tensor> host(
            MNN::Tensor::createHostTensorFromDevice(device, false));
    if (!host || !host->host<float>()) return false;
    std::memcpy(host->host<float>(), values, count * sizeof(float));
    device->copyFromHostTensor(host.get());
    return true;
}

inline bool copyUint8Input(
        MNN::Interpreter* interpreter,
        MNN::Session* session,
        const char* name,
        const uint8_t* values,
        size_t count) {
    if (!interpreter || !session || !values) return false;
    MNN::Tensor* device = interpreter->getSessionInput(session, name);
    if (!device || static_cast<size_t>(device->elementSize()) != count) return false;
    std::unique_ptr<MNN::Tensor> host(
            MNN::Tensor::createHostTensorFromDevice(device, false));
    if (!host || !host->host<uint8_t>()) return false;
    std::memcpy(host->host<uint8_t>(), values, count);
    device->copyFromHostTensor(host.get());
    return true;
}

inline bool copyFloatOutput(
        MNN::Interpreter* interpreter,
        MNN::Session* session,
        const char* name,
        float* values,
        size_t count) {
    if (!interpreter || !session || !values) return false;
    MNN::Tensor* device = interpreter->getSessionOutput(session, name);
    if (!device || static_cast<size_t>(device->elementSize()) != count) return false;
    std::unique_ptr<MNN::Tensor> host(
            MNN::Tensor::createHostTensorFromDevice(device, false));
    if (!host || !host->host<float>()) return false;
    device->copyToHostTensor(host.get());
    std::memcpy(values, host->host<float>(), count * sizeof(float));
    return true;
}

}  // namespace mobilei2v
