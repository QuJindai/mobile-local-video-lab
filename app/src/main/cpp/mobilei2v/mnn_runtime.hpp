#pragma once

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "mnn_port_admission.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mobilei2v {

struct FloatPort {
    const char* name;
    std::array<int, 5> shape;
    size_t rank;
    size_t count;
};

inline bool fail(std::string& error, const std::string& message) {
    error = message;
    return false;
}

inline bool finiteFloats(const float* values, size_t count) {
    if (!values) return false;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) return false;
    }
    return true;
}

inline bool validateFloatTensor(
        const MNN::Tensor* tensor, const FloatPort& port, std::string& error, bool host = false) {
    const std::string label(port.name);
    if (!tensor) return fail(error, label + ": missing tensor");
    const auto type = tensor->getType();
    if (type.code == halide_type_float && type.bits == 16) {
        return fail(error, label + ": DT_HALF/FP16 runtime port rejected; FP32 required");
    }
    if (type.code != halide_type_float || type.bits != 32 || type.lanes != 1) {
        return fail(error, label + ": unsupported runtime type (code=" +
                std::to_string(type.code) + ", bits=" + std::to_string(type.bits) +
                ", lanes=" + std::to_string(type.lanes) + "); FP32 required");
    }
    if (port.rank == 0 || port.rank > port.shape.size() ||
            tensor->dimensions() != static_cast<int>(port.rank)) {
        return fail(error, label + ": rank mismatch");
    }
    size_t count = 1;
    for (size_t i = 0; i < port.rank; ++i) {
        const int extent = port.shape[i];
        if (extent <= 0 || tensor->length(static_cast<int>(i)) != extent) {
            return fail(error, label + ": shape mismatch at dimension " + std::to_string(i));
        }
        if (count > static_cast<size_t>(std::numeric_limits<int>::max()) /
                sizeof(float) / static_cast<size_t>(extent)) {
            return fail(error, label + ": tensor byte count overflows MNN int size");
        }
        count *= static_cast<size_t>(extent);
    }
    const auto layout = tensor->getDimensionType();
    if (count != port.count || (layout != MNN::Tensor::CAFFE &&
            (host || layout != MNN::Tensor::CAFFE_C4))) {
        return fail(error, label + ": exact logical shape/count and channel-first layout required");
    }
    const size_t logicalBytes = count * sizeof(float);
    size_t paddedCount = count;
    if (!host && port.rank > 1) {
        const size_t channels = static_cast<size_t>(port.shape[1]);
        const size_t paddedChannels = (channels + 3) / 4 * 4;
        const size_t planes = count / channels;
        if (planes > static_cast<size_t>(std::numeric_limits<int>::max()) /
                sizeof(float) / paddedChannels) {
            return fail(error, label + ": packed FP32 byte count overflow");
        }
        paddedCount = planes * paddedChannels;
    }
    // Pinned getDimensionType() canonicalizes CAFFE_C4 to CAFFE. usize() still
    // includes its channel padding. Accept only plain or exactly C4-padded
    // device storage; host wrappers must remain exact, unpadded FP32 buffers.
    const size_t physicalBytes = tensor->usize();
    if ((physicalBytes != logicalBytes && (host || physicalBytes != paddedCount * sizeof(float))) ||
            tensor->size() != static_cast<int>(physicalBytes) ||
            tensor->elementSize() != static_cast<int>(physicalBytes / sizeof(float))) {
        return fail(error, label + ": invalid plain/CAFFE_C4 FP32 storage byte count");
    }
    return true;
}

// mmap avoids a separate application-owned file buffer. The pinned MNN
// createFromBuffer still copies the model; the mapping need not outlive it.
inline std::unique_ptr<MNN::Interpreter> createInterpreterMmap(
        const std::string& path, std::string& error) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fail(error, path + ": cannot open model");
        return nullptr;
    }
    struct stat st{};
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0 ||
            static_cast<uintmax_t>(st.st_size) > std::numeric_limits<size_t>::max()) {
        close(fd);
        fail(error, path + ": invalid model file size/type");
        return nullptr;
    }
    const size_t size = static_cast<size_t>(st.st_size);
    void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fail(error, path + ": model mmap failed");
        return nullptr;
    }
    struct Mapping {
        void* data;
        size_t size;
        ~Mapping() { munmap(data, size); }
    } mapping{mapped, size};
    madvise(mapped, size, MADV_SEQUENTIAL);
    if (!validateSerializedFloatInputs(mapped, size, error)) return nullptr;
    std::unique_ptr<MNN::Interpreter> interpreter(
            MNN::Interpreter::createFromBuffer(mapped, size));
    if (!interpreter) {
        fail(error, path + ": MNN rejected model buffer");
        return nullptr;
    }
    interpreter->setExternalFile((path + ".weight").c_str());
    return interpreter;
}

inline std::string ensureCacheDir(const std::string& modelDir) {
    if (modelDir.empty()) return {};
    std::error_code ec;
    const auto cache = std::filesystem::path(modelDir) / "cache";
    std::filesystem::create_directories(cache, ec);
    return ec ? std::string{} : cache.string();
}

struct OpenClSession {
    std::unique_ptr<MNN::Interpreter> interpreter;
    MNN::Session* session = nullptr;
    std::string cacheFile;
    const MNN::Backend* deviceBackend = nullptr;
    std::map<const MNN::OperatorInfo*, std::vector<MNN::Tensor*>> executionOutputs;

    explicit operator bool() const { return interpreter && session; }
};

inline bool validateOpenClSession(const OpenClSession& graph, std::string& error) {
    if (!graph) return fail(error, "missing OpenCL session");
    // One ScheduleConfig. Pinned BACKENDS writes backend IDs, not a count prefix.
    int backends[2] = {-1, -1};
    if (!graph.interpreter->getSessionInfo(
            graph.session, MNN::Interpreter::BACKENDS, backends) ||
            backends[0] != MNN_FORWARD_OPENCL || backends[1] != -1) {
        return fail(error, "production CPU/unknown backend rejected (main=" +
                std::to_string(backends[0]) + ")");
    }
    int resizeStatus = -1;
    if (!graph.interpreter->getSessionInfo(
            graph.session, MNN::Interpreter::RESIZE_STATUS, &resizeStatus) || resizeStatus != 0) {
        return fail(error, "MNN session resize/allocation incomplete (status=" +
                std::to_string(resizeStatus) + ")");
    }
    return true;
}

inline OpenClSession createOpenClSession(
        const std::string& modelPath, const std::string& modelDir,
        const std::string& cacheName, std::string& error) {
    OpenClSession result;
    result.interpreter = createInterpreterMmap(modelPath, error);
    if (!result.interpreter) return result;
    result.interpreter->setSessionMode(MNN::Interpreter::Session_Debug);
    result.interpreter->setSessionMode(MNN::Interpreter::Session_Backend_Fix);
    result.interpreter->setSessionMode(MNN::Interpreter::Session_Codegen_Disable);
    const std::string cacheDir = ensureCacheDir(modelDir);
    if (!cacheDir.empty()) {
        result.cacheFile = (std::filesystem::path(cacheDir) / cacheName).string();
        result.interpreter->setCacheFile(result.cacheFile.c_str());
    }
    MNN::ScheduleConfig config;
    MNN::BackendConfig backend;
    config.type = MNN_FORWARD_OPENCL;
    config.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST;
    backend.precision = MNN::BackendConfig::Precision_High;
    backend.power = MNN::BackendConfig::Power_High;
    backend.memory = MNN::BackendConfig::Memory_Low;
    config.backendConfig = &backend;
    auto runtime = MNN::Interpreter::createRuntime({config});
    if (runtime.first.size() != 1 || runtime.first.begin()->first != MNN_FORWARD_OPENCL ||
            !runtime.first.begin()->second) {
        fail(error, modelPath + ": OpenCL runtime unavailable; CPU fallback rejected");
        return {};
    }
    result.session = result.interpreter->createSession(config, runtime);
    if (!validateOpenClSession(result, error)) return {};
    return result;
}

// With exactly one verified OpenCL runtime, the only other execution backend
// in pinned MNN is its CPU backup. CPU tensors have host storage, no deviceId.
// Compare opaque getBackend() identities; do not depend on private Backend ABI.
inline bool checkDeviceTensor(
        OpenClSession& graph, MNN::Tensor* tensor, const std::string& label,
        std::string& error, bool establishBackend = false) {
    const auto* backend = tensor ? graph.interpreter->getBackend(graph.session, tensor) : nullptr;
    if (!tensor || !backend || tensor->deviceId() == 0 || tensor->host<void>() != nullptr) {
        return fail(error, label + ": production CPU/unknown tensor backend rejected");
    }
    if (establishBackend && !graph.deviceBackend) graph.deviceBackend = backend;
    if (!graph.deviceBackend || backend != graph.deviceBackend) {
        return fail(error, label + ": tensor backend differs from verified OpenCL ports");
    }
    return true;
}

inline bool checkExecutionOutputs(
        OpenClSession& graph, const std::vector<MNN::Tensor*>& outputs,
        const MNN::OperatorInfo* info, std::string& error) {
    if (!info || outputs.empty()) return fail(error, "execution backend cannot be verified");
    for (auto* tensor : outputs) {
        if (!checkDeviceTensor(graph, tensor, "op " + info->name(), error)) return false;
    }
    return true;
}

inline bool auditOpenClExecutions(OpenClSession& graph, std::string& error) {
    if (!validateOpenClSession(graph, error) || !graph.deviceBackend) return false;
    graph.executionOutputs.clear();
    bool ok = true;
    // Public debug callbacks expose the actual workOutputs after CPU/GPU wraps.
    // Skip neural executions during this metadata audit. MNN may still perform
    // internal transfer/housekeeping commands that have no OperatorInfo.
    const auto skip = [](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo*) {
        return false;
    };
    const auto inspect = [&](const std::vector<MNN::Tensor*>& outputs, const MNN::OperatorInfo* info) {
        ok = checkExecutionOutputs(graph, outputs, info, error);
        if (ok) graph.executionOutputs.emplace(info, outputs);
        return ok;
    };
    const auto code = graph.interpreter->runSessionWithCallBackInfo(graph.session, skip, inspect, true);
    if (!ok) return false;
    if (code != MNN::NO_ERROR || graph.executionOutputs.empty()) {
        return fail(error, "OpenCL execution audit failed or callbacks unavailable (code=" +
                std::to_string(code) + ")");
    }
    // No resize or graph rebuilding is allowed after this audit.
    graph.interpreter->releaseModel();
    return true;
}

inline bool runOpenClSession(OpenClSession& graph, std::string& error) {
    if (!validateOpenClSession(graph, error)) return false;
    bool ok = true;
    size_t visited = 0;
    const auto before = [&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) {
        const auto found = graph.executionOutputs.find(info);
        if (found == graph.executionOutputs.end()) {
            ok = fail(error, "unaudited execution rejected");
        } else {
            ok = checkExecutionOutputs(graph, found->second, info, error);
        }
        return ok;  // A rejected operation is skipped, then after stops the run.
    };
    const auto after = [&](const std::vector<MNN::Tensor*>& outputs, const MNN::OperatorInfo* info) {
        if (!ok) return false;
        ++visited;
        ok = checkExecutionOutputs(graph, outputs, info, error);
        return ok;
    };
    const auto code = graph.interpreter->runSessionWithCallBackInfo(graph.session, before, after, true);
    if (!ok) return false;
    if (code != MNN::NO_ERROR || visited != graph.executionOutputs.size()) {
        return fail(error, "OpenCL execution failed/incomplete (code=" + std::to_string(code) + ")");
    }
    return true;
}

inline std::unique_ptr<MNN::Tensor> plainFloatHost(
        const FloatPort& port, float* values, std::string& error) {
    std::vector<int> shape(port.shape.begin(), port.shape.begin() + port.rank);
    std::unique_ptr<MNN::Tensor> host(MNN::Tensor::create<float>(shape, values, MNN::Tensor::CAFFE));
    if (!validateFloatTensor(host.get(), port, error, true) || host->host<float>() != values) return nullptr;
    size_t stride = 1;
    for (size_t i = port.rank; i > 0; --i) {
        if (host->stride(static_cast<int>(i - 1)) != static_cast<int>(stride)) {
            fail(error, std::string(port.name) + ": host tensor is not plain contiguous CAFFE");
            return nullptr;
        }
        stride *= static_cast<size_t>(port.shape[i - 1]);
    }
    return host;
}

inline bool copyFloatInput(
        OpenClSession& graph, const FloatPort& port, const float* values,
        size_t count, std::string& error) {
    if (!graph || !values || count != port.count) return fail(error, "invalid FP32 input/count");
    auto* device = graph.interpreter->getSessionInput(graph.session, port.name);
    if (!validateFloatTensor(device, port, error) ||
            !checkDeviceTensor(graph, device, port.name, error)) return false;
    if (!finiteFloats(values, count)) return fail(error, std::string(port.name) + ": non-finite input");
    // The host wrapper borrows a typed, length-checked buffer. MNN reads it;
    // wait before returning so JNI can safely release its array elements.
    auto host = plainFloatHost(port, const_cast<float*>(values), error);
    if (!host) return false;
    if (!device->copyFromHostTensor(host.get()) ||
            device->wait(MNN::Tensor::MAP_TENSOR_WRITE, true) != MNN::NO_ERROR) {
        return fail(error, std::string(port.name) + ": host-to-OpenCL copy failed");
    }
    return true;
}

inline bool copyFloatOutput(
        OpenClSession& graph, const FloatPort& port, float* values,
        size_t count, std::string& error) {
    if (!graph || !values || count != port.count) return fail(error, "invalid FP32 output/count");
    auto* device = graph.interpreter->getSessionOutput(graph.session, port.name);
    if (!validateFloatTensor(device, port, error) ||
            !checkDeviceTensor(graph, device, port.name, error)) return false;
    auto host = plainFloatHost(port, values, error);
    if (!host) return false;
    // A successful-but-incomplete backend copy cannot masquerade as finite data.
    std::fill_n(values, count, std::numeric_limits<float>::quiet_NaN());
    if (!device->copyToHostTensor(host.get()) ||
            device->wait(MNN::Tensor::MAP_TENSOR_READ, true) != MNN::NO_ERROR) {
        return fail(error, std::string(port.name) + ": OpenCL-to-host copy failed");
    }
    if (!finiteFloats(values, count)) return fail(error, std::string(port.name) + ": non-finite output");
    return true;
}

}  // namespace mobilei2v
