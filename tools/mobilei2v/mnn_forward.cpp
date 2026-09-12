// Host CPU diagnostic for qualify_mnn.py, not an Android/Adreno runner.
// Build against the pinned parent's libMNN (no extra libraries):
// g++ -std=c++11 -O2 mnn_forward.cpp -I MNN/include
//     -I MNN/3rd_party -L BUILD -lMNN
//     -Wl,-rpath,BUILD -o mnn_forward
// CLI: mnn_forward --manifest /absolute/path/manifest.json
// Only the original FP32 VAE encoder/decoder contracts are supported.

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <rapidjson/document.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
using Json = rapidjson::Value;
using Shapes = std::map<std::string, std::vector<int>>;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void keys(const Json& value, const std::set<std::string>& expected) {
    require(value.IsObject(), "manifest value must be an object");
    std::set<std::string> actual;
    for (auto it = value.MemberBegin(); it != value.MemberEnd(); ++it) {
        std::string name(it->name.GetString(), it->name.GetStringLength());
        require(actual.insert(name).second, "duplicate manifest key: " + name);
    }
    require(actual == expected, "manifest object has missing or extra keys");
}

std::string string_value(const Json& value) {
    require(value.IsString(), "manifest string required");
    std::string result(value.GetString(), value.GetStringLength());
    require(!result.empty() && result.find('\0') == std::string::npos, "invalid manifest string");
    return result;
}

std::string path_value(const Json& value) {
    auto result = string_value(value);
    require(result[0] == '/', "manifest file paths must be absolute");
    return result;
}

size_t checked_count(const std::vector<int>& shape) {
    require(!shape.empty() && shape.size() <= 5, "invalid VAE tensor rank");
    // MNN's host allocation/elementSize APIs use int byte counts in this pin.
    const size_t limit = static_cast<size_t>(std::numeric_limits<int>::max()) / sizeof(float);
    size_t count = 1;
    for (int dim : shape) {
        require(dim > 0 && count <= limit / static_cast<size_t>(dim), "tensor byte count overflow");
        count *= static_cast<size_t>(dim);
    }
    return count;
}

struct Fixture {
    std::string name;
    std::string path;
    std::vector<int> shape;
    size_t count;
};

std::vector<Fixture> fixtures(const Json& value, const Shapes& expected, std::set<std::string>& paths) {
    std::set<std::string> names;
    for (const auto& pair : expected) {
        names.insert(pair.first);
    }
    keys(value, names);
    std::vector<Fixture> result;
    for (const auto& pair : expected) {
        const auto& entry = value[pair.first.c_str()];
        keys(entry, {"dtype", "path", "shape"});
        require(string_value(entry["dtype"]) == "float32", "only float32 fixture ports are supported");
        require(entry["shape"].IsArray(), "fixture shape must be an array");
        std::vector<int> shape;
        for (auto it = entry["shape"].Begin(); it != entry["shape"].End(); ++it) {
            require(it->IsInt(), "shape dimension must be an integer");
            shape.push_back(it->GetInt());
        }
        require(shape == pair.second, "manifest differs from original VAE shape: " + pair.first);
        auto path = path_value(entry["path"]);
        require(paths.insert(path).second, "model/input/output paths must be distinct");
        result.push_back({pair.first, path, shape, checked_count(shape)});
    }
    return result;
}

void check_tensor(const MNN::Tensor* tensor, const Fixture& fixture, bool host = false) {
    require(tensor != nullptr, "missing MNN tensor: " + fixture.name);
    const auto type = tensor->getType();
    require(type.code == halide_type_float && type.bits == 32 && type.lanes == 1,
            "MNN port is not scalar-lane float32: " + fixture.name);
    require(tensor->shape() == fixture.shape, "MNN port shape mismatch: " + fixture.name);
    const auto layout = tensor->getDimensionType();
    // CPU kernels may pack channel-first tensors as CAFFE_C4 internally.
    // The materialized host fixture must always be plain NCHW/CAFFE.
    require(layout == MNN::Tensor::CAFFE || (!host && layout == MNN::Tensor::CAFFE_C4),
            "MNN port must retain channel-first ONNX layout: " + fixture.name);
    require(checked_count(tensor->shape()) == fixture.count, "MNN tensor count mismatch");
    if (host) {
        require(tensor->host<float>() != nullptr, "host tensor allocation failed");
        require(tensor->usize() == fixture.count * sizeof(float), "host float32 byte count mismatch");
    }
}

void check_ports(const std::map<std::string, MNN::Tensor*>& ports, const std::vector<Fixture>& expected) {
    require(ports.size() == expected.size(), "MNN graph has extra or missing ports");
    for (const auto& fixture : expected) {
        auto it = ports.find(fixture.name);
        require(it != ports.end(), "MNN graph is missing port: " + fixture.name);
        check_tensor(it->second, fixture);
    }
}

void read_input(const Fixture& fixture, MNN::Tensor& host) {
    const auto bytes = fixture.count * sizeof(float);
    std::ifstream stream(fixture.path, std::ios::binary | std::ios::ate);
    require(stream.is_open() && stream.tellg() == static_cast<std::streamoff>(bytes),
            "input binary byte count mismatch: " + fixture.name);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(host.host<float>()), static_cast<std::streamsize>(bytes));
    require(stream.good() && stream.gcount() == static_cast<std::streamsize>(bytes),
            "input binary read failed: " + fixture.name);
    require(stream.peek() == std::char_traits<char>::eof(), "input binary grew during read");
    for (size_t index = 0; index < fixture.count; ++index) {
        require(std::isfinite(host.host<float>()[index]), "input contains nonfinite float32 values");
    }
}

void write_output(const Fixture& fixture, const MNN::Tensor& host) {
    // O_EXCL prevents stale outputs or input/model aliases from being overwritten.
    int descriptor = ::open(fixture.path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    require(descriptor >= 0, "cannot create fresh output: " + fixture.path + ": " + std::strerror(errno));
    FILE* stream = ::fdopen(descriptor, "wb");
    if (!stream) {
        ::close(descriptor);
        throw std::runtime_error("cannot open output stream: " + fixture.path);
    }
    const auto written = std::fwrite(host.host<float>(), sizeof(float), fixture.count, stream);
    const int closed = std::fclose(stream);
    require(written == fixture.count && closed == 0, "output binary write failed: " + fixture.name);
}

void forward(const char* manifest_path) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE float32 host required");
    const uint32_t endian = 1;
    require(*reinterpret_cast<const unsigned char*>(&endian) == 1, "little-endian host required");
    std::ifstream input(manifest_path, std::ios::binary | std::ios::ate);
    require(input.is_open(), "cannot open manifest");
    const auto length = input.tellg();
    require(length > 0 && length <= 65536, "invalid manifest byte count");
    std::string content(static_cast<size_t>(length), '\0');
    input.seekg(0);
    input.read(&content[0], static_cast<std::streamsize>(length));
    require(input.good() && content.find('\0') == std::string::npos, "cannot read manifest JSON");
    rapidjson::Document manifest;
    manifest.Parse<rapidjson::kParseValidateEncodingFlag>(content.c_str());
    require(!manifest.HasParseError(), "invalid manifest JSON");
    keys(manifest, {"format", "stage", "byte_order", "model", "threads", "inputs", "outputs"});
    require(string_value(manifest["format"]) == "mobilei2v-mnn-forward-v1", "unsupported manifest version");
    require(string_value(manifest["byte_order"]) == "little", "binary byte order must be little endian");
    require(manifest["threads"].IsInt() && manifest["threads"].GetInt() >= 1
            && manifest["threads"].GetInt() <= 16, "threads must be 1..16");
    auto stage = string_value(manifest["stage"]);
    require(stage == "encoder" || stage == "decoder", "unsupported VAE stage");
    const Shapes input_shapes = stage == "encoder"
        ? Shapes{{"image", {1, 3, 720, 1280}}, {"posterior_epsilon", {1, 128, 1, 23, 40}}}
        : Shapes{{"latent", {1, 128, 3, 23, 40}}};
    const Shapes output_shapes = stage == "encoder"
        ? Shapes{{"guide", {1, 128, 1, 23, 40}}}
        : Shapes{{"video", {1, 3, 17, 720, 1280}}};
    const auto model_path = path_value(manifest["model"]);
    std::set<std::string> paths{model_path, manifest_path};
    const auto inputs = fixtures(manifest["inputs"], input_shapes, paths);
    const auto outputs = fixtures(manifest["outputs"], output_shapes, paths);
    {
        std::ifstream model_file(model_path, std::ios::binary | std::ios::ate);
        require(model_file.is_open() && model_file.tellg() > 0
                && model_file.tellg() <= std::numeric_limits<int>::max(), "invalid bounded MNN model file");
    }
    // qualify_mnn.py rejects actual ONNX HALF ports before conversion: this
    // pinned Tensor::setType does not implement DT_HALF. Never infer safety
    // from the manifest alone, or reinterpret a half buffer as float here.
    std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(model_path.c_str()));
    require(net != nullptr, "MNN could not load the real converted graph");
    net->setSessionMode(MNN::Interpreter::Session_Release);
    net->setSessionMode(MNN::Interpreter::Session_Backend_Fix);
    // Establish FP32 parity before enabling transformed convolution kernels.
    // In this pinned CPU factory, level 0 selects dense convolution instead
    // of Winograd. The original weights, graph and tolerances stay unchanged.
    net->setSessionHint(MNN::Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
    std::cout << "MNN WINOGRAD_MEMORY_LEVEL: 0 (dense CPU convolution)\n";
    MNN::BackendConfig backend;
    backend.precision = MNN::BackendConfig::Precision_High;
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.backupType = MNN_FORWARD_CPU;
    config.numThread = manifest["threads"].GetInt();
    config.backendConfig = &backend;
    auto* session = net->createSession(config);
    require(session != nullptr, "MNN CPU session creation failed");
    int resize_status = -1;
    require(net->getSessionInfo(session, MNN::Interpreter::RESIZE_STATUS, &resize_status) && resize_status == 0,
            "MNN CPU session is not ready; resizing is forbidden for this exact contract");
    int backends[2] = {-1, -1};
    // The pinned CPURuntime selects AVX2Backend on x86. It reports the
    // internal CPU_EXTENSION type; this is still CPU, not an accelerator.
    const auto is_cpu = [](int type) {
        return type == MNN_FORWARD_CPU || type == MNN_FORWARD_CPU_EXTENSION;
    };
    require(net->getSessionInfo(session, MNN::Interpreter::BACKENDS, backends)
            && is_cpu(backends[0]) && (backends[1] == -1 || is_cpu(backends[1])),
            "MNN session is not CPU-only");
    std::cout << "MNN CPU backend types: " << backends[0] << ", " << backends[1] << '\n';
    check_ports(net->getSessionInputAll(session), inputs);
    check_ports(net->getSessionOutputAll(session), outputs);
    for (const auto& fixture : inputs) {
        auto* device = net->getSessionInputAll(session).at(fixture.name);
        MNN::Tensor host(device, MNN::Tensor::CAFFE);
        check_tensor(&host, fixture, true);
        read_input(fixture, host);
        require(device->copyFromHostTensor(&host), "MNN copyFromHostTensor failed: " + fixture.name);
    }
    const auto error = net->runSession(session);
    require(error == MNN::NO_ERROR, "MNN runSession failed with code " + std::to_string(static_cast<int>(error)));
    check_ports(net->getSessionInputAll(session), inputs);
    check_ports(net->getSessionOutputAll(session), outputs);
    for (const auto& fixture : outputs) {
        const auto* device = net->getSessionOutputAll(session).at(fixture.name);
        MNN::Tensor host(device, MNN::Tensor::CAFFE);
        check_tensor(&host, fixture, true);
        // A no-op/partial copy cannot inherit zero-filled memory and pass.
        std::fill_n(host.host<float>(), fixture.count, std::numeric_limits<float>::quiet_NaN());
        require(device->copyToHostTensor(&host), "MNN copyToHostTensor failed: " + fixture.name);
        write_output(fixture, host);  // Python records nonfinite/numerical failures.
    }
    require(net->releaseSession(session), "MNN releaseSession failed");
    std::cout << "MNN CPU forward completed; Python parity comparison is still required.\n";
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[1]) != "--manifest") {
        std::cerr << "usage: mnn_forward --manifest /absolute/path/manifest.json\n";
        return 2;
    }
    try {
        forward(argv[2]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MNN CPU diagnostic failed: " << error.what() << '\n';
        return 1;
    }
}
