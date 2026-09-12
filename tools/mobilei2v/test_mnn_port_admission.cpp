// Real serialized MNN Input declarations, not synthetic runtime tensor types.
#include "mnn_port_admission.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

std::vector<uint8_t> model(MNN::DataType dtype) {
    MNN::NetT net;
    net.tensorName = {"posterior_epsilon"};
    net.outputName = {"posterior_epsilon"};
    net.tensorNumber = 1;
    auto op = std::make_unique<MNN::OpT>();
    op->name = "posterior_epsilon";
    op->type = MNN::OpType_Input;
    op->outputIndexes = {0};
    op->main.type = MNN::OpParameter_Input;
    auto* input = new MNN::InputT;
    input->dims = {1, 128, 1, 23, 40};
    input->dtype = dtype;
    input->dformat = MNN::MNN_DATA_FORMAT_NCHW;
    op->main.value = input;
    net.oplists.emplace_back(std::move(op));
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(MNN::Net::Pack(builder, &net));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

int main() {
    std::string error;
    for (const auto dtype : {MNN::DataType_DT_FLOAT, MNN::DataType_DT_HALF,
            MNN::DataType_DT_BFLOAT16, MNN::DataType_DT_DOUBLE,
            MNN::DataType_DT_INT32, MNN::DataType_DT_UINT8}) {
        auto bytes = model(dtype);
        const bool accepted = mobilei2v::validateSerializedFloatInputs(bytes.data(), bytes.size(), error);
        if (accepted != (dtype == MNN::DataType_DT_FLOAT)) {
            throw std::runtime_error("serialized dtype admission mismatch: " + std::to_string(dtype));
        }
        if (!accepted && error.find("declared dtype=") == std::string::npos) {
            throw std::runtime_error("missing declared-type diagnostic");
        }
    }
    auto bytes = model(MNN::DataType_DT_FLOAT);
    if (mobilei2v::validateSerializedFloatInputs(bytes.data(), 3, error)) {
        throw std::runtime_error("truncated buffer accepted");
    }
    bytes[0] = 0xff; bytes[1] = 0xff; bytes[2] = 0xff; bytes[3] = 0x7f;
    if (mobilei2v::validateSerializedFloatInputs(bytes.data(), bytes.size(), error)) {
        throw std::runtime_error("invalid root offset accepted");
    }
    std::cout << "PASS serialized MNN FP32 admission; HALF/BFLOAT16/DOUBLE/integer and malformed inputs rejected\n";
}
