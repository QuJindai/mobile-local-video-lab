#pragma once

#include "MNN_generated.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace mobilei2v {

// Inspect the serialized declaration before InitNet can call setType(). The
// pinned Release Tensor::setType(DT_HALF) leaves its default Float32 unchanged.
inline bool validateSerializedFloatInputs(
        const void* data, size_t size, std::string& error) {
    if (!data || size < sizeof(flatbuffers::uoffset_t) ||
            size >= static_cast<size_t>(std::numeric_limits<flatbuffers::soffset_t>::max())) {
        error = "invalid serialized MNN buffer size";
        return false;
    }
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(data), size);
    if (!MNN::VerifyNetBuffer(verifier)) {
        error = "invalid serialized MNN graph";
        return false;
    }
    const auto* net = MNN::GetNet(data);
    if (!net->oplists() || (net->subgraphs() && net->subgraphs()->size() != 0)) {
        error = "MobileI2V requires one explicit-input graph";
        return false;
    }
    size_t inputs = 0;
    for (const auto* op : *net->oplists()) {
        if (!op) { error = "serialized MNN graph contains a null operation"; return false; }
        if (op->type() != MNN::OpType_Input) continue;
        const auto* input = op->main_as_Input();
        if (!input || input->dtype() != MNN::DataType_DT_FLOAT) {
            error = "serialized input " + (op->name() ? op->name()->str() : "unnamed")
                    + ": declared dtype=" + std::to_string(input ? int(input->dtype()) : -1)
                    + " rejected; FP32 required before MNN initialization";
            return false;
        }
        ++inputs;
    }
    if (inputs == 0) {
        error = "serialized MNN graph has no explicit inputs";
        return false;
    }
    return true;
}

}  // namespace mobilei2v
