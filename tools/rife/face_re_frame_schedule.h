#ifndef FACE_RE_FRAME_SCHEDULE_H
#define FACE_RE_FRAME_SCHEDULE_H

#include <stdint.h>

namespace face_re
{

struct RifeFrameSample
{
    int first;
    int second;
    float timestep;
};

// Endpoint-inclusive position: i * (input_count - 1) / (output_count - 1).
// Invalid bounds leave sample untouched. Both indices of every valid sample
// address an adjacent input pair; the last output uses the last pair at t=1.
inline bool sample_rife_frame(int input_count, int output_count, int output_index,
                              RifeFrameSample& sample)
{
    if (input_count < 2 || output_count < 2 || output_index < 0 || output_index >= output_count)
        return false;

    // A 64-bit product holds any two nonnegative 32-bit int frame indices.
    // Integer division preserves exact input anchors and never rounds the pair
    // index up as a floating-point position near an endpoint could do.
    const int64_t numerator = static_cast<int64_t>(output_index) * (input_count - 1);
    const int denominator = output_count - 1;
    int first = static_cast<int>(numerator / denominator);
    float timestep = static_cast<float>(static_cast<double>(numerator % denominator) / denominator);

    if (first == input_count - 1)
    {
        first = input_count - 2;
        timestep = 1.f;
    }

    sample.first = first;
    sample.second = first + 1;
    sample.timestep = timestep;
    return true;
}

} // namespace face_re

#endif // FACE_RE_FRAME_SCHEDULE_H
