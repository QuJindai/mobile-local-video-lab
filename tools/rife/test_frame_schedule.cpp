#include "face_re_frame_schedule.h"

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static int checks = 0;

// Keep checks active in Release builds with -DNDEBUG.
static void check(bool ok, const char* message)
{
    ++checks;
    if (!ok)
    {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static void expect_sample(int inputs, int outputs, int index,
                          int first, int second, float timestep)
{
    face_re::RifeFrameSample sample = {-1, -1, -1.f};
    check(face_re::sample_rife_frame(inputs, outputs, index, sample),
          "valid sample accepted");
    if (sample.first != first || sample.second != second || sample.timestep != timestep)
    {
        std::fprintf(stderr, "%d -> %d at %d: expected (%d, %d, %.9g), got (%d, %d, %.9g)\n",
                     inputs, outputs, index, first, second, timestep,
                     sample.first, sample.second, sample.timestep);
        check(false, "sample matches the hand-derived schedule");
    }
    check(sample.first >= 0 && sample.second < inputs && sample.second == sample.first + 1,
          "both frame indices are in bounds and adjacent");
    check(std::isfinite(sample.timestep) && sample.timestep >= 0.f && sample.timestep <= 1.f,
          "timestep is finite and within [0, 1]");
}

static void two_inputs_seventeen_outputs()
{
    // The old count/numframe schedule produced 16/17 at the midpoint and
    // clamped outputs 9..16 to the endpoint. All 17 samples must advance.
    expect_sample(2, 17, 0, 0, 1, 0.f);
    expect_sample(2, 17, 8, 0, 1, 0.5f);
    expect_sample(2, 17, 16, 0, 1, 1.f);
    const float expected[] = {
        0.f, 0.0625f, 0.125f, 0.1875f, 0.25f, 0.3125f, 0.375f, 0.4375f,
        0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f, 1.f
    };
    for (int i = 0; i < 17; ++i)
        expect_sample(2, 17, i, 0, 1, expected[i]);
}

static void five_inputs_thirty_three_outputs()
{
    // Timeline quarter anchors hit each input exactly, including the last.
    expect_sample(5, 33, 0, 0, 1, 0.f);
    expect_sample(5, 33, 8, 1, 2, 0.f);
    expect_sample(5, 33, 16, 2, 3, 0.f);
    expect_sample(5, 33, 24, 3, 4, 0.f);
    expect_sample(5, 33, 32, 3, 4, 1.f);

    const float within_pair[] = {0.f, 0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f};
    for (int pair = 0; pair < 4; ++pair)
        for (int offset = 0; offset < 8; ++offset)
            expect_sample(5, 33, pair * 8 + offset, pair, pair + 1, within_pair[offset]);
}

static void downsampling_and_nonintegral_spacing()
{
    expect_sample(2, 2, 0, 0, 1, 0.f);
    expect_sample(2, 2, 1, 0, 1, 1.f);
    expect_sample(5, 2, 0, 0, 1, 0.f);
    expect_sample(5, 2, 1, 3, 4, 1.f);
    expect_sample(5, 3, 1, 2, 3, 0.f);

    const int first[] = {0, 0, 0, 0, 1, 1, 1, 1};
    const float timestep[] = {0.f, 2.f / 7.f, 4.f / 7.f, 6.f / 7.f,
                             1.f / 7.f, 3.f / 7.f, 5.f / 7.f, 1.f};
    for (int i = 0; i < 8; ++i)
        expect_sample(3, 8, i, first[i], first[i] + 1, timestep[i]);
}

static void extreme_counts_do_not_overflow_or_round_the_pair()
{
    expect_sample(INT_MAX, INT_MAX, 1, 1, 2, 0.f);
    expect_sample(INT_MAX, INT_MAX, INT_MAX - 2, INT_MAX - 2, INT_MAX - 1, 0.f);
    expect_sample(INT_MAX, INT_MAX, INT_MAX - 1, INT_MAX - 2, INT_MAX - 1, 1.f);
    expect_sample(INT_MAX, 3, 1, (INT_MAX - 1) / 2, (INT_MAX - 1) / 2 + 1, 0.f);
    expect_sample(2, INT_MAX, (INT_MAX - 1) / 2, 0, 1, 0.5f);
    expect_sample(2, INT_MAX, INT_MAX - 1, 0, 1, 1.f);
}

static void invalid_bounds_are_rejected_without_changing_the_sample()
{
    const int invalid[][3] = {
        {0, 17, 0}, {1, 17, 0}, {-1, 17, 0}, {INT_MIN, 17, 0},
        {2, 0, 0}, {2, 1, 0}, {2, -1, 0}, {2, INT_MIN, 0},
        {2, 17, -1}, {2, 17, INT_MIN}, {2, 17, 17}, {2, 17, INT_MAX},
        {INT_MAX, INT_MAX, INT_MAX}
    };
    for (const auto& bounds : invalid)
    {
        face_re::RifeFrameSample sample = {7, 8, 0.25f};
        check(!face_re::sample_rife_frame(bounds[0], bounds[1], bounds[2], sample),
              "invalid count or output index rejected");
        check(sample.first == 7 && sample.second == 8 && sample.timestep == 0.25f,
              "invalid sample leaves the result untouched");
    }
}

int main()
{
    two_inputs_seventeen_outputs();
    five_inputs_thirty_three_outputs();
    downsampling_and_nonintegral_spacing();
    extreme_counts_do_not_overflow_or_round_the_pair();
    invalid_bounds_are_rejected_without_changing_the_sample();
    std::printf("PASS: RIFE endpoint-inclusive schedule (%d checks, including Release bounds checks)\n", checks);
    return 0;
}
