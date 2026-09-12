#!/usr/bin/env python3
"""Patch only the expected scheduling anchors in pinned RIFE src/main.cpp.

Run after checking out a7532fc3f9f8f008cd6eecd6f2ffe2a9698e0cf7.
Every preimage must occur exactly once. Validate all anchors before writing;
source drift and repeated application are errors, never best-effort patches.
"""

import argparse
from pathlib import Path
import sys


PINNED_COMMIT = "a7532fc3f9f8f008cd6eecd6f2ffe2a9698e0cf7"

INCLUDE_BEFORE = b'#include "rife.h"\n'
INCLUDE_AFTER = INCLUDE_BEFORE + b'\n#include <limits.h>\n#include "face_re_frame_schedule.h"\n'

COUNTS_BEFORE = b'''            const int count = filenames.size();
            if (numframe == 0)
                numframe = count * 2;

'''
COUNTS_AFTER = br'''            if (filenames.size() < 2 || filenames.size() > static_cast<size_t>(INT_MAX))
            {
                fprintf(stderr, "RIFE schedule requires 2..INT_MAX input frames\n");
                return -1;
            }
            const int count = static_cast<int>(filenames.size());
            if (numframe == 0)
            {
                if (count > INT_MAX / 2)
                {
                    fprintf(stderr, "RIFE default output frame count would overflow\n");
                    return -1;
                }
                numframe = count * 2;
            }
            if (numframe < 2)
            {
                fprintf(stderr, "RIFE schedule requires at least 2 output frames\n");
                return -1;
            }

'''

SCHEDULE_BEFORE = br'''            double scale = (double)count / numframe;
            for (int i=0; i<numframe; i++)
            {
                // TODO provide option to control timestep interpolate method
//                 float fx = (float)((i + 0.5) * scale - 0.5);
                float fx = i * scale;
                int sx = static_cast<int>(floor(fx));
                fx -= sx;

                if (sx < 0)
                {
                    sx = 0;
                    fx = 0.f;
                }
                if (sx >= count - 1)
                {
                    sx = count - 2;
                    fx = 1.f;
                }

//                 fprintf(stderr, "%d %f %d\n", i, fx, sx);

                path_t filename0 = filenames[sx];
                path_t filename1 = filenames[sx + 1];

'''
SCHEDULE_AFTER = br'''            for (int i=0; i<numframe; i++)
            {
                face_re::RifeFrameSample sample;
                if (!face_re::sample_rife_frame(count, numframe, i, sample))
                {
                    fprintf(stderr, "invalid RIFE frame schedule bounds\n");
                    return -1;
                }
                const float fx = sample.timestep;

                path_t filename0 = filenames[sample.first];
                path_t filename1 = filenames[sample.second];

'''

REPLACEMENTS = (
    ("rife.h inclusion", INCLUDE_BEFORE, INCLUDE_AFTER),
    ("input/output counts", COUNTS_BEFORE, COUNTS_AFTER),
    ("directory frame schedule", SCHEDULE_BEFORE, SCHEDULE_AFTER),
)


def patched_source(source: bytes) -> bytes:
    if b"face_re_frame_schedule.h" in source or b"face_re::sample_rife_frame" in source:
        raise ValueError("source already contains the FACE-RE schedule patch")
    for name, before, _after in REPLACEMENTS:
        matches = source.count(before)
        if matches != 1:
            raise ValueError(f"expected exactly one {name} anchor for {PINNED_COMMIT}, found {matches}")
    for _name, before, after in REPLACEMENTS:
        source = source.replace(before, after, 1)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_cpp", type=Path, help="pristine pinned src/main.cpp")
    args = parser.parse_args()
    try:
        patched = patched_source(args.main_cpp.read_bytes())
        args.main_cpp.write_bytes(patched)
    except (OSError, ValueError) as error:
        print(f"RIFE schedule patch failed: {error}", file=sys.stderr)
        return 1
    print(f"Patched {args.main_cpp}: endpoint-inclusive-v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
