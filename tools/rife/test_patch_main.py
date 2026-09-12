#!/usr/bin/env python3
"""Test against a pristine pinned src/main.cpp supplied as the first argument.

The successful-patch test compiles and runs the actual patched directory
schedule with synthetic filenames, without ncnn, a GPU, or network access.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


RIFE_TOOLS = Path(__file__).resolve().parent


class PinnedMainPatchTests(unittest.TestCase):
    upstream_source = b""

    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="face-re-rife-patch-test-", dir="/tmp")
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)
        self.main = self.scratch / "main.cpp"
        self.main.write_bytes(self.upstream_source)

    def patch(self):
        return subprocess.run(
            [sys.executable, str(RIFE_TOOLS / "patch_main.py"), str(self.main)],
            capture_output=True, text=True, check=False,
        )

    def test_patched_directory_schedule_runs_the_sampler(self):
        result = self.patch()
        self.assertEqual(result.returncode, 0, result.stderr)
        source = self.main.read_text(encoding="utf-8")
        includes = source.split('#include "rife.h"\n', 1)[1].split('#include "filesystem_utils.h"', 1)[0]
        collection = source.split('    // collect input and output filepath\n', 1)[1].split(
            '    path_t modeldir = sanitize_dirpath(model);', 1
        )[0]
        harness = r'''
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
''' + includes + r'''
typedef std::string path_t;
#define PATHSTR(s) s
static int input_count;
static bool path_is_directory(const path_t&) { return true; }
static int list_directory(const path_t&, std::vector<path_t>& filenames)
{
    for (int i = 0; i < input_count; ++i)
        filenames.push_back(std::to_string(i));
    return 0;
}
int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    input_count = std::atoi(argv[1]);
    int numframe = std::atoi(argv[2]);
    float timestep = 0.5f;
    path_t inputpath = "frames", outputpath = "output";
    path_t input0path, input1path, pattern = "%08d", format = "png";
''' + collection + r'''
    for (size_t i = 0; i < output_files.size(); ++i)
        std::printf("%s %s %.9g\n", input0_files[i].c_str(), input1_files[i].c_str(), timesteps[i]);
    return 0;
}
'''
        harness_path = self.scratch / "schedule.cpp"
        harness_path.write_text(harness, encoding="utf-8")
        shutil.copyfile(RIFE_TOOLS / "face_re_frame_schedule.h", self.scratch / "face_re_frame_schedule.h")
        executable = self.scratch / "schedule"
        build = subprocess.run(
            [os.environ.get("HOST_CXX", "c++"), "-std=c++11", "-O2", "-DNDEBUG",
             "-Wall", "-Wextra", "-Werror", str(harness_path), "-o", str(executable)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(build.returncode, 0, build.stderr)

        def run(inputs, outputs):
            return subprocess.run([str(executable), str(inputs), str(outputs)],
                                  capture_output=True, text=True, check=False)

        def samples(inputs, outputs):
            result = run(inputs, outputs)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [line.split() for line in result.stdout.splitlines()]
            return [(int(a.rsplit("/", 1)[1]), int(b.rsplit("/", 1)[1]), float(t)) for a, b, t in rows]

        two = samples(2, 17)
        self.assertEqual(len(two), 17)
        self.assertEqual(two[0], (0, 1, 0.0))
        self.assertEqual(two[8], (0, 1, 0.5))
        self.assertEqual(two[-1], (0, 1, 1.0))
        self.assertLess(two[-2][2], 1.0)

        five = samples(5, 33)
        self.assertEqual(len(five), 33)
        for index, expected in [(0, (0, 1, 0.0)), (8, (1, 2, 0.0)), (16, (2, 3, 0.0)),
                                (24, (3, 4, 0.0)), (32, (3, 4, 1.0))]:
            self.assertEqual(five[index], expected)
        default = samples(2, 0)
        self.assertEqual(len(default), 4)
        self.assertEqual(default[0], (0, 1, 0.0))
        self.assertEqual(default[-1], (0, 1, 1.0))
        for inputs, outputs in [(0, 17), (1, 17), (0, 0), (1, 0), (2, 1), (2, -1)]:
            with self.subTest(inputs=inputs, outputs=outputs):
                invalid = run(inputs, outputs)
                self.assertEqual(invalid.returncode, 255, invalid.stderr)  # main returns -1
                self.assertEqual(invalid.stdout, "")

    def test_mismatched_or_duplicated_anchors_leave_source_untouched(self):
        source = self.upstream_source
        include = b'#include "rife.h"\n'
        count_start = source.index(b'            const int count = filenames.size();\n')
        count_end = source.index(b'            input0_files.resize(numframe);', count_start)
        count_block = source[count_start:count_end]
        schedule_start = source.index(b'            double scale = (double)count / numframe;\n')
        schedule_end = source.index(b'#if _WIN32\n                wchar_t tmp[256];', schedule_start)
        schedule_block = source[schedule_start:schedule_end]
        for name, anchor in [("include", include), ("counts", count_block), ("schedule", schedule_block)]:
            for mutation, replacement in [("missing", b""), ("duplicated", anchor + anchor)]:
                with self.subTest(anchor=name, mutation=mutation):
                    changed = source.replace(anchor, replacement, 1)
                    self.main.write_bytes(changed)
                    result = self.patch()
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(self.main.read_bytes(), changed)
        changed = source.replace(b'double scale = (double)count / numframe;',
                                 b'double scale = (double)(count - 1) / numframe;', 1)
        self.main.write_bytes(changed)
        result = self.patch()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.main.read_bytes(), changed)

    def test_repeated_patch_is_rejected_without_editing(self):
        first = self.patch()
        self.assertEqual(first.returncode, 0, first.stderr)
        patched = self.main.read_bytes()
        second = self.patch()
        self.assertNotEqual(second.returncode, 0, second.stdout)
        self.assertEqual(self.main.read_bytes(), patched)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_main", type=Path)
    args, unittest_args = parser.parse_known_args()
    PinnedMainPatchTests.upstream_source = args.upstream_main.read_bytes()
    unittest.main(argv=[sys.argv[0], *unittest_args])
