cmake_minimum_required(VERSION 3.18)
project(mobilei2v_gpu LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT ANDROID)
    message(FATAL_ERROR "mobilei2v_gpu is Android-arm64 only")
endif()
if(NOT CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    message(FATAL_ERROR "mobilei2v_gpu only supports arm64-v8a")
endif()
if(NOT DEFINED MNN_SOURCE_DIR OR NOT EXISTS "${MNN_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "MNN_SOURCE_DIR must point at pinned MNN source")
endif()

# Match the proven Local-Dream Android GPU configuration. MNN is linked
# statically so the APK has one self-contained accelerator JNI library.
set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_FOR_ANDROID ON CACHE BOOL "" FORCE)
set(MNN_BUILD_FOR_ANDROID_COMMAND ON CACHE BOOL "" FORCE)
set(MNN_USE_LOGCAT OFF CACHE BOOL "" FORCE)
set(MNN_LOW_MEMORY ON CACHE BOOL "" FORCE)
set(MNN_CPU_WEIGHT_DEQUANT_GEMM ON CACHE BOOL "" FORCE)
set(MNN_OPENCL ON CACHE BOOL "" FORCE)
set(MNN_SUPPORT_TRANSFORMER_FUSE ON CACHE BOOL "" FORCE)

add_subdirectory("${MNN_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/mnn" EXCLUDE_FROM_ALL)

add_library(mobilei2v_gpu SHARED
    mobilei2v_jni.cpp
)

target_include_directories(mobilei2v_gpu PRIVATE
    "${MNN_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_options(mobilei2v_gpu PRIVATE
    -O3
    -fvisibility=hidden
    -fno-exceptions
)

target_link_options(mobilei2v_gpu PRIVATE
    "-Wl,-z,max-page-size=16384"
)

target_link_libraries(mobilei2v_gpu PRIVATE
    MNN
    android
    log
    dl
    EGL
    GLESv2
)
