cmake_minimum_required(VERSION 3.22.1)

project("tinyvoca" LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Target-specific knobs.  KleidiAI gives a large NEON matvec win on arm64-v8a
# (the tiny-voca text encoder is a handful of small GEMMs).  x86_64 is kept for
# emulator support and simply uses ggml's portable kernels.
# ---------------------------------------------------------------------------
if(DEFINED ANDROID_ABI)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(TINYVOCA_KLEIDIAI ON)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(TINYVOCA_KLEIDIAI OFF)
    else()
        message(FATAL_ERROR "tiny-voca TTS supports only arm64-v8a and x86_64, got ${ANDROID_ABI}")
    endif()
else()
    set(GGML_SYSTEM_ARCH "ARM")
    set(TINYVOCA_KLEIDIAI OFF)
endif()

# Library-only build.  LLAMA_BUILD_TOOLS is required because tools/CMakeLists.txt
# is what adds the mtmd library (which contains the tiny-voca pipeline).
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_HTML     OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS     OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(GGML_METAL           OFF CACHE BOOL "" FORCE)
set(GGML_CUDA            OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN          OFF CACHE BOOL "" FORCE)
set(GGML_SYCL            OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI    ${TINYVOCA_KLEIDIAI} CACHE BOOL "" FORCE)
set(MTMD_VIDEO           OFF CACHE BOOL "" FORCE)

set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../native/llama.cpp)
if(NOT EXISTS ${LLAMA_SRC}/CMakeLists.txt)
    message(FATAL_ERROR "llama.cpp submodule missing - run: git submodule update --init --recursive")
endif()
add_subdirectory(${LLAMA_SRC} llama.cpp)

add_library(tinyvoca_jni SHARED tinyvoca_jni.cpp)

target_compile_definitions(tinyvoca_jni PRIVATE
        GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
        GGML_CPU_KLEIDIAI=$<BOOL:${TINYVOCA_KLEIDIAI}>
        GGML_OPENMP=OFF)

target_include_directories(tinyvoca_jni PRIVATE
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/tools/mtmd)

target_link_libraries(tinyvoca_jni
        llama
        llama-common
        mtmd
        android
        log)
