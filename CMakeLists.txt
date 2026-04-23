cmake_minimum_required(VERSION 3.10)
project(build_llama)

set(LLAMA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_SRC} build-llama)
if(NOT TARGET common AND NOT TARGET llama-common AND NOT TARGET llama_common)
    message(WARNING "强行加载 common 静态库...")
    add_subdirectory(${LLAMA_SRC}/common build-common)
endif()
add_library(${CMAKE_PROJECT_NAME} SHARED
        library.cpp)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src
        ${LLAMA_SRC}/vendor)
set(COMMON_TARGET_NAME common)

if(TARGET llama-common)
    set(COMMON_TARGET_NAME llama-common)
elseif(TARGET llama_common)
    set(COMMON_TARGET_NAME llama_common)
endif()

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        ${COMMON_TARGET_NAME}
        android
        log)
