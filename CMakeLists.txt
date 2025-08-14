cmake_minimum_required(VERSION 3.18...3.31)

# Project info
project(fft_comparator
  VERSION 1.0
  LANGUAGES CXX CUDA
)

# C++ and CUDA Standard v.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Flags config
if(MSVC)
  set(cpp_flags /W4 /WX)
  # set(cuda_flags -Xcompiler=/W4;-Xcompiler=/WX)
  set(cuda_flags -Xcompiler=/W4)
else()
  set(cpp_flags -Wall -Wextra -Werror)
  set(cuda_flags -Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-Werror)
endif()

# FetchContent: CLI11 | Find FFTW3 i cuFFT
include(FetchContent)
FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v2.2.0
)
FetchContent_MakeAvailable(CLI11)

find_package(CUDAToolkit REQUIRED)
find_package(FFTW3 QUIET)

#
add_executable(fft_comparator
  src/main.cpp
  src/fft_cpu.cpp
  src/fft_gpu.cu
  src/wav_signal.cpp
)


if(FFTW3_FOUND)
    message(STATUS "Found FFTW: ${FFTW3_LIBRARIES}")
    target_include_directories(fft_comparator PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_link_libraries(fft_comparator PRIVATE ${FFTW3_LIBRARIES})
else()
    message(WARNING "FFTW not found, falling back to manual configuration")

    # Ręczne ścieżki — możesz je ustawić przez zmienne lub domyślnie
    # set(FFTW3_INCLUDE_DIR /usr/include)
    # set(FFTW3_LIBRARY /usr/lib/x86_64-linux-gnu/libfftw3.so)
    # FFTW3 manual integration
    include_directories(/usr/include)
    link_directories(/usr/lib/x86_64-linux-gnu)
    set(FFTW3_INCLUDE_DIR /usr/include)

    target_include_directories(fft_comparator PRIVATE ${FFTW3_INCLUDE_DIR})
    target_link_libraries(fft_comparator PRIVATE fftw3)

endif()




#Include directories
target_include_directories(fft_comparator PRIVATE
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  ${CMAKE_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(fft_comparator PRIVATE
  CUDA::cufft
  CLI11::CLI11
)

# CUDA config
set_target_properties(fft_comparator PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_PROPAGATE_HOST_FLAGS OFF
)

# Target flags
target_compile_options(fft_comparator PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${cpp_flags}>
  $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)
