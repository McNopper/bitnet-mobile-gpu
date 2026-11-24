# BitNet build system (CMake + Ninja): Windows host and Android cross-builds.
#
# Android builds use the NDK CMake toolchain file (see the 'android' preset in
# CMakePresets.json: CMAKE_TOOLCHAIN_FILE + ANDROID_ABI/ANDROID_PLATFORM/
# ANDROID_STL). Targets, flags and linkage stay identical — the toolchain only
# swaps compiler and sysroot. Slang -> SPIR-V compilation (add_slang_kernel)
# always runs on the host via slangc from the Vulkan SDK, so the Android
# preset also needs the SDK installed locally.
#
# The backend libraries and the harness executables are created only when
# their source files exist (if(EXISTS) guards).

cmake_minimum_required(VERSION 3.24)
project(bitnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Compile DB for clang-tidy (verify gate); generated for every preset.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warnings: zero-tolerance gate. Strict set for
# clang/gcc (host + NDK), /W4 fallback for MSVC; warnings are errors via
# CMAKE_COMPILE_WARNING_AS_ERROR (also pinned ON in both CMakePresets.json
# cache variables so configure-time consumers inherit it). Fix at the source;
# never suppress without a code comment explaining why and a dated TODO.
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
  add_compile_options(/W4)
endif()

# slangc ships with the LunarG Vulkan SDK (also used by the Android preset:
# shaders compile on the host, only the C++ targets cross-compile).
find_program(SLANGC_EXE slangc
  HINTS "$ENV{VULKAN_SDK}/Bin" "C:/VulkanSDK/1.4.363.1/Bin")
if(NOT SLANGC_EXE)
  message(FATAL_ERROR
    "slangc not found - install the Vulkan SDK or set VULKAN_SDK "
    "(searched \$ENV{VULKAN_SDK}/Bin and C:/VulkanSDK/1.4.363.1/Bin)")
endif()

# Vulkan linkage: the NDK sysroot ships libvulkan.so, the host uses the SDK
# (headers + loader via imported target).
if(ANDROID)
  find_library(VULKAN_LIB vulkan)
  if(NOT VULKAN_LIB)
    message(FATAL_ERROR "libvulkan not found in the Android NDK sysroot")
  endif()
else()
  find_package(Vulkan REQUIRED)
endif()

# Slang compute kernels -> SPIR-V at <build>/spv/<profile>/<entry>.spv.
# One custom command per (entry, GPU profile); all collected into the ALL
# target 'spv_kernels'. Profiles encode per-GPU tuning knobs:
# slangc -D defines override the kernel defaults. The harness picks the
# profile at runtime from the caps probe (device name -> profile).

# GPU profiles: one explicit block per target GPU. Adding a GPU =
# copying a block and tuning the defines. The harness matches the profile at
# runtime via caps.device_name.

set(BITNET_FIXED_KERNELS
  alu_countbits_probe
  alu_fma_probe
  bw_probe
  null_probe
  stream_bw_probe
  vec_load_probe)

# Kernel families of the ONE Slang source file (every matmul kernel of
# this project). The macro compiles each entry from
# bitnet_kernels.slang; the lists below exist for readability/grouping
# only - CMake does not care which list an entry lives in.
set(BITNET_BITPLANE_ENTRIES
  gemv_bitplane_rowserial
  gemv_bitplane_ksplit
  gemv_bitplane_ksplit_il
  gemv_bitplane_rowcoop
  gemm_bitplane_naive
  gemm_bitplane_ksplit
  gemm_bitplane_cellcoop
  gemm_bitplane_tiled)
set(BITNET_PLANE_ENTRIES
  gemv_plane_rowcoop
  gemm_plane_tiled
  gemm_plane_microtile
  gemm_plane_microtile_image)
set(BITNET_DOT_ENTRIES
  gemv_dot_rowcoop
  gemm_dot_tiled)
set(BITNET_KERNEL_SOURCE "poc/slang/bitnet_kernels.slang")

# SPIR-V post-optimization (Khronos SPIRV-Tools spirv-opt performance
# passes: dead-code elimination, branch simplification, compact IDs).
# Applied when spirv-opt is on PATH (ships with the Vulkan SDK).
find_program(SPIRV_OPT_EXE spirv-opt)

function(add_slang_kernel_variant slang_file entry profile extra_args)
  # Optional 5th arg: output NAME (default = entry). The packed-dot family
  # compiles its SINGLE source twice per profile - hardware + emulated
  # (-DNO_INT_DOT) - so the emulated module needs a different output name
  # while slangc keeps compiling the same entry point.
  set(_out_name "${entry}")
  if(ARGC GREATER 4)
    set(_out_name "${ARGV4}")
  endif()
  if(NOT IS_ABSOLUTE "${slang_file}")
    set(slang_file "${CMAKE_CURRENT_SOURCE_DIR}/${slang_file}")
  endif()
  set(out_dir "${CMAKE_BINARY_DIR}/spv/${profile}")
  set(out "${out_dir}/${_out_name}.spv")
  # NO -fvk-use-scalar-layout: our buffer types (uint/uint4/int arrays, 4xuint
  # uniform block) have IDENTICAL std430/std140 layouts, and std430/std140 are
  # guaranteed by core Vulkan - no scalarBlockLayout feature query/enable
  # needed, no optional-feature edge cases. Verified via spirv-dis
  # (strides 16/4, uniform offsets 0/4/8/12) and the bit-exact harness.
  if(SPIRV_OPT_EXE)
    set(raw "${out_dir}/${_out_name}.raw.spv")
    add_custom_command(
      OUTPUT "${out}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${out_dir}"
      COMMAND "${SLANGC_EXE}" "${slang_file}" -stage compute -entry "${entry}"
              -target spirv ${extra_args} -o "${raw}"
      COMMAND "${SPIRV_OPT_EXE}" -O "${raw}" -o "${out}"
      DEPENDS "${slang_file}"
      COMMENT "slangc[${profile}]+spirv-opt: ${_out_name} -> spv/${profile}/${_out_name}.spv"
      VERBATIM)
  else()
    add_custom_command(
      OUTPUT "${out}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${out_dir}"
      COMMAND "${SLANGC_EXE}" "${slang_file}" -stage compute -entry "${entry}"
              -target spirv ${extra_args} -o "${out}"
      DEPENDS "${slang_file}"
      COMMENT "slangc[${profile}]: ${_out_name} -> spv/${profile}/${_out_name}.spv"
      VERBATIM)
  endif()
  set_property(GLOBAL APPEND PROPERTY BITNET_SPV_OUTPUTS "${out}")
endfunction()

macro(bitnet_compile_kernels_for
      profile device_match gemv_threads gemv6_threads gemv_dot_threads tile_m tile_n chunk
      tile_threads gemv6_rows_per_group v7_rows v7_cols v7_threads gemv_champion
      gemm_champion)
  # Define names MUST match the #ifndef guards in poc/slang/bitnet_kernels.slang
  # - a mismatch silently compiles kernel defaults while the host dispatches
  # for the profile values (the generated gpu_profiles.h and the LocalSize
  # guard of the backend catch it). One value per define - duplicates are
  # last-wins and fragile. Every value the HOST dispatches on is in this
  # list AND in the generated table - no host-side re-derivation. The
  # binding-overlap disable (E39001) is FILE-WIDE because one source
  # carries every family: the weight/activation slots (bindings 0/1)
  # have THREE independent typed views - uint (bitplane/dot), uint4
  # (plane), int8_t (coopmat) - each compiled entry point references
  # EXACTLY ONE view (slangc validates the whole file's declaration list
  # regardless of entry, even though codegen strips the rest).
  set(_defs "-DGEMV_BITPLANE_ROWCOOP_THREADS=${gemv_threads}"
            "-DGEMV_PLANE_ROWCOOP_THREADS=${gemv6_threads}"
            "-DGEMV_DOT_ROWCOOP_THREADS=${gemv_dot_threads}"
            "-DGEMM_BITPLANE_CELLCOOP_THREADS=${gemv_threads}"
            "-DTILE_M=${tile_m}"
            "-DTILE_N=${tile_n}"
            "-DCHUNK=${chunk}"
            "-DTILE_THREADS=${tile_threads}"
            "-DROWS_PER_GROUP=${gemv6_rows_per_group}"
            "-DGEMM_PLANE_MICROTILE_RM=${v7_rows}"
            "-DGEMM_PLANE_MICROTILE_RN=${v7_cols}"
            "-DGEMM_PLANE_MICROTILE_THREADS=${v7_threads}"
            "-warnings-disable" "39001")
  foreach(_entry IN LISTS BITNET_FIXED_KERNELS)
    add_slang_kernel_variant("poc/slang/${_entry}.slang" "${_entry}" "${profile}" "${_defs}")
  endforeach()
  # Every production kernel - one source, one entry point per variant; the
  # same -D knobs go to every compilation (unused ones are dead defines,
  # unreferenced groupshared allocations are stripped).
  foreach(_entry IN LISTS BITNET_BITPLANE_ENTRIES)
    add_slang_kernel_variant("${BITNET_KERNEL_SOURCE}" "${_entry}" "${profile}" "${_defs}")
  endforeach()
  foreach(_entry IN LISTS BITNET_PLANE_ENTRIES)
    add_slang_kernel_variant("${BITNET_KERNEL_SOURCE}" "${_entry}" "${profile}" "${_defs}")
  endforeach()
  # Coop-mat lane: same unified source and knobs (the warning disable
  # above already covers its binding-overlap case).
  add_slang_kernel_variant("${BITNET_KERNEL_SOURCE}" "gemm_coopmat" "${profile}" "${_defs}")
  # Packed-dot emulated variants: the SAME unified
  # source compiled with -DNO_INT_DOT (own multiply-accumulate per int8
  # lane, bit-exact) for devices WITHOUT shaderIntegerDotProduct. The
  # emulated module exists for EVERY profile, so the integer path always
  # runs; selection is capability-driven (caps probe / forced variant) at
  # runtime.
  foreach(_entry IN LISTS BITNET_DOT_ENTRIES)
    add_slang_kernel_variant("${BITNET_KERNEL_SOURCE}" "${_entry}" "${profile}" "${_defs}")
    add_slang_kernel_variant("${BITNET_KERNEL_SOURCE}" "${_entry}" "${profile}"
                             "${_defs};-DNO_INT_DOT" "${_entry}_emulated")
  endforeach()
  # Single source of truth: every macro call also grows the generated
  # gpu_profiles.h row (configure_file at the end of the configure step).
  set_property(GLOBAL APPEND PROPERTY BITNET_PROFILE_ROWS
    "    {\"${device_match}\", \"${profile}\", ${gemv_threads}, ${gemv6_threads}, ${gemv_dot_threads}, ${tile_m}, ${tile_n}, ${chunk}, ${tile_threads}, ${gemv6_rows_per_group}, ${v7_rows}, ${v7_cols}, ${v7_threads}, \"${gemv_champion}\", \"${gemm_champion}\"},")
endmacro()

# Champion kernels per role: GEMV = decode, GEMM = prefill. A champion must
# be legal for EVERY shape of its role, including the 128256-row output
# head. The three mobile profiles are measured (system-level A/B in two
# orders, kernel matrix; report section 5.1).
# - mali-g57: decode gemv_plane_rowcoop (573 ms per token at rows-per-group
#   1; next candidate gemv_bitplane_rowcoop 684); prefill
#   gemm_bitplane_naive (fastest 64-token block of the largest layer:
#   281 ms; gemm_plane_tiled 299).
#   gemv6_rows_per_group 8 measured 2026-11-24: 455 ms/token vs 573 at 1
#   (A/B on the device, -20.5% decode, head 71 vs 94 ms).
# - rtx5070 / default: gemv_bitplane_rowserial (decode) and
#   gemm_plane_tiled (prefill). The desktop is the correctness lane; its
#   champions are not tuned.
bitnet_compile_kernels_for("default" "" 128 128 128 4 32 8 128 1 4 2 64 "gemv_bitplane_rowserial" "gemm_plane_tiled")
bitnet_compile_kernels_for("mali-g57" "Mali-G57" 64 16 32 4 64 16 256 8 4 2 64 "gemv_plane_rowcoop" "gemm_bitplane_naive")
bitnet_compile_kernels_for("rtx5070" "NVIDIA GeForce RTX 5070" 128 128 128 4 32 8 128 1 4 2 64 "gemv_bitplane_rowserial" "gemm_plane_tiled")
# - mali-g52 (Galaxy Tab A8): workgroup limit 384, 32 KiB shared, Vulkan
#   1.1; the mali-g57 knob set fits its limits. Decode
#   gemv_plane_rowcoop (222 ms per token; gemv_bitplane_rowcoop 300), the
#   same champion as on Mali-G57. Prefill gemm_plane_tiled (99 ms for the
#   64-token block of the largest layer; gemm_bitplane_naive 384) - the
#   opposite of Mali-G57: the champion is measured per device.
#   gemv6_rows_per_group 8 measured 2026-11-24: 151 ms/token vs 222 at 1
#   (ABBA on the device, -31/-34% in both orders, head 23 vs 35 ms).
bitnet_compile_kernels_for("mali-g52" "Mali-G52" 64 16 32 4 64 16 256 8 4 2 64 "gemv_plane_rowcoop" "gemm_plane_tiled")
# - adreno740 (Quest 3): workgroup limit 1024, 32 KiB shared, subgroup
#   range 64..128; integer dot present, acceleration flags false. The
#   match string is the EXACT caps device name (gpu_profiles.h.in matches
#   case-insensitively, never by substring). The cooperative-row kernels
#   are the fastest on single layer shapes but dispatch one workgroup per
#   output row: 128256 groups for the head exceed this device's 65535
#   workgroup-count limit and the caps gate rejects them. Decode champion
#   among the legal candidates: gemv_bitplane_ksplit_il (23.4 ms per
#   token, interleaved-16 layout; plain gemv_bitplane_ksplit 43.4) -
#   ABBA on the device 2026-11-24, -45% decode, head 2.9 vs 7.4 ms.
#   Prefill gemm_bitplane_naive (1040 ms per 64-token block;
#   gemm_plane_tiled 1979; gemm_bitplane_tiled and
#   gemm_dot_tiled lose the device: one submission exceeds about 2 s).
bitnet_compile_kernels_for("adreno740" "Adreno (TM) 740" 128 128 128 4 32 8 128 1 4 2 64 "gemv_bitplane_ksplit_il" "gemm_bitplane_naive")

# Tuning-sweep profiles (NOT auto-selected; select with poc_bench --profile
# <name> and the usual --spv-dir root). Constraint set for Mali-G57: TILE_THREADS = TILE_M *
# TILE_N <= 512, shared storage = (TILE_N*2 + TILE_M*8) * CHUNK * 4 B <= 32 KB.
# Sweep profiles carry no champions ("" - never auto-selected).
bitnet_compile_kernels_for("sw-mali-tn64-c32" "" 128 128 128 4 64 32 256 1 4 2 64 "" "")
bitnet_compile_kernels_for("sw-mali-tn128-c16" "" 128 128 128 4 128 16 512 1 4 2 64 "" "")
bitnet_compile_kernels_for("sw-mali-tn64-m8-c16" "" 128 128 128 8 64 16 512 1 4 2 64 "" "")
bitnet_compile_kernels_for("sw-gemv-t64" "" 64 64 64 4 32 8 128 1 4 2 64 "" "")
bitnet_compile_kernels_for("sw-gemv6-t16" "" 16 16 16 4 32 8 128 1 4 2 64 "" "")
bitnet_compile_kernels_for("sw-gemv6-t32" "" 32 32 32 4 32 8 128 1 4 2 64 "" "")
bitnet_compile_kernels_for("sw-gemv-t128" "" 128 128 128 4 32 8 128 1 4 2 64 "" "")
bitnet_compile_kernels_for("sw-gemv-t256" "" 256 256 256 4 32 8 128 1 4 2 64 "" "")
bitnet_compile_kernels_for("sw-gemv-t32" "" 32 32 32 4 32 8 128 1 4 2 64 "" "")
bitnet_compile_kernels_for("sw-mali-tn64-c8-t128" "" 64 64 32 4 64 8 128 1 4 2 64 "" "")
bitnet_compile_kernels_for("sw-mali-tn64-c8-t256" "" 64 64 32 4 64 8 256 1 4 2 64 "" "")

# Single source of truth: gpu_profiles.h is GENERATED from the same
# bitnet_compile_kernels_for values that feed the slangc -D defines
# (hand-maintained copies drift). Included as "backend/gpu_profiles.h".
set(BITNET_PROFILE_ROWS "")
get_property(_rows GLOBAL PROPERTY BITNET_PROFILE_ROWS)
foreach(_r IN LISTS _rows)
  string(APPEND BITNET_PROFILE_ROWS "${_r}
")
endforeach()
configure_file(poc/backend/gpu_profiles.h.in
               ${CMAKE_BINARY_DIR}/generated/backend/gpu_profiles.h @ONLY)


get_property(BITNET_SPV_OUTPUTS GLOBAL PROPERTY BITNET_SPV_OUTPUTS)
add_custom_target(spv_kernels ALL DEPENDS ${BITNET_SPV_OUTPUTS})

# Common core: packed formats, CPU reference, deterministic tensor generation,
# BITNET1 container I/O.
add_library(bitnet_common STATIC
  poc/common/format.cpp
  poc/common/reference.cpp
  poc/common/tensor_gen.cpp
  poc/common/bitnet1.cpp)
target_include_directories(bitnet_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/poc ${CMAKE_BINARY_DIR}/generated)

# Vulkan backend (factory bitnet::make_vulkan_backend).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/poc/vulkan/vulkan_backend.cpp")
  add_library(bitnet_vulkan STATIC
    poc/vulkan/vulkan_backend.cpp
    poc/vulkan/vulkan_dispatch.cpp
    poc/vulkan/vulkan_query.cpp
    poc/vulkan/vulkan_objects.cpp)
  target_include_directories(bitnet_vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/poc)
  if(ANDROID)
    target_link_libraries(bitnet_vulkan PUBLIC ${VULKAN_LIB})
  else()
    target_link_libraries(bitnet_vulkan PUBLIC Vulkan::Vulkan)
  endif()
endif()

# Metal backend: PREPARED-ONLY stub - own folder, no Metal API calls,
# compiles on every platform and reports Metal as unavailable at runtime.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/poc/metal/metal_backend.cpp")
  add_library(bitnet_metal STATIC poc/metal/metal_backend.cpp)
  target_include_directories(bitnet_metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/poc)
endif()

# The seam links both factories (capability-driven selection stays on the
# Vulkan lane; the Metal stub exists so a real Metal backend drops in later
# without touching common code or callers).
if(TARGET bitnet_metal)
  set(BITNET_BACKEND_LIBS bitnet_common bitnet_vulkan bitnet_metal)
else()
  set(BITNET_BACKEND_LIBS bitnet_common bitnet_vulkan)
endif()

# Harness executables. BITNET_SPV_DIR points at the
# spv_kernels output dir (forward slashes); add_dependencies makes the .spv
# files build before the binaries are up to date.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/poc/tests/test_main.cpp" AND TARGET bitnet_vulkan)
  enable_testing()
add_executable(poc_unit_tests poc/tests/unit_tests.cpp)
target_link_libraries(poc_unit_tests PRIVATE ${BITNET_BACKEND_LIBS})
add_test(NAME unit_tests COMMAND poc_unit_tests)
# Full GPU harness (core set; BITNET_EXPERIMENTAL=1 adds the experimental kernels).
# Needs the compiled SPIR-V tree next to the build directory.
add_test(NAME poc_tests COMMAND poc_tests --spv-dir ${CMAKE_BINARY_DIR}/spv)

add_executable(poc_tests
  poc/tests/test_main.cpp
  poc/tests/caps_report.cpp
  poc/tests/coopmat_lane.cpp
  poc/tests/golden_test.cpp
  poc/tests/real_model_test.cpp)
  target_link_libraries(poc_tests PRIVATE ${BITNET_BACKEND_LIBS})
  target_compile_definitions(poc_tests PRIVATE
    "BITNET_SPV_DIR=\"${CMAKE_BINARY_DIR}/spv\""
    "BITNET_GOLDEN_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/data/golden_vector.txt\"")
  add_dependencies(poc_tests spv_kernels)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/poc/bench/bench_main.cpp" AND TARGET bitnet_vulkan)
  add_executable(poc_bench
    poc/bench/bench_main.cpp
    poc/bench/bench_support.cpp
    poc/bench/bench_system.cpp)
  target_link_libraries(poc_bench PRIVATE ${BITNET_BACKEND_LIBS})
  target_compile_definitions(poc_bench PRIVATE "BITNET_SPV_DIR=\"${CMAKE_BINARY_DIR}/spv\"")
  add_dependencies(poc_bench spv_kernels)
endif()

# CPU memory-bandwidth probe: the CPU-side counterpart of poc_bench's GPU
# bw_probe (report section 7). Standalone - no backend, no kernels.
find_package(Threads REQUIRED)
add_executable(poc_membw poc/bench/membw_main.cpp)
target_link_libraries(poc_membw PRIVATE Threads::Threads)

