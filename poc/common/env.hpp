// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#pragma once
// Portable, warning-free environment access (avoid platform ifdefs by
// structure). MSVC deprecates plain getenv (C4996)
// while the C11 getenv_s is not available on the Android NDK - the pragma
// suppression lives HERE, once, with the rationale, instead of being
// duplicated per harness file. The values are process-local harness flags
// (trusted input, never freed).
//
// Not thread-safe (same contract as the C getenv the standard gives us);
// the harnesses are single-threaded by contract.

#include <cstdlib> // getenv / _putenv_s / setenv
#include <string>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996) // getenv CRT deprecation, see above
#endif

namespace bitnet {

// Value of an environment variable as a string; "" when unset.
[[nodiscard]] inline std::string env_string(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string();
}

// true when the variable is set to a non-empty value.
[[nodiscard]] inline bool env_nonempty(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

} // namespace bitnet

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#ifdef _WIN32
namespace bitnet {
// Forward a variable to backends that discover paths via env (see
// BITNET_SPV_DIR). _putenv_s on Windows, setenv on POSIX/NDK.
inline void set_env(const char* name, const char* value) {
    _putenv_s(name, value);
}
} // namespace bitnet
#else
namespace bitnet {
inline void set_env(const char* name, const char* value) {
    setenv(name, value, 1);
}
} // namespace bitnet
#endif
