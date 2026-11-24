#include "bitnet1.hpp"

#include <array>
#include <cstring>
#include <fstream>

namespace bitnet {

namespace {
constexpr std::array<char, 4> kMagic = {'T', 'E', 'T', 'H'};
constexpr uint32_t kVersion = 1;
} // namespace

bool write_bitnet1(const std::string& path, const int8_t* weights, uint32_t n, uint32_t k,
                   const float* gamma) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    f.write(kMagic.data(), 4);
    const std::array<uint32_t, 3> header = {kVersion, n, k};
    f.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
    f.write(reinterpret_cast<const char*>(weights),
            static_cast<std::streamsize>(static_cast<size_t>(n) * k));
    f.write(reinterpret_cast<const char*>(gamma), static_cast<std::streamsize>(n * sizeof(float)));
    return static_cast<bool>(f);
}

bool load_bitnet1(const std::string& path, LoadedWeights& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    std::array<char, 4> magic = {0, 0, 0, 0};
    f.read(magic.data(), 4);
    if (std::memcmp(magic.data(), kMagic.data(), 4) != 0) {
        return false;
    }
    std::array<uint32_t, 3> header = {0, 0, 0}; // version, n, k
    f.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (header[0] != kVersion || header[1] == 0 || header[2] == 0) {
        return false;
    }
    out.n = header[1];
    out.k = header[2];
    out.weights.resize(static_cast<size_t>(out.n) * out.k);
    out.gamma.resize(out.n);
    f.read(reinterpret_cast<char*>(out.weights.data()),
           static_cast<std::streamsize>(out.weights.size()));
    f.read(reinterpret_cast<char*>(out.gamma.data()),
           static_cast<std::streamsize>(out.gamma.size() * sizeof(float)));
    return f.good();
}

} // namespace bitnet
