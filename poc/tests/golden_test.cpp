// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Known-answer vector of the harness: the hand-computed sample input and
// expected output, checked three ways - against the in-source closed form,
// against the committed vector file, and through every core GPU kernel.

#include "common/reference.hpp"
#include "tests/suites.hpp"

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Known-answer golden vector: a tiny fixed input with
// hand-computed int32 outputs, run through every CORE kernel. The values
// follow the closed form documented in data/golden_vector.txt;
// the committed file is parsed and cross-checked against the SAME closed
// form here, so artifact and in-source expectation cannot drift. The CPU
// oracle must reproduce the hand-computed values (exact int arithmetic),
// and every core GPU kernel must emit them bit-exactly.
constexpr uint32_t kGoldenM = 4;
constexpr uint32_t kGoldenN = 4;
constexpr uint32_t kGoldenK = 64;

constexpr int32_t golden_act(uint32_t /*row*/, uint32_t j) {
    // All activation rows identical: +3 on even columns, -2 on odd.
    return (j % 2 == 0) ? 3 : -2;
}

constexpr int32_t golden_w(uint32_t i, uint32_t j) {
    // +1 when (i+j) is even, -1 otherwise (alternating checkerboard).
    return ((i + j) % 2 == 0) ? 1 : -1;
}

constexpr int32_t golden_out(uint32_t i) {
    // K=64: 32 even columns * (+3 * +/-1) + 32 odd columns * (-2 * -/+1).
    return (i % 2 == 0) ? 160 : -160;
}

struct GoldenData {
    std::vector<int8_t> act;     // kGoldenM * kGoldenK
    std::vector<int8_t> w;       // kGoldenN * kGoldenK
    std::vector<int32_t> expect; // kGoldenN (same vector per activation row)
};

static GoldenData make_golden_data() {
    GoldenData g;
    g.act.resize(static_cast<size_t>(kGoldenM) * kGoldenK);
    g.w.resize(static_cast<size_t>(kGoldenN) * kGoldenK);
    g.expect.resize(kGoldenN);
    for (uint32_t r = 0; r < kGoldenM; ++r) {
        for (uint32_t j = 0; j < kGoldenK; ++j) {
            g.act[static_cast<size_t>(r) * kGoldenK + j] = static_cast<int8_t>(golden_act(r, j));
        }
    }
    for (uint32_t i = 0; i < kGoldenN; ++i) {
        for (uint32_t j = 0; j < kGoldenK; ++j) {
            g.w[static_cast<size_t>(i) * kGoldenK + j] = static_cast<int8_t>(golden_w(i, j));
        }
        g.expect[i] = golden_out(i);
    }
    return g;
}

// Parses the committed golden file: "<key> <int> <int> ..." lines; keys:
// shape_m/shape_n/shape_k (single value) and act/w/gemm_expect/gemv_expect
// (repeated lines accumulate). Returns false on unreadable files.
static bool parse_golden_file(const std::string& path, uint32_t& m, uint32_t& n, uint32_t& k,
                              std::vector<int32_t>& act, std::vector<int32_t>& w,
                              std::vector<int32_t>& gemm_expect,
                              std::vector<int32_t>& gemv_expect) {
    std::ifstream f(path);
    if (!f) {
        std::printf("  golden file NOT FOUND: %s\n", path.c_str());
        return false;
    }
    const std::map<std::string, std::vector<int32_t>*> lists{
        {"act", &act}, {"w", &w}, {"gemm_expect", &gemm_expect}, {"gemv_expect", &gemv_expect}};
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "shape_m") {
            ss >> m;
            continue;
        }
        if (key == "shape_n") {
            ss >> n;
            continue;
        }
        if (key == "shape_k") {
            ss >> k;
            continue;
        }
        const auto entry = lists.find(key);
        if (entry == lists.end()) {
            std::printf("  golden file: unknown key '%s'\n", key.c_str());
            return false;
        }
        int32_t v = 0;
        while (ss >> v) {
            entry->second->push_back(v);
        }
    }
    return m != 0U && n != 0U && k != 0U;
}

// Step 1: the CPU oracle must reproduce the hand-computed values exactly.
static bool golden_oracle_matches(const GoldenData& g) {
    std::vector<int32_t> gemv_ref(kGoldenN, 0);
    std::vector<int32_t> gemm_ref(static_cast<size_t>(kGoldenM) * kGoldenN, 0);
    bitnet::matmul_ref(g.w.data(), g.act.data(), gemm_ref.data(), kGoldenM, kGoldenN, kGoldenK);
    bitnet::matmul_ref(g.w.data(), g.act.data(), gemv_ref.data(), 1, kGoldenN, kGoldenK);
    for (size_t i = 0; i < gemm_ref.size(); ++i) {
        if (gemm_ref[i] != g.expect[i % kGoldenN]) {
            std::printf("  FAIL: CPU oracle row %zu disagrees with the hand-computed "
                        "value (%d != %d)\n",
                        i / kGoldenN, gemm_ref[i], g.expect[i % kGoldenN]);
            return false;
        }
    }
    for (uint32_t i = 0; i < kGoldenN; ++i) {
        if (gemv_ref[i] != g.expect[i]) {
            std::printf("  FAIL: CPU GEMV oracle disagrees (%d != %d)\n", gemv_ref[i], g.expect[i]);
            return false;
        }
    }
    return true;
}

// Step 2: the committed artifact must match the in-source closed form.
static bool golden_file_matches(const std::string& path, const GoldenData& g) {
    uint32_t fm = 0;
    uint32_t fn = 0;
    uint32_t fk = 0;
    std::vector<int32_t> fact;
    std::vector<int32_t> fw;
    std::vector<int32_t> fexpect_gemm;
    std::vector<int32_t> fexpect_gemv;
    if (!parse_golden_file(path, fm, fn, fk, fact, fw, fexpect_gemm, fexpect_gemv)) {
        std::printf("  FAIL: golden file unreadable or incomplete\n");
        return false;
    }
    // The file stores ONE activation row (all m rows are identical by
    // construction - documented in the file header); the weight matrix and
    // the expectation blocks are complete.
    bool ok = fm == kGoldenM && fn == kGoldenN && fk == kGoldenK &&
              fact.size() == static_cast<size_t>(kGoldenK) && fw.size() == g.w.size() &&
              fexpect_gemv.size() == kGoldenN &&
              fexpect_gemm.size() == static_cast<size_t>(kGoldenM) * kGoldenN;
    for (uint32_t j = 0; ok && j < kGoldenK; ++j) {
        ok = fact[j] == g.act[j];
    }
    for (size_t i = 0; ok && i < g.w.size(); ++i) {
        ok = fw[i] == g.w[i];
    }
    for (uint32_t i = 0; ok && i < kGoldenN; ++i) {
        ok = fexpect_gemv[i] == g.expect[i];
    }
    for (size_t i = 0; ok && i < fexpect_gemm.size(); ++i) {
        ok = fexpect_gemm[i] == g.expect[i % kGoldenN];
    }
    if (!ok) {
        std::printf("  FAIL: golden file contents drift from the in-source closed form\n");
    }
    return ok;
}

// Step 3: every core GPU kernel must emit the hand-computed values (the
// shared compare_tensor path dispatches every applicable core kernel and compares bit-exactly
// against the oracle; the oracle is proven equal to the hand values in
// step 1, and the GPU outputs are additionally compared against them).
static bool golden_gpu_matches(const GoldenData& g) {
    for (const uint32_t m : {1U, kGoldenM}) {
        const Shape s{.m = m, .n = kGoldenN, .k = kGoldenK, .binary = false};
        bitnet::TernaryTensor t;
        t.m = m;
        t.n = kGoldenN;
        t.k = kGoldenK;
        t.binary = false;
        t.weights = g.w;
        t.activations.resize(static_cast<size_t>(m) * kGoldenK);
        for (uint32_t r = 0; r < m; ++r) {
            for (uint32_t j = 0; j < kGoldenK; ++j) {
                t.activations[static_cast<size_t>(r) * kGoldenK + j] =
                    static_cast<int8_t>(golden_act(r, j));
            }
        }
        try {
            std::vector<int32_t> gpu_out;
            const int64_t mm = compare_tensor(s, t, gpu_out);
            bool exact = mm == 0;
            for (size_t i = 0; exact && i < gpu_out.size(); ++i) {
                exact = gpu_out[i] == g.expect[i % kGoldenN];
            }
            if (!exact) {
                std::printf("  FAIL: GPU output disagrees with the hand-computed values "
                            "(m=%u, %lld mismatches)\n",
                            m, static_cast<long long>(mm));
                return false;
            }
        } catch (const std::exception& e) {
            std::printf("  FAIL: %s\n", e.what());
            return false;
        }
    }
    return true;
}

int run_golden_vector_test(const std::string& golden_path) {
    std::printf("[golden] known-answer vector m=1/%u n=%u k=%u (hand-computed)\n", kGoldenM,
                kGoldenN, kGoldenK);
    const GoldenData g = make_golden_data();
    const bool ok =
        golden_oracle_matches(g) && golden_file_matches(golden_path, g) && golden_gpu_matches(g);
    if (ok) {
        std::printf("  golden file verified against the in-source closed form (%s)\n",
                    golden_path.c_str());
    }
    // Sample record: a reviewer sees input, output and expectation.
    std::printf("  sample act[0][0..7]: ");
    for (uint32_t j = 0; j < 8; ++j) {
        std::printf("%d ", golden_act(0, j));
    }
    std::printf("\n  sample w[0][0..7]:   ");
    for (uint32_t j = 0; j < 8; ++j) {
        std::printf("%d ", golden_w(0, j));
    }
    std::printf("\n  expected per output: %+d / %+d (alternating)\n", golden_out(0), golden_out(1));
    if (ok) {
        std::printf("  PASS (golden: file + oracle + every applicable core kernel bit-exact)\n");
        return 0;
    }
    return 1;
}