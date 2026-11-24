// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// CPU memory-bandwidth probe -> CSV on stdout.
//
// Question (report section 7): how many bytes per second can T CPU threads
// read from DRAM on this device? It is the CPU-side counterpart of
// poc_bench's GPU bw_probe and the denominator of the "is CPU decode
// bandwidth-bound?" check: a dense decode step reads every weight byte once
// per token, so tokens/s <= read bandwidth / bytes per token.
//
// Method (sustained sequential read, in the spirit of STREAM, McCalpin
// 1995): every thread owns a private buffer (default 64 MB, far beyond the
// caches of the target SoCs) and sums it in 1 MB chunks for a fixed time
// window. Threads start together; each records its own byte count and
// elapsed time. The reported rate is the SUM of the per-thread rates, so a
// slow core lowers the total by its own share only. Threads are not pinned:
// the scheduler places them, as in an unpinned llama.cpp run.
//
// Work-was-done guard: the buffers are filled so that the words of every
// chunk add up (mod 2^32) to a known constant; a thread's checksum must
// equal chunk count x constant, otherwise the run fails (exit 1). A
// negative control at the start of every run alters one word and requires
// the guard to reject that buffer.
//
// CSV columns:
//   probe,threads,buffer_mb,window_ms,reps,gb_per_s_avg,gb_per_s_min,gb_per_s_max

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <latch>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr double kBytesPerGB = 1e9;         // decimal GB, as every bandwidth figure in this repo
constexpr size_t kChunkWords = 250000;      // 1 MB of uint32 between two stop-flag checks
constexpr uint32_t kChunkSum = 0x5A5A5A5BU; // sum (mod 2^32) of every filled chunk; odd
constexpr uint32_t kLcgMul = 1664525U;      // linear congruential generator: data pattern only
constexpr uint32_t kLcgAdd = 1013904223U;
constexpr uint32_t kMaxThreads = 64;
constexpr uint32_t kMaxBufferMb = 1024;
constexpr uint32_t kMinWindowMs = 100;
constexpr uint32_t kMaxWindowMs = 60000;
constexpr uint32_t kMaxReps = 20;

struct Args {
    std::vector<uint32_t> threads; // thread counts to measure; empty = 1, 2, 4, all cores
    uint32_t buffer_mb = 64;
    uint32_t window_ms = 1000;
    uint32_t reps = 3;
};

uint32_t parse_u32(std::string_view text, uint32_t lo, uint32_t hi, const char* what) {
    uint32_t value = 0;
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc() || ptr != text.data() + text.size() || value < lo || value > hi) {
        throw std::invalid_argument(std::string(what) + " expects an integer in [" +
                                    std::to_string(lo) + ", " + std::to_string(hi) + "], got '" +
                                    std::string(text) + "'");
    }
    return value;
}

std::vector<uint32_t> parse_thread_list(std::string_view list) {
    std::vector<uint32_t> counts;
    while (!list.empty()) {
        const size_t comma = list.find(',');
        counts.push_back(parse_u32(list.substr(0, comma), 1, kMaxThreads, "--threads"));
        list = comma == std::string_view::npos ? std::string_view() : list.substr(comma + 1);
    }
    if (counts.empty()) {
        throw std::invalid_argument("--threads expects a comma-separated list of thread counts");
    }
    return counts;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i += 2) {
        const std::string_view option = argv[i];
        if (option != "--threads" && option != "--mb" && option != "--window-ms" &&
            option != "--reps") {
            throw std::invalid_argument("unknown option '" + std::string(option) +
                                        "' (options: --threads LIST --mb N --window-ms N "
                                        "--reps N)");
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for '" + std::string(option) + "'");
        }
        const std::string_view value = argv[i + 1];
        if (option == "--threads") {
            args.threads = parse_thread_list(value);
        } else if (option == "--mb") {
            args.buffer_mb = parse_u32(value, 1, kMaxBufferMb, "--mb");
        } else if (option == "--window-ms") {
            args.window_ms = parse_u32(value, kMinWindowMs, kMaxWindowMs, "--window-ms");
        } else {
            args.reps = parse_u32(value, 1, kMaxReps, "--reps");
        }
    }
    return args;
}

// 1, 2, 4 and all cores (deduplicated, ascending) - the thread counts of the
// llama-bench CPU baseline plus the full machine.
std::vector<uint32_t> default_thread_counts() {
    const uint32_t cores = std::max(1U, std::thread::hardware_concurrency());
    std::vector<uint32_t> counts;
    for (const uint32_t candidate : {1U, 2U, 4U, cores}) {
        if (candidate <= cores && std::ranges::find(counts, candidate) == counts.end()) {
            counts.push_back(candidate);
        }
    }
    std::ranges::sort(counts);
    return counts;
}

// Fills `words` (a whole number of chunks) with a non-constant pattern whose
// sum over every chunk is kChunkSum (mod 2^32): the last word of a chunk
// closes the sum.
void fill_buffer(std::span<uint32_t> words, uint32_t seed) {
    uint32_t value = seed;
    for (size_t begin = 0; begin < words.size(); begin += kChunkWords) {
        const std::span<uint32_t> chunk = words.subspan(begin, kChunkWords);
        uint32_t running = 0U;
        for (uint32_t& word : chunk.first(kChunkWords - 1U)) {
            value = value * kLcgMul + kLcgAdd;
            word = value;
            running += value;
        }
        chunk.back() = kChunkSum - running;
    }
}

struct WorkerResult {
    uint64_t chunks = 0;
    double seconds = 0.0;
    uint32_t checksum = 0;
};

// The reader threads of ONE measurement window. The destructor stops and
// joins whatever was started, so a failed thread creation cannot leave a
// joinable std::thread behind (which would call std::terminate).
class WorkerGroup {
  public:
    explicit WorkerGroup(size_t threads)
        : ready_(static_cast<std::ptrdiff_t>(threads)), results_(threads) {
        threads_.reserve(threads);
    }
    WorkerGroup(const WorkerGroup&) = delete;
    WorkerGroup& operator=(const WorkerGroup&) = delete;
    WorkerGroup(WorkerGroup&&) = delete;
    WorkerGroup& operator=(WorkerGroup&&) = delete;
    ~WorkerGroup() { stop_and_join(); }

    // Starts one reader per buffer, releases all of them together, lets them
    // read for `window`, stops them and returns the per-thread results.
    std::span<const WorkerResult> run(std::span<const std::vector<uint32_t>> buffers,
                                      std::chrono::milliseconds window) {
        for (size_t i = 0; i < results_.size(); ++i) {
            threads_.emplace_back([this, words = std::span<const uint32_t>(buffers[i]),
                                   &result = results_[i]] { read_loop(words, result); });
        }
        ready_.wait();
        release();
        std::this_thread::sleep_for(window);
        stop_and_join();
        return results_;
    }

  private:
    void release() {
        if (!released_) {
            released_ = true;
            start_.count_down();
        }
    }

    void stop_and_join() {
        stop_.store(true, std::memory_order_relaxed);
        release();
        for (std::thread& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    void read_loop(std::span<const uint32_t> words, WorkerResult& result) {
        ready_.count_down();
        start_.wait();
        const auto begin_time = std::chrono::steady_clock::now();
        uint32_t checksum = 0U;
        uint64_t chunks = 0U;
        size_t offset = 0;
        while (!stop_.load(std::memory_order_relaxed)) {
            const std::span<const uint32_t> chunk = words.subspan(offset, kChunkWords);
            checksum = std::accumulate(chunk.begin(), chunk.end(), checksum);
            ++chunks;
            offset = offset + kChunkWords < words.size() ? offset + kChunkWords : 0;
        }
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - begin_time;
        result = {.chunks = chunks, .seconds = elapsed.count(), .checksum = checksum};
    }

    std::latch ready_;
    std::latch start_{1};
    std::atomic<bool> stop_{false};
    bool released_ = false; // main thread only
    std::vector<WorkerResult> results_;
    std::vector<std::thread> threads_;
};

struct Sample {
    double gb_per_s = 0.0; // sum of the per-thread rates
    bool checksum_ok = true;
    std::vector<double> per_thread_gb_per_s;
};

// One window with one reader per buffer.
Sample measure(std::span<const std::vector<uint32_t>> buffers, std::chrono::milliseconds window) {
    WorkerGroup group(buffers.size());
    Sample sample;
    for (const WorkerResult& result : group.run(buffers, window)) {
        const double bytes = static_cast<double>(result.chunks) *
                             static_cast<double>(kChunkWords * sizeof(uint32_t));
        const double rate = result.seconds > 0.0 ? bytes / kBytesPerGB / result.seconds : 0.0;
        sample.per_thread_gb_per_s.push_back(rate);
        sample.gb_per_s += rate;
        // c chunks, each adding up to kChunkSum: c x kChunkSum (mod 2^32).
        const uint32_t expected = static_cast<uint32_t>(result.chunks) * kChunkSum;
        sample.checksum_ok = sample.checksum_ok && result.checksum == expected;
    }
    return sample;
}

// All repetitions of one thread count -> one CSV row plus the per-thread
// rates of the last repetition. Returns false on a checksum mismatch.
bool report_thread_count(std::span<const std::vector<uint32_t>> buffers, const Args& args) {
    std::vector<double> rates;
    Sample last;
    bool checksums_ok = true;
    for (uint32_t rep = 0; rep < args.reps; ++rep) {
        last = measure(buffers, std::chrono::milliseconds(args.window_ms));
        rates.push_back(last.gb_per_s);
        checksums_ok = checksums_ok && last.checksum_ok;
    }
    const double average =
        std::accumulate(rates.begin(), rates.end(), 0.0) / static_cast<double>(rates.size());
    const auto [lowest, highest] = std::ranges::minmax(rates);
    std::printf("membw_read,%zu,%u,%u,%u,%.3f,%.3f,%.3f\n", buffers.size(), args.buffer_mb,
                args.window_ms, args.reps, average, lowest, highest);
    std::printf("#   per-thread GB/s (last repetition):");
    for (const double rate : last.per_thread_gb_per_s) {
        std::printf(" %.3f", rate);
    }
    std::printf("\n");
    return checksums_ok;
}

// Negative control for the work-was-done guard (once per run): a reader of
// a buffer with ONE altered word must be rejected. A guard that cannot fail
// proves nothing; this line in the log is the evidence that it can. The
// altered word is read r >= 1 times, which shifts the checksum by r (mod
// 2^32) - never zero within a window.
bool guard_negative_control(std::vector<std::vector<uint32_t>>& buffers) {
    buffers.front().front() += 1U;
    const Sample sample = measure(std::span<const std::vector<uint32_t>>(buffers.data(), 1),
                                  std::chrono::milliseconds(kMinWindowMs));
    buffers.front().front() -= 1U;
    const bool rejected = !sample.checksum_ok;
    std::printf("membw,negative_control,altered_word_rejected=%u\n", rejected ? 1U : 0U);
    return rejected;
}

int run(const Args& args) {
    const std::vector<uint32_t> counts =
        args.threads.empty() ? default_thread_counts() : args.threads;
    const uint32_t max_threads = std::ranges::max(counts);
    std::vector<std::vector<uint32_t>> buffers(max_threads);
    for (uint32_t t = 0; t < max_threads; ++t) {
        buffers[t].resize(static_cast<size_t>(args.buffer_mb) * kChunkWords);
        fill_buffer(buffers[t], t + 1U);
    }
    std::printf("# cpu memory-bandwidth probe: hardware_concurrency=%u, %u MB private buffer per "
                "thread, %u ms window, %u repetitions, threads not pinned\n",
                std::thread::hardware_concurrency(), args.buffer_mb, args.window_ms, args.reps);
    if (!guard_negative_control(buffers)) {
        std::printf("FAIL: negative control - a buffer with an altered word passed the checksum "
                    "guard\n");
        return 1;
    }
    std::printf("probe,threads,buffer_mb,window_ms,reps,gb_per_s_avg,gb_per_s_min,gb_per_s_max\n");
    bool checksums_ok = true;
    for (const uint32_t threads : counts) {
        const std::span<const std::vector<uint32_t>> used(buffers.data(), threads);
        checksums_ok = report_thread_count(used, args) && checksums_ok;
    }
    if (!checksums_ok) {
        std::printf("FAIL: checksum mismatch - a reader did not read the bytes it reported\n");
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 2;
    }
}
