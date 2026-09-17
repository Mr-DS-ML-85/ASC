#pragma once

#include <vector>
#include <string>

namespace asc {

struct BandwidthResult {
    std::string mode;
    size_t size_bytes;
    double time_ms;
    double bandwidth_gbps;
};

struct DecodeResult {
    std::string model;
    int n_ctx;
    int fill_len;
    double avg_ms;
    double tokens_per_sec;
    double kv_size_mb;
};

std::vector<BandwidthResult> benchmark_dram_bandwidth(
    const std::vector<int>& sizes_mb, int n_iter = 20);

DecodeResult benchmark_decode(
    const std::string& model_path, int n_ctx,
    int n_threads = 8, int n_gpu_layers = 99);

}  // namespace asc
