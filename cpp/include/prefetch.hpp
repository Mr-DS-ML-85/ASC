#pragma once

#include <vector>
#include <cstdint>

namespace asc {

struct PrefetchStats {
    int n_prefetched;
    size_t total_bytes;
};

class StreamingPrefetchQueue {
public:
    StreamingPrefetchQueue();
    ~StreamingPrefetchQueue() = default;

    void prefetch(const float* kv_data, int n_layers, int seq_len, int n_heads, int head_dim,
                  const int32_t* block_indices, int n_blocks, int block_size);
    void synchronize();
    PrefetchStats get_stats() const;
    void reset_stats();

private:
    int n_prefetched_;
    size_t total_bytes_;
};

}  // namespace asc
