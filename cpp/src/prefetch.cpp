#include "prefetch.hpp"
#include <cstring>

namespace asc {

StreamingPrefetchQueue::StreamingPrefetchQueue() : n_prefetched_(0), total_bytes_(0) {}

void StreamingPrefetchQueue::prefetch(
    const float* kv_data, int n_layers, int seq_len, int n_heads, int head_dim,
    const int32_t* block_indices, int n_blocks, int block_size) {

    // Simulate prefetch by touching memory regions
    // In real implementation, this would use cudaMemcpAsync
    for (int b = 0; b < n_blocks; ++b) {
        int32_t idx = block_indices[b];
        int start = idx * block_size;
        int end = std::min(start + block_size, seq_len);
        if (start >= seq_len) continue;

        // Touch each layer's KV data for this block
        size_t block_bytes = 0;
        for (int layer = 0; layer < n_layers; ++layer) {
            for (int kv = 0; kv < 2; ++kv) {
                size_t offset = ((size_t)layer * 2 * seq_len * n_heads * head_dim)
                              + ((size_t)kv * seq_len * n_heads * head_dim)
                              + ((size_t)start * n_heads * head_dim);
                size_t bytes = (end - start) * n_heads * head_dim * sizeof(float);
                // Touch memory (prefetch simulation)
                volatile float sink = 0;
                const float* ptr = kv_data + offset / sizeof(float);
                for (size_t i = 0; i < bytes / sizeof(float); i += 64) {
                    sink += ptr[i];
                }
                block_bytes += bytes;
            }
        }
        n_prefetched_++;
        total_bytes_ += block_bytes;
    }
}

void StreamingPrefetchQueue::synchronize() {
    // No-op for CPU simulation
}

PrefetchStats StreamingPrefetchQueue::get_stats() const {
    return {n_prefetched_, total_bytes_};
}

void StreamingPrefetchQueue::reset_stats() {
    n_prefetched_ = 0;
    total_bytes_ = 0;
}

}  // namespace asc
