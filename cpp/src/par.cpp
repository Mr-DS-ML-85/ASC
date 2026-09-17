#include "par.hpp"
#include <algorithm>
#include <numeric>
#include <random>

namespace asc {

PredictiveAttentionRouter::PredictiveAttentionRouter(int d_model, int n_blocks, int top_k)
    : d_model_(d_model), n_blocks_(n_blocks), top_k_(top_k) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    auto init_weights = [&](std::vector<float>& w, int rows, int cols) {
        w.resize(rows * cols);
        for (auto& v : w) v = dist(gen);
    };

    auto init_bias = [&](std::vector<float>& b, int size) {
        b.resize(size, 0.0f);
    };

    // Layer 1: d_model -> 2048
    init_weights(w1_, 2048, d_model_);
    init_bias(b1_, 2048);

    // Layer 2: 2048 -> 2048
    init_weights(w2_, 2048, 2048);
    init_bias(b2_, 2048);

    // Layer 3: 2048 -> n_blocks
    init_weights(w3_, n_blocks_, 2048);
    init_bias(b3_, n_blocks_);
}

std::vector<float> PredictiveAttentionRouter::relu(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::max(0.0f, x[i]);
    }
    return result;
}

std::vector<float> PredictiveAttentionRouter::matvec(const float* W, const float* x, int rows, int cols) {
    std::vector<float> result(rows);
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum += W[r * cols + c] * x[c];
        }
        result[r] = sum;
    }
    return result;
}

PARResult PredictiveAttentionRouter::predict(const float* hidden_state) {
    // Layer 1: ReLU(W1 * x + b1)
    auto h1 = matvec(w1_.data(), hidden_state, 2048, d_model_);
    for (int i = 0; i < 2048; ++i) h1[i] += b1_[i];
    h1 = relu(h1);

    // Layer 2: ReLU(W2 * h1 + b2)
    auto h2 = matvec(w2_.data(), h1.data(), 2048, 2048);
    for (int i = 0; i < 2048; ++i) h2[i] += b2_[i];
    h2 = relu(h2);

    // Layer 3: W3 * h2 + b3
    auto logits = matvec(w3_.data(), h2.data(), n_blocks_, 2048);
    for (int i = 0; i < n_blocks_; ++i) logits[i] += b3_[i];

    // Top-k selection
    std::vector<int32_t> indices(n_blocks_);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k_, indices.end(),
        [&](int a, int b) { return logits[a] > logits[b]; });
    indices.resize(top_k_);

    return {indices, logits};
}

int PredictiveAttentionRouter::param_count() const {
    return 2048 * d_model_ + 2048 +  // Layer 1
           2048 * 2048 + 2048 +      // Layer 2
           n_blocks_ * 2048 + n_blocks_;  // Layer 3
}

}  // namespace asc
