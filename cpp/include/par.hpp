#pragma once

#include <vector>
#include <cstdint>

namespace asc {

struct PARResult {
    std::vector<int32_t> top_indices;
    std::vector<float> logits;
};

class PredictiveAttentionRouter {
public:
    PredictiveAttentionRouter(int d_model, int n_blocks, int top_k = 8);
    ~PredictiveAttentionRouter() = default;

    PARResult predict(const float* hidden_state);
    int param_count() const;
    int d_model() const { return d_model_; }
    int n_blocks() const { return n_blocks_; }
    int top_k() const { return top_k_; }

private:
    int d_model_;
    int n_blocks_;
    int top_k_;

    // MLP weights [d_model -> 2048 -> 2048 -> n_blocks]
    std::vector<float> w1_, b1_;
    std::vector<float> w2_, b2_;
    std::vector<float> w3_, b3_;

    std::vector<float> relu(const std::vector<float>& x);
    std::vector<float> matvec(const float* W, const float* x, int rows, int cols);
};

}  // namespace asc
