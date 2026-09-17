#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "par.hpp"
#include "prefetch.hpp"
#include "benchmark.hpp"

namespace py = pybind11;

PYBIND11_MODULE(asc_cpp, m) {
    m.doc() = "Attention Stream Cache — C++ backend";

    // PAR
    py::class_<asc::PredictiveAttentionRouter>(m, "PredictiveAttentionRouter")
        .def(py::init<int, int, int>(), py::arg("d_model"), py::arg("n_blocks"), py::arg("top_k") = 8)
        .def("predict", [](asc::PredictiveAttentionRouter& self, std::vector<float>& hidden) {
            auto result = self.predict(hidden.data());
            return py::make_tuple(result.top_indices, result.logits);
        })
        .def("param_count", &asc::PredictiveAttentionRouter::param_count)
        .def_readonly("d_model", &asc::PredictiveAttentionRouter::d_model)
        .def_readonly("n_blocks", &asc::PredictiveAttentionRouter::n_blocks)
        .def_readonly("top_k", &asc::PredictiveAttentionRouter::top_k);

    // Prefetch Queue
    py::class_<asc::StreamingPrefetchQueue>(m, "StreamingPrefetchQueue")
        .def(py::init<>())
        .def("get_stats", &asc::StreamingPrefetchQueue::get_stats)
        .def("reset_stats", &asc::StreamingPrefetchQueue::reset_stats);

    // DRAM Benchmark
    m.def("benchmark_dram_bandwidth", &asc::benchmark_dram_bandwidth,
          py::arg("sizes_mb"), py::arg("n_iter") = 20);

    // BandwidthResult
    py::class_<asc::BandwidthResult>(m, "BandwidthResult")
        .def_readonly("mode", &asc::BandwidthResult::mode)
        .def_readonly("size_bytes", &asc::BandwidthResult::size_bytes)
        .def_readonly("time_ms", &asc::BandwidthResult::time_ms)
        .def_readonly("bandwidth_gbps", &asc::BandwidthResult::bandwidth_gbps);
}
