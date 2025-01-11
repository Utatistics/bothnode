#include "randomwalk.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(node2vec, m) {
    m.def("biased_random_walk", &biased_random_walk, "Biased Random Walk");
    m.def("generate_skip_gram_pairs", &generate_skip_gram_pairs, "Generate Skip-Gram Pairs");
}

