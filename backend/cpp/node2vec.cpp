#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

namespace py = pybind11;

// Biased Random Walk function
std::vector<std::vector<int>> biased_random_walk(
    const std::vector<std::vector<int>>& successors,
    const std::vector<int>& init_nodes,
    int walk_length,
    float p,
    float q) {
    
    std::vector<std::vector<int>> walks;
    std::mt19937 gen(std::random_device{}());

    for (int init_node : init_nodes) {
        std::vector<int> walk = {init_node};
        while (walk.size() < walk_length) {
            int cur = walk.back();
            const auto& neighbors = successors[cur];
            if (neighbors.empty()) break;

            int next_node;
            if (walk.size() == 1) {  // First step
                std::uniform_int_distribution<> dist(0, neighbors.size() - 1);
                next_node = neighbors[dist(gen)];
            } else {  // Subsequent steps
                int prev = walk[walk.size() - 2];
                std::vector<float> probs;
                for (int neighbor : neighbors) {
                    if (neighbor == prev) {
                        probs.push_back(1.0f / p);
                    } else if (std::find(successors[prev].begin(), successors[prev].end(), neighbor) != successors[prev].end()) {
                        probs.push_back(1.0f);
                    } else {
                        probs.push_back(1.0f / q);
                    }
                }
                // Normalize probabilities
                float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
                for (auto& prob : probs) prob /= sum;

                std::discrete_distribution<> dist(probs.begin(), probs.end());
                next_node = neighbors[dist(gen)];
            }
            walk.push_back(next_node);
        }
        walks.push_back(walk);
    }
    return walks;
}

// Skip-Gram Pair Generation function
std::vector<std::pair<int, int>> generate_skip_gram_pairs(
    const std::vector<std::vector<int>>& walks, int window_size) {
    
    std::vector<std::pair<int, int>> skip_gram_pairs;
    for (const auto& walk : walks) {
        for (size_t i = 0; i < walk.size(); ++i) {
            int target = walk[i];
            for (int j = std::max(0, static_cast<int>(i) - window_size);
                 j <= std::min(static_cast<int>(walk.size()) - 1, static_cast<int>(i) + window_size); ++j) {
                if (i != j) {
                    skip_gram_pairs.emplace_back(target, walk[j]);
                }
            }
        }
    }
    return skip_gram_pairs;
}

PYBIND11_MODULE(node2vec, m) {
    m.def("biased_random_walk", &biased_random_walk, "Biased Random Walk");
    m.def("generate_skip_gram_pairs", &generate_skip_gram_pairs, "Generate Skip-Gram Pairs");
}

