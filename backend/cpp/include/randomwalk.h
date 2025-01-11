#ifndef RANDOMWALK_H
#define RANDOMWALK_H

#include <vector>
#include <utility>

// Biased Random Walk function
std::vector<std::vector<int>> biased_random_walk(
    const std::vector<std::vector<int>>& successors,
    const std::vector<int>& init_nodes,
    int walk_length,
    float p,
    float q);

// Skip-Gram Pair Generation function
std::vector<std::pair<int, int>> generate_skip_gram_pairs(
    const std::vector<std::vector<int>>& walks, int window_size);

#endif // RANDOMWALK_H

