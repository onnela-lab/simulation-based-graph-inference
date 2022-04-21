#ifndef SIMULATION_BASED_GRAPH_INFERENCE_GENERATORS_HPP_
#define SIMULATION_BASED_GRAPH_INFERENCE_GENERATORS_HPP_

#include <random>
#include "graph.hpp"

namespace SimulationBasedGraphInference {
    inline std::mt19937& get_random_engine() {
        static std::mt19937 random_engine;
        return random_engine;
    }

    inline Graph& generate_poisson_random_attachment(Graph &graph, std::size_t num_nodes, float rate) {
        std::poisson_distribution<size_t> attachment_distribution(rate);
        std::mt19937& engine = get_random_engine();
        for (auto node = graph.get_num_nodes(); node < num_nodes; node++) {
            // Sample the neighbors.
            auto degree = std::min(node, attachment_distribution(engine));
            std::vector<node_t> neighbors;
            std::sample(graph.nodes.begin(), graph.nodes.end(), std::back_inserter(neighbors), degree, engine);
            // Create connections.
            graph.add_node(node);
            for (auto neighbor : neighbors) {
                graph.add_edge(node, neighbor);
            }
        }
        return graph;
    }
}

#endif  // SIMULATION_BASED_GRAPH_INFERENCE_GENERATORS_HPP_
