#ifndef SIMULATION_BASED_GRAPH_INFERENCE_GRAPH_HPP_
#define SIMULATION_BASED_GRAPH_INFERENCE_GRAPH_HPP_

#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

namespace SimulationBasedGraphInference {
    using node_t = long;
    using node_set_t = std::set<node_t>;
    using neighbor_map_t = std::map<node_t, node_set_t>;

    class Graph {
        private:

        inline static node_set_t EMPTY_NODESET;

        public:

        node_set_t nodes;
        neighbor_map_t neighbor_map;

        std::size_t get_num_nodes() {
            return this->nodes.size();
        }

        std::size_t get_num_edges() {
            std::size_t num_edges = 0;
            for (auto const& [node, neighbors] : neighbor_map) {
                num_edges += neighbors.size();
            }
            return num_edges / 2;
        }

        void add_node(node_t node) {
            nodes.insert(node);
        }

        bool has_node(node_t node) {
            return nodes.find(node) != nodes.end();
        }

        void assert_has_node(node_t node) {
            if (!has_node(node)) {
                throw std::out_of_range("node does not exist");
            }
        }

        void remove_node(node_t node) {
            assert_has_node(node);
            // Remove all edges from neighbors to this node.
            for (const auto& neighbor : get_neighbors(node)) {
                remove_directed_edge(neighbor, node);
            }
            // Remove all edges from this node to neighbors, and, finally, the node itself.
            neighbor_map.erase(node);
            nodes.erase(node);
        }

        void add_directed_edge(node_t source, node_t target) {
            auto lb = neighbor_map.lower_bound(source);
            if (lb == neighbor_map.end() || source != lb->first) {
                lb = neighbor_map.insert(lb, std::make_pair(source, node_set_t()));
            }
            lb->second.insert(target);
        }

        void remove_directed_edge(node_t source, node_t target) {
            bool exists;
            auto it = neighbor_map.find(source);
            if (it == neighbor_map.end()) {
                exists = false;
            } else {
                exists = it->second.erase(target);
            }
            if (!exists) {
                throw std::out_of_range("edge {} -> {} does not exist");
            }
        }

        void remove_edge(node_t node1, node_t node2) {
            remove_directed_edge(node1, node2);
            remove_directed_edge(node2, node1);
        }

        void add_edge(node_t node1, node_t node2) {
            if (node1 == node2) {
                throw std::runtime_error("self loops are not allowed");
            }
            assert_has_node(node1);
            assert_has_node(node2);
            add_directed_edge(node1, node2);
            add_directed_edge(node2, node1);
        }

        const node_set_t& get_neighbors(node_t node) {
            assert_has_node(node);
            auto it = neighbor_map.find(node);
            if (it == neighbor_map.end()) {
                return EMPTY_NODESET;
            }
            return it->second;
        }

        std::size_t get_degree(node_t node) {
            assert_has_node(node);
            auto it = neighbor_map.find(node);
            if (it == neighbor_map.end()) {
                return 0;
            }
            return it->second.size();
        }
    };
}

#endif  // SIMULATION_BASED_GRAPH_INFERENCE_GRAPH_HPP_
