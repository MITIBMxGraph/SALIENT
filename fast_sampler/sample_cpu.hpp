#pragma once

#include <torch/torch.h>

#include <random>
#include <tuple>

#include "parallel_hashmap/phmap.h"
#include "utils.hpp"

thread_local std::mt19937 gen;

inline auto get_initial_sample_adj_hash_map(const std::vector<int64_t>& n_ids) {
  phmap::flat_hash_map<int64_t, int64_t> n_id_map;
  for (size_t i = 0; i < n_ids.size(); ++i) {
    n_id_map[n_ids[i]] = i;
  }
  return n_id_map;
}

using SingleSample = std::tuple<torch::Tensor, torch::Tensor,
                                std::vector<int64_t>, torch::Tensor>;

// Returns `rowptr`, `col`, `n_id`, `e_id`
inline SingleSample sample_adj(torch::Tensor rowptr, torch::Tensor col,
                               std::vector<int64_t> n_ids,
                               phmap::flat_hash_map<int64_t, int64_t>& n_id_map,
                               int64_t num_neighbors, bool replace,
                               bool pin_memory = false) {
  const auto idx_size = n_ids.size();

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  const auto col_data = col.data_ptr<int64_t>();

  auto out_rowptr =
      torch::empty(idx_size + 1, rowptr.options().pinned_memory(pin_memory));
  const auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();
  out_rowptr_data[0] = 0;

  // adjacency vector of (col, e_id)
  std::vector<std::vector<std::tuple<int64_t, int64_t>>> cols(idx_size);

  const auto expand_neighborhood = [&](auto add_neighbors) -> void {
    for (size_t i = 0; i < idx_size; ++i) {
      const auto n = n_ids[i];
      const auto row_start = rowptr_data[n];
      const auto row_end = rowptr_data[n + 1];
      const auto neighbor_count = row_end - row_start;

      const auto add_neighbor = [&](const int64_t p) -> void {
        const auto e = row_start + p;
        const auto c = col_data[e];

        auto ins = n_id_map.insert({c, n_ids.size()});
        if (ins.second) {
          n_ids.push_back(c);
        }
        cols[i].push_back(std::make_tuple(ins.first->second, e));
      };

      add_neighbors(neighbor_count, add_neighbor);
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }
  };

  if (num_neighbors <
      0) {  // No sampling ======================================
    expand_neighborhood([](const int64_t neighbor_count, auto add_neighbor) {
      for (int64_t j = 0; j < neighbor_count; j++) {
        add_neighbor(j);
      }
    });
  } else if (replace) {  // Sample with replacement
                         // =============================
    expand_neighborhood(
        [num_neighbors](const int64_t neighbor_count, auto add_neighbor) {
          if (neighbor_count <= 0) return;
          for (int64_t j = 0; j < num_neighbors; j++) {
            add_neighbor(gen() % neighbor_count);
          }
        });
  } else {  // Sample without replacement via Robert Floyd algorithm
            // ============
    std::vector<int64_t> perm;
    perm.reserve(num_neighbors);
    expand_neighborhood([num_neighbors, &perm](const int64_t neighbor_count,
                                               auto add_neighbor) {
      perm.clear();

      if (neighbor_count <= num_neighbors) {
        for (int64_t j = 0; j < neighbor_count; j++) {
          add_neighbor(j);
        }
      } else {  // See: https://www.nowherenearithaca.com/2013/05/
                //      robert-floyds-tiny-and-beautiful.html
        for (int64_t j = neighbor_count - num_neighbors; j < neighbor_count;
             j++) {
          const int64_t option = gen() % j;
          auto winner = option;
          if (std::find(perm.cbegin(), perm.cend(), option) == perm.cend()) {
            perm.push_back(option);
            winner = option;
          } else {
            perm.push_back(j);
            winner = j;
          }

          add_neighbor(winner);
        }
      }
    });
  }

  const auto E = out_rowptr_data[idx_size];
  auto out_col = torch::empty(E, col.options().pinned_memory(pin_memory));
  const auto out_col_data = out_col.data_ptr<int64_t>();
  auto out_e_id = torch::empty(E, col.options().pinned_memory(pin_memory));
  const auto out_e_id_data = out_e_id.data_ptr<int64_t>();

  {
    size_t i = 0;
    for (auto& col_vec : cols) {
      std::sort(col_vec.begin(), col_vec.end(),
                [](const auto& a, const auto& b) -> bool {
                  return std::get<0>(a) < std::get<0>(b);
                });
      for (const auto& value : col_vec) {
        out_col_data[i] = std::get<0>(value);
        out_e_id_data[i] = std::get<1>(value);
        i += 1;
      }
    }
  }

  return std::make_tuple(std::move(out_rowptr), std::move(out_col),
                         std::move(n_ids), std::move(out_e_id));
}

inline SingleSample sample_adj(torch::Tensor rowptr, torch::Tensor col,
                               std::vector<int64_t> n_ids,
                               int64_t num_neighbors, bool replace,
                               bool pin_memory = false) {
  auto n_id_map = get_initial_sample_adj_hash_map(n_ids);
  return sample_adj(std::move(rowptr), std::move(col), std::move(n_ids),
                    n_id_map, num_neighbors, replace, pin_memory);
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
           int64_t num_neighbors, bool replace, bool pin_memory = false) {
  const auto idx_data = idx.data_ptr<int64_t>();
  auto res = sample_adj(std::move(rowptr), std::move(col),
                        {idx_data, idx_data + idx.numel()}, num_neighbors,
                        replace, pin_memory);
  auto& n_ids = std::get<2>(res);
  return std::make_tuple(std::move(std::get<0>(res)),
                         std::move(std::get<1>(res)), vector_to_tensor(n_ids),
                         std::move(std::get<3>(res)));
}
