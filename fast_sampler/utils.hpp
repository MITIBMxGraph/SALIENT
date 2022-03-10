#pragma once

#include <torch/torch.h>

#define CHECK_CPU(x) \
  TORCH_INTERNAL_ASSERT(x.device().is_cpu(), #x " must be CPU tensor")
#define CHECK_INPUT(x) TORCH_INTERNAL_ASSERT(x, "Input mismatch")

template <typename scalar_t>
inline torch::Tensor vector_to_tensor(const std::vector<scalar_t>& vec,
                                      bool pin_memory = false) {
  auto tensor = torch::empty(
      vec.size(), torch::TensorOptions()
                      .dtype(torch::CppTypeToScalarType<scalar_t>::value)
                      .device(torch::kCPU)
                      .layout(torch::kStrided)
                      .pinned_memory(pin_memory)
                      .requires_grad(false));
  const auto tensor_data = tensor.template data_ptr<scalar_t>();
  std::copy(vec.begin(), vec.end(), tensor_data);
  return tensor;
}
