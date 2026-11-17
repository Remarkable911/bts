#include <torch/extension.h>
#include <optional>
void select_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&,
                 const std::optional<at::Tensor>&);
void select_deterministic_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&,
                               const std::optional<at::Tensor>&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("select", &select_cuda);
  m.def("select_deterministic", &select_deterministic_cuda);
}
