#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vedernikova_k_gauss_3x3_horizontal_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void CalculateGaussMatrix();
  void CalculateNewPixelValue(int x, int y);

 private:
  std::vector<double> kernel_;
  std::vector<double> in_matr_;
  std::vector<double> out_matr_;
  int cols_;
  int rows_;
};

}  // namespace vedernikova_k_gauss_3x3_horizontal_seq