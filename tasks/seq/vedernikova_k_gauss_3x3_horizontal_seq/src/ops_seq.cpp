#include "seq/vedernikova_k_gauss_3x3_horizontal_seq/include/ops_seq.hpp"

#include <cmath>
#include <numbers>
#include <vector>

bool vedernikova_k_gauss_3x3_horizontal_seq::TestTaskSequential::PreProcessingImpl() {
  cols_ = static_cast<int>(task_data->inputs_count[0]);
  rows_ = static_cast<int>(task_data->inputs_count[1]);
  in_matr_.reserve(rows_ * cols_);
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  for (int i = 0; i < cols_ * rows_; ++i) {
    in_matr_.emplace_back(in_ptr[i]);
  }
  kernel_.reserve(9);
  CalculateGaussMatrix();
  out_matr_ = std::vector<double>(cols_ * rows_);
  return true;
}

bool vedernikova_k_gauss_3x3_horizontal_seq::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  if ((task_data->inputs_count[0] > 2 || task_data->inputs_count[0] == 0) &&
      (task_data->inputs_count[1] > 2 || task_data->inputs_count[1] == 0)) {
    return task_data->outputs_count[0] == task_data->inputs_count[0] * task_data->inputs_count[1];
  }
  return false;
}

bool vedernikova_k_gauss_3x3_horizontal_seq::TestTaskSequential::RunImpl() {
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      CalculateNewPixelValue(i, j);
    }
  }

  return true;
}

bool vedernikova_k_gauss_3x3_horizontal_seq::TestTaskSequential::PostProcessingImpl() {
  for (int i = 0; i < cols_ * rows_; i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = out_matr_[i];
  }
  return true;
}

void vedernikova_k_gauss_3x3_horizontal_seq::TestTaskSequential::CalculateGaussMatrix() {
  double sum = 0.0;
  double sigma = 1.0;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      kernel_.emplace_back(1 / (2 * std::numbers::pi * sigma * sigma) *
                           std::pow(std::numbers::e, -((x * x + y * y) / (2.0 * sigma * sigma))));
      sum += kernel_.back();
    }
  }
  for (auto &&el : kernel_) {
    el /= sum;
  }
}

void vedernikova_k_gauss_3x3_horizontal_seq::TestTaskSequential::CalculateNewPixelValue(int x, int y) {
  for (int c = 0; c < 3; c++) {
    if ((0 <= y - 1) && (y - 1 < cols_) && (0 <= x - 1 + c) && (x - 1 + c <= rows_)) {
      int index = ((y - 1) * rows_) + x - 1 + c;
      if (index >= 0 && index < rows_ * cols_) {
        out_matr_[(y * rows_) + x] += in_matr_[index] * kernel_[c];
      }
    }
  }
  for (int c = 3; c < 6; c++) {
    if ((0 <= y - 1 + c) && (y - 1 + c <= rows_)) {
      int index = (y * rows_) + x - 4 + c;
      if (index >= 0 && index < rows_ * cols_) {
        out_matr_[(y * rows_) + x] += in_matr_[index] * kernel_[c];
      }
    }
  }
  for (int c = 6; c < 9; c++) {
    if ((0 <= y - 1) && (y - 1 < cols_) && (0 <= x - 1 + c) && (x - 1 + c <= rows_)) {
      int index1 = (y * rows_) + x - 7 + c;
      if (index1 >= 0 && index1 < rows_ * cols_) {
        out_matr_[(y * rows_) + x] += in_matr_[index1] * kernel_[c];
      }
    }
  }
}