#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/vedernikova_k_gauss_3x3_horizontal_seq/include/ops_seq.hpp"

TEST(vedernikova_k_gauss_3x3_horizontal_seq, validation_false) {
  // Create data
  std::vector<double> in{20.0, 7.0};
  int cols = 1;
  int rows = 1;

  std::vector<double> out(9);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  vedernikova_k_gauss_3x3_horizontal_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(vedernikova_k_gauss_3x3_horizontal_seq, image_3x3) {
  // Create data
  std::vector<double> in{2.0, 11.0, 75.0, 245.0, 134.0, 100.0, 200.0, 128.0, 198.0};

  std::vector<double> expected{42.0, 54.0, 39.0, 102.0, 122.0, 81.0, 97.0, 117.0, 78.0};

  int cols = 3;
  int rows = 3;

  std::vector<double> out(9);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  vedernikova_k_gauss_3x3_horizontal_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected, out);
}

TEST(vedernikova_k_gauss_3x3_horizontal_seq, image_4x4) {
  // Create data
  std::vector<double> in{95.0,  61.0, 100.0, 88.0,  200.0, 253.0, 200.0, 128.0,
                         222.0, 81.0, 255.0, 100.0, 222.0, 50.0,  190.0, 255.0};
  std::vector<double> expected{70.0,  97.0,  92.0,  61.0,  126.0, 171.0, 156.0, 100.0,
                               137.0, 178.0, 174.0, 128.0, 89.0,  109.0, 121.0, 107.0};
  int cols = 4;
  int rows = 4;

  std::vector<double> out(16);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  vedernikova_k_gauss_3x3_horizontal_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected, out);
}
