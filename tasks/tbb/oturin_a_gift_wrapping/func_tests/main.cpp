#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/oturin_a_gift_wrapping/include/ops_tbb.hpp"

namespace {
using namespace oturin_a_gift_wrapping_tbb;

Coord RandCoord(int r) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<int> dist(-r, r);
  return {.x = dist(rng), .y = dist(rng)};
}

void DoCommonTest(std::vector<Coord> &in, std::vector<Coord> &answer, std::vector<Coord> &out) {
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  for (std::size_t i = 0; i < answer.size(); i++) {
    EXPECT_EQ(answer[i].x, out[i].x) << out[i].x << '_' << out[i].y << ' ' << answer[i].x << '_' << answer[i].y;
    EXPECT_EQ(answer[i].y, out[i].y) << out[i].x << '_' << out[i].y << ' ' << answer[i].x << '_' << answer[i].y;
  }
}

const double kPi = std::numbers::pi;

}  // namespace

TEST(oturin_a_gift_wrapping_tbb, test_empty) {
  std::vector<oturin_a_gift_wrapping_tbb::Coord> in = {};
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(0);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  oturin_a_gift_wrapping_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_FALSE(test_task_tbb.Validation());
}

// NOLINTBEGIN(modernize-use-designated-initializers)
TEST(oturin_a_gift_wrapping_tbb, test_too_small) {
  std::vector<oturin_a_gift_wrapping_tbb::Coord> in = {{-4, 4}, {-2, 4}};
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(0);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  oturin_a_gift_wrapping_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_FALSE(test_task_tbb.Validation());
}

TEST(oturin_a_gift_wrapping_tbb, test_same_points) {
  std::vector<oturin_a_gift_wrapping_tbb::Coord> in = {{0, 0}, {0, 0}, {0, 0}};
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(0);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  oturin_a_gift_wrapping_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_FALSE(test_task_tbb.PreProcessing());
}

TEST(oturin_a_gift_wrapping_tbb, test_on_line) {
  std::vector<oturin_a_gift_wrapping_tbb::Coord> in = {{0, 4}, {0, 3}, {0, 2}};
  std::vector<oturin_a_gift_wrapping_tbb::Coord> answer = {{0, 4}, {0, 3}, {0, 2}};
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(answer.size());

  DoCommonTest(in, answer, out);
}

TEST(oturin_a_gift_wrapping_tbb, test_triangle) {
  std::vector<oturin_a_gift_wrapping_tbb::Coord> in = {{0, 10}, {10, -10}, {-10, -10}};
  std::vector<oturin_a_gift_wrapping_tbb::Coord> answer = {{-10, -10}, {0, 10}, {10, -10}};
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(answer.size());

  DoCommonTest(in, answer, out);
}

TEST(oturin_a_gift_wrapping_tbb, test_square) {
  std::vector<oturin_a_gift_wrapping_tbb::Coord> in;
  std::vector<oturin_a_gift_wrapping_tbb::Coord> answer = {{-1, 1}, {1, 1}, {1, -1}, {-1, -1}};
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(answer.size());
  in = answer;

  DoCommonTest(in, answer, out);
}

TEST(oturin_a_gift_wrapping_tbb, test_square_morePoints) {
  std::vector<oturin_a_gift_wrapping_tbb::Coord> in;
  std::vector<oturin_a_gift_wrapping_tbb::Coord> answer = {{-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {-1, -1}, {-1, 0}};
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(answer.size());
  in = answer;

  DoCommonTest(in, answer, out);
}

TEST(oturin_a_gift_wrapping_tbb, test_circle) {
  int points = 30;
  int circle_radius = 100;

  std::vector<oturin_a_gift_wrapping_tbb::Coord> in;
  std::vector<oturin_a_gift_wrapping_tbb::Coord> answer(points);
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(answer.size());
  for (int i = 0; i < points; i++) {
    answer[i] = {int(-std::cos(2 * i * kPi / points) * circle_radius),
                 int(std::sin(2 * i * kPi / points) * circle_radius)};
  }
  in = answer;

  DoCommonTest(in, answer, out);
}

TEST(oturin_a_gift_wrapping_tbb, test_circle_shuffled) {
  int points = 30;
  int circle_radius = 100;

  std::vector<oturin_a_gift_wrapping_tbb::Coord> in;
  std::vector<oturin_a_gift_wrapping_tbb::Coord> answer(points);
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(answer.size());
  for (int i = 0; i < points; i++) {
    answer[i] = {int(-std::cos(2 * i * kPi / points) * circle_radius),
                 int(std::sin(2 * i * kPi / points) * circle_radius)};
  }
  in = answer;

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(in.begin(), in.end(), g);

  DoCommonTest(in, answer, out);
}

TEST(oturin_a_gift_wrapping_tbb, test_star) {
  int points = 10;
  int star_radius = 100;

  std::vector<oturin_a_gift_wrapping_tbb::Coord> in(points);
  std::vector<oturin_a_gift_wrapping_tbb::Coord> answer;
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(points);
  for (int i = 0; i < points; i++) {
    in[i] = {int(-std::cos((2 * i * kPi / points) + (kPi / points)) * star_radius / (i % 2 + 1)),
             int(std::sin((2 * i * kPi / points) + (kPi / points)) * star_radius / (i % 2 + 1))};
    if ((points % 2) != 0) {
      answer.push_back(in[i]);
    }
  }

  DoCommonTest(in, answer, out);
}

TEST(oturin_a_gift_wrapping_tbb, test_star_bigAndShuffled) {
  int points = 30;
  int star_radius = 100;

  std::vector<oturin_a_gift_wrapping_tbb::Coord> in(points);
  std::vector<oturin_a_gift_wrapping_tbb::Coord> answer;
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(points);
  for (int i = 0; i < points; i++) {
    in[i] = {int(-std::cos((2 * i * kPi / points) + (kPi / points)) * star_radius / (i % 2 + 1)),
             int(std::sin((2 * i * kPi / points) + (kPi / points)) * star_radius / (i % 2 + 1))};
    if ((points % 2) != 0) {
      answer.push_back(in[i]);
    }
  }

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(in.begin(), in.end(), g);

  DoCommonTest(in, answer, out);
}

TEST(oturin_a_gift_wrapping_tbb, test_predefined1) {
  std::vector<oturin_a_gift_wrapping_tbb::Coord> in = {{-4, 4}, {-2, 4},  {1, 2},  {-3, 2}, {-1, 0},
                                                       {-4, 1}, {-2, -2}, {1, -3}, {0, -1}, {2, 0}};
  std::vector<oturin_a_gift_wrapping_tbb::Coord> answer = {{-4, 4}, {-2, 4},  {1, 2}, {2, 0},
                                                           {1, -3}, {-2, -2}, {-4, 1}};
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(answer.size());

  DoCommonTest(in, answer, out);
}

TEST(oturin_a_gift_wrapping_tbb, test_random_10plus4) {
  // Create data
  auto gen = [&]() { return RandCoord(5); };
  std::vector<oturin_a_gift_wrapping_tbb::Coord> in(10);
  std::ranges::generate(in.begin(), in.end(), gen);
  std::vector<oturin_a_gift_wrapping_tbb::Coord> answer = {{-6, 6}, {6, 6}, {6, -6}, {-6, -6}};
  in.insert(in.end(), answer.begin(), answer.end());
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(answer.size());
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(in.begin(), in.end(), g);

  DoCommonTest(in, answer, out);
}

TEST(oturin_a_gift_wrapping_tbb, test_random_30plus11) {
  // Create data
  auto gen = [&]() { return RandCoord(10); };
  std::vector<oturin_a_gift_wrapping_tbb::Coord> in(30);
  std::ranges::generate(in.begin(), in.end(), gen);
  std::vector<oturin_a_gift_wrapping_tbb::Coord> answer = {
      {-20, 10}, {-10, 14}, {8, 13}, {16, 6}, {16, 5}, {16, 0}, {16, -14}, {0, -19}, {-12, -22}, {-15, -15}, {-19, 0}};
  in.insert(in.end(), answer.begin(), answer.end());
  std::vector<oturin_a_gift_wrapping_tbb::Coord> out(answer.size());
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(in.begin(), in.end(), g);

  DoCommonTest(in, answer, out);
}
// NOLINTEND(modernize-use-designated-initializers)