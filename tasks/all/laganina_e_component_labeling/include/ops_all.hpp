#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace laganina_e_component_labeling_all {
class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  boost::mpi::communicator world_;

  int m_;
  int n_;
  int local_m_;
  int local_n_;

  std::vector<int> local_bin_;
  std::vector<int> binary_;

  void InitializeParents(std::vector<int>& parent);
  void ProcessSweep(bool reverse, std::vector<int>& parent, bool& changed) const;
  static bool UnionNodes(int a, int b, std::vector<int>& parent);
  bool CheckNeighbor(int nr, int nc, int current, std::vector<int>& parent) const;
  bool ProcessRow(int row_idx, bool reverse, std::vector<int>& parent) const;
  static int FindRoot(std::vector<int>& parent, int x);
  void FinalizeRoots(std::vector<int>& parent) const;
  void AssignLabels(std::vector<int>& parent);
  void LabelConnectedComponents();
  void RelabelImage();
  [[nodiscard]] bool IsValidCoordinate(int x, int y) const;
  void Normalize();
  void ProcessConnectedComponent(int start_x, int start_y, int label,
                                 const std::vector<std::pair<int, int>>& directions);
};

inline void NormalizeLabels(std::vector<int>& vec) {
  std::vector<int> unique_labels;
  std::unordered_set<int> seen;

  for (int val : vec) {
    if (val != 0 && seen.find(val) == seen.end()) {
      unique_labels.push_back(val);
      seen.insert(val);
    }
  }

  std::unordered_map<int, int> label_map;
  int current_label = 1;
  for (int val : unique_labels) {
    label_map[val] = current_label++;
  }
  for (int& val : vec) {
    if (val != 0) {
      val = label_map[val];
    }
  }
}
}  // namespace laganina_e_component_labeling_all
