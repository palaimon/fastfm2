//
// Copyright [2020] <palaimon.io>
//

#ifndef FASTFM_CORE2_FASTFM_DATASETS_H_
#define FASTFM_CORE2_FASTFM_DATASETS_H_

#include <unordered_map>
#include <vector>

#include "fastfm_impl.h"

#define LOGURU_REPLACE_GLOG 1
#include "../3rdparty/loguru/loguru.hpp"

namespace fastfm {
namespace utils {

class DataGenerator {
 private:
  void create_design_matrix();
  void create_model_parameter();
  void create_target();

  std::vector<int> group_sizes_;
  std::vector<int> rank_;
  std::vector<double> stddev_;
  int n_samples_;
  int n_features_;
  int rng_seed_;

  SpMat x_;
  Vector y_;

  Matrix w3_;
  Matrix w2_;
  Vector w1_;
  double w0_{};

 public:
  DataGenerator(int n_samples = 100, std::vector<int> group_size = {2, 5},
                std::vector<int> rank = {1, 1, 3},
                std::vector<double> stddev = {2, 3, 2}, int rng_seed = 123);
  Vector y_class(double noise_stddev = 0.0,
                 double flip_proba = 0.1,
                 double class_ratio = 0.5);
  Vector y_reg(double noise_stddev = 1.0);
  RowSpMat x_csr() { return RowSpMat(x_); }
  SpMat x_csc() { return x_; }

  Matrix w3() { return w3_; }
  Matrix w2() { return w2_; }
  Vector w1() { return w1_; }
  double* w0() { return &w0_; }
};

class RecDataGenerator {
 private:
  void create_design_matrix();
  void create_model_parameter();
  void create_target();

  int rank_;
  std::vector<double> stddev_;

  int n_context_;
  int n_context_features_;
  int n_item_;
  int n_item_features_;
  int n_active_features_;
  int n_features_;
  int rng_seed_;

  RowSpMat x_c_;
  RowSpMat x_i_;
  RowSpMat r_;

  Matrix w2_;
  Vector w1_;

 public:
  RecDataGenerator(int n_context = 50,
                   int n_item = 100,
                   int rank = 2,
                   int n_context_features = 100,
                   int n_item_features = 200,
                   int n_active_features = 5,
                   std::vector<double> stddev = {2, 3},
                   int rng_seed = 123);
  RowSpMat R() { return RowSpMat(r_); }
  RowSpMat x_c() { return x_c_; }
  RowSpMat x_i() { return x_i_; }

  Matrix w2() { return w2_; }
  Vector w1() { return w1_; }
};

}  // namespace utils
}  // namespace fastfm

#endif  // FASTFM_CORE2_FASTFM_DATASETS_H_
