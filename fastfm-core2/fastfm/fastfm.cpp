//
// Copyright [2020] <palaimon.io>
//

#include <memory>

#include "fastfm.h"
#include "fastfm_impl.h"
#include "solvers/solvers.h"

#define LOGURU_IMPLEMENTATION 1
#define LOGURU_REPLACE_GLOG 1
#include "../3rdparty/loguru/loguru.hpp"

#ifdef SGD
#include "sgd.h"
#include "bpr.h"
#endif
#ifdef ICD
#include "icd.h"
#endif
#ifdef RANKING
#include "ranking.h"
#endif

namespace fastfm {

Settings::Settings() : mImpl(new Settings::Impl()) {}

Settings::Settings(const std::map<std::string, std::string>& settings_map)
    : mImpl(new Settings::Impl(settings_map)) {
//    LOG(INFO) << ">>Settings Ctor from map\n";
//    for (const auto& item : settings_map) {
//        LOG(INFO) << item.first
//                  << ':'
//                  << item.second;
//    }
//    LOG(INFO) << "<<Settings Ctor from map\n";
}

Settings::~Settings() {
  delete mImpl;
}

Model::Model() : mImpl(new Model::Impl()) {
  mImpl->coef_ = new ModelParam();
}

// Destructor
Model::~Model() {
  // delete allocated pointers
  delete mImpl->coef_;
  delete mImpl;
}

void Model::add_vector(const std::string& name, double* data, size_t size) {
  if (name == "w0") {
        CHECK_EQ(size, 1);
    mImpl->coef_->setw0_ptr(data);
  } else if (name == "w1") {
        CHECK_GE(size, 0);
    mImpl->coef_->setw1(data, size);
  } else {
    mImpl->coef_->add_vector(name, data, size);
  }
}

void Model::add_matrix(const std::string& name,
                       double* data,
                       size_t rows,
                       size_t cols,
                       bool rowMajor) {
      CHECK_GE(rows, 0);
      CHECK_GE(cols, 0);

  if (name == "w2") {
    mImpl->coef_->setw2(data, rows, cols);
  } else if (name == "w3") {
    mImpl->coef_->setw3(data, rows, cols);
  } else {
        LOG(ERROR) << "Matrix " << name << " not supported.";
  }
}

void Model::add_scalar_map(const std::string& keys,
                           double* values,
                           size_t size) {
  mImpl->coef_->setMapValues(keys, values, size);
}

Data::Data() : mImpl(new Data::Impl()) {}

Data::~Data() {
  delete mImpl;
}

void Data::add_vector(const std::string& name,
                      double* data,
                      const size_t size) {
  if (name == "y_true") {
    mImpl->wrap_train_target_memory(data, size);
  } else if (name == "y_pred" || name == "y_train_pred") {
    mImpl->wrap_pred_memory(data, size);
  } else {
    mImpl->add_vector(name, data, size);
  }
}

void Data::add_matrix(const std::string& name,
                      double* data,
                      size_t rows,
                      size_t cols,
                      bool rowMajor) {
      CHECK_GE(rows, 0);
      CHECK_GE(cols, 0);
  if (name == "y_rec") {
    mImpl->wrap_pred_memory(data, rows, cols);
  } else {
    CHECK(false) << "Name: " << name << " is not supported";
  }
}

void Data::add_sparse_matrix(const std::string& name,
                             double* data,
                             size_t rows,
                             size_t cols,
                             int nnz,
                             int* outer,
                             int* inner,
                             bool col_major) {
  if (name == "x" || name == "x_c" || name == "x_i") {
    if (col_major) { mImpl
          ->wrap_design_matrix_col_major(name, data, rows, cols, nnz, outer,
                                         inner);
    } else {
      mImpl->wrap_design_matrix_row_major(name, data, rows, cols, nnz, outer,
                                          inner);
    }
  } else {
    CHECK(false) << "Name: " << name << " is not supported";
  }
}

void predict(Model* m, Data* d) {
  #ifdef RANKING
  if (Internal::get_impl(d)->is_ranking()) {
    TopNRetrieval(m, d);
    return;
  }
  #endif

  #ifdef CD
  if (Internal::get_impl(d)->has_col_major()) {
    cd::Predict(m, d);
    return;
  }
  #endif

  #ifdef SGD
  //    else  // sgd - default case if exists
  {
    sgd::Predict(m, d);
    return;
  }
  #endif

  CHECK(false) << "Solver is not supported!";
}

void fit(Settings* s,
         Model* m,
         Data* d,
         fit_callback_t cb,
         python_function_t python_func) {
  Data::Impl* data = Internal::get_impl(d);
  Model::Impl* model = Internal::get_impl(m);
  Settings::Impl* settings = Internal::get_impl(s);

  settings->settings_.rank_w2 = model->coef_->getw2().rows();
  settings->settings_.rank_w3 = model->coef_->getw3().rows();

  #ifdef CD
  if ((settings->settings_.solver == "cd"
      || settings->settings_.solver == "mcmc") &
      (settings->settings_.loss == "squared"
          || settings->settings_.loss == "logistic")) {
    data->check_col_major_train();
    cd::FitSquareLoss(d, m, s, cb, python_func);
    return;
  }
  #endif

  #ifdef SGD
  if (settings->settings_.solver == "sgd") {
    if (settings->settings_.loss == "bpr") {
      bpr::Fit(d, m, s, cb, python_func);
    } else {
      data->check_row_major_train();
      sgd::Fit(d, m, s, cb, python_func);
    }
    return;
  }
  #endif

  #ifdef ICD
  if (settings->settings_.solver == "icd") {
    data->check_icd_train();
    icd::FitSquareLoss(d, m, s);
    return;
  }
      #endif

  CHECK(false)
  << "Solver: " << settings->settings_.solver << " is not supported";
}

void fit(Settings* s, Model* m, Data* d) {
  fit(s, m, d, nullptr, nullptr);
}

}  // namespace fastfm
