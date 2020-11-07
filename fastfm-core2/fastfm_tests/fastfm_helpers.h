//
// Copyright [2020] <palaimon.io>
//

#ifndef FASTFM_CORE2_FASTFM_TESTS_FASTFM_HELPERS_H_
#define FASTFM_CORE2_FASTFM_TESTS_FASTFM_HELPERS_H_

#include "fastfm.h"
#include "fastfm_decl.h"
#include "fastfm_impl.h"

namespace fastfm {

class DataFactory {
 public:
  DataFactory() {
    pd = new Data();
  }

  explicit DataFactory(SpMat& x,  // NOLINT(runtime/references)
                       Vector* prediction = nullptr,
                       Vector* target = nullptr) {
    pd = new Data();
    if (nullptr != prediction) {
      pd->add_vector("y_pred", prediction->data(), prediction->size());
    }
    if (nullptr != target) {
      pd->add_vector("y_true", target->data(), target->size());
    }
    pd->add_sparse_matrix("x", x.valuePtr(), x.rows(), x.cols(), x.nonZeros(),
                          x.outerIndexPtr(), x.innerIndexPtr(), true);
  }

  explicit DataFactory(RowSpMat& x,  // NOLINT(runtime/references)
                       Vector* prediction = nullptr,
                       Vector* target = nullptr) {
    pd = new Data();
    if (nullptr != prediction) {
      pd->add_vector("y_pred", prediction->data(), prediction->size());
    }
    if (nullptr != target) {
      pd->add_vector("y_true", target->data(), target->size());
    }
    pd->add_sparse_matrix("x", x.valuePtr(), x.rows(), x.cols(), x.nonZeros(),
                          x.outerIndexPtr(),
                          x.innerIndexPtr(), false);
  }

  template<class MatrixType>
  DataFactory(MatrixType& x,  // NOLINT(runtime/references)
              Vector* prediction, Vector* target,
              SpMat& x_c, SpMat& x_i)  // NOLINT(runtime/references)
              :  DataFactory(x, prediction, target) {
    pd->add_sparse_matrix("x_c", x_c.valuePtr(), x_c.rows(), x_c.cols(),
                          x_c.nonZeros(), x_c.outerIndexPtr(),
                          x_c.innerIndexPtr(), true);
    pd->add_sparse_matrix("x_i", x_i.valuePtr(), x_i.rows(), x_i.cols(),
                          x_i.nonZeros(), x_i.outerIndexPtr(),
                          x_i.innerIndexPtr(), true);
  }

  Data* get() {
    assert(pd != nullptr);
    return pd;
  }

 private:
  Data* pd{};
};

class ModelFactory {
 public:
  ModelFactory() {
    pm = new Model();
  }

  explicit ModelFactory(fastfm::ModelParam* pCoef) {
    pm = new Model();
    Model::Impl* modelImpl = Internal::get_impl(pm);
    modelImpl->coef_ = pCoef;
  }

  ModelFactory(double* w0, VectorRef w1) {
    pm = new Model();
    pm->add_vector("w0", w0, 1);
    pm->add_vector("w1", w1.data(), w1.size());
  }

  ModelFactory(double* w0, VectorRef w1, MatrixRef w2) {
    pm = new Model();

    pm->add_vector("w0", w0, 1);
    pm->add_vector("w1", w1.data(), w1.size());
    pm->add_matrix("w2", w2.data(), w2.rows(), w2.cols(), true);
  }

  ModelFactory(double* w0, VectorRef w1, MatrixRef w2, MatrixRef w3) {
    pm = new Model();
    pm->add_vector("w0", w0, 1);
    pm->add_vector("w1", w1.data(), w1.size());
    pm->add_matrix("w2", w2.data(), w2.rows(), w2.cols(), true);
    pm->add_matrix("w3", w3.data(), w3.rows(), w3.cols(), true);
  }

  Model* get() {
    return pm;
  }

 private:
  Model* pm;
};

}  // namespace fastfm
#endif  // FASTFM_CORE2_FASTFM_TESTS_FASTFM_HELPERS_H_
