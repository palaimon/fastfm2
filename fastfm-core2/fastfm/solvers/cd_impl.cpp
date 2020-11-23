// Copyright (C) 2020 Palaimon GmbH
//
// Author: Immanuel Bayer (design & implementation)
//         Alexander Pisarenko (refactoring)
//
// Licensed under the GNU Affero General Public License, Version 3.0
// (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
//      https://www.gnu.org/licenses/agpl-3.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cd_impl.h"

#include <Eigen/Sparse>
#include <Eigen/Core>
#include <string>

#define LOGURU_REPLACE_GLOG 1
#include "../../3rdparty/loguru/loguru.hpp"

#if !EXTERNAL_RELEASE
#include "mcmc.h"
#include "fm_utils.h"
#include "pre_release.h"
#endif

namespace fastfm {
namespace cd {
namespace impl {

void Predict(constSpMatRef x,
             constMatrixRef w2,
             constVectorRef w1,
             const double w0,
             VectorRef res) {
  // Set to zero first
  res *= 0;
//    CHECK(!std::isnan(w0));
  // res = w_0
  res.setConstant(w0);

//    for (int a=0; a< w1.size(); ++a){
//        CHECK(!std::isnan(a));
//    }

  // res += X * w.T
  if (w1.size() != 0) {
        CHECK_EQ(x.cols(), w1.size());
    res += x * w1;
  }

  // res += sum_i sum_j x_i * x_j * <v_i, v_j>
  for (int k = 0; k < w2.rows(); ++k) {
    Vector xv_sum = Vector::Zero(x.rows());
    for (int l = 0; l < x.cols(); ++l) {
      for (constSpMatRef::InnerIterator it(x, l); it; ++it) {
        const double x_l = it.value();
        const int row = it.row();
        const double w_k_l = w2.coeffRef(k, l);
//                CHECK(!std::isnan(w_k_l));
        xv_sum.coeffRef(row) += w_k_l * x_l;
        res.coeffRef(row) -= .5 * w_k_l * w_k_l * x_l * x_l;
      }
    }
    res += xv_sum.cwiseProduct(xv_sum) * .5;
  }
//    for (auto h=0; h< res.size(); ++h){
//        CHECK(!std::isnan(res.coeff(h)));
//    }
}

void Predict(constSpMatRef x,
             constMatrixRef w3,
             constMatrixRef w2,
             constVectorRef w1,
             const double w0,
             VectorRef res) {
  // Get the second order Predictions.
  Predict(x, w2, w1, w0, res);

  // Return predictions if now third order parameter are given.
  if (w3.rows() == 0 || w3.cols() == 0) {
    return;
  }

      CHECK_EQ(x.cols(), w3.cols());
  for (int k = 0; k < w3.rows(); ++k) {
    Vector xv_sum = Vector::Zero(x.rows());
    Vector x2v2_sum = Vector::Zero(x.rows());
    for (int l = 0; l < x.cols(); ++l) {
      for (constSpMatRef::InnerIterator it(x, l); it; ++it) {
        const double x_l = it.value();
        const int row = it.row();
        const double w_k_l = w3.coeffRef(k, l);

        xv_sum.coeffRef(row) += w_k_l * x_l;
        x2v2_sum.coeffRef(row) += w_k_l * w_k_l * x_l * x_l;
        res.coeffRef(row) += (1. / 3) * w_k_l * w_k_l * w_k_l * x_l * x_l * x_l;
      }
    }
    res += (1. / 6) * xv_sum.cwiseProduct(xv_sum).cwiseProduct(xv_sum);
    res -= .5 * xv_sum.cwiseProduct(x2v2_sum);
  }
}

void FitSquareLoss(constSpMatRef x, constVectorRef y, constVectorRef cost,
                   SolverSettings settings, ModelParam* coef,
                   fit_callback_t cb, python_function_t python_func) {
  Vector y_pred;
  FitSquareLoss(x, y, cost, settings, coef, y_pred, cb, python_func);
}

void FitSquareLoss(constSpMatRef x, constVectorRef y, constVectorRef cost,
                   SolverSettings settings, ModelParam* coef) {
  FitSquareLoss(x, y, cost, settings, coef, nullptr, nullptr);
}

void FitSquareLoss(constSpMatRef x, constVectorRef y, constVectorRef cost,
                   SolverSettings settings, ModelParam* coef, VectorRef res,
                   fit_callback_t cb, python_function_t python_func) {
  const int n_samples = x.rows();
  const int n_features = x.cols();
  const bool second_order = settings.rank_w2 > 0;
  const bool third_order = settings.rank_w3 > 0;
  const bool irls = settings.loss == "logistic";
  const bool is_mcmc = settings.solver == "mcmc";

  #if !EXTERNAL_RELEASE
  mcmc::GibbsSampler sampler(123);
  #endif

  double step_size = 1;

  Vector weight;
  if (irls) {
    #if !EXTERNAL_RELEASE
    weight = Vector::Zero(y.size());
    step_size = settings.step_size;
    #endif
  } else if (cost.size() > 0) {
    // TODO(Immanuel): avoid copy
    weight = cost;
  }

  Vector err(y.size());
  int i = 0;
  for (; i < settings.iter; ++i) {
    // init err with predictions
    if (third_order) {
      Predict(x,
              coef->getw3(), coef->getw2(), coef->getw1(), coef->getw0(),
              err);
    } else {
      Predict(x,
              coef->getw2(), coef->getw1(), coef->getw0(),
              err);
    }

    // save prediction
    #if !EXTERNAL_RELEASE
    if (is_mcmc) {
      // train
      utils::streaming_mean(i, err, res);

      // test
      if (cb != nullptr && python_func != nullptr) {
        cb(R"({"stage": "update_prediction"})", python_func);
      }
    }
    #endif

    // err = y - y_pred
    if (irls) {
      #if !EXTERNAL_RELEASE
      // calculate error and cost based on working response
      logistic_error_weight(y, &err, &weight);
      if (cost.rows() > 0) {
        weight = weight * cost;
      }
      #endif
    } else {
      err = y + -1 * err;
    }

    #if !EXTERNAL_RELEASE
    if (is_mcmc) {
      if (cb != nullptr && python_func != nullptr) {
        cb(R"({"stage": "draw_mcmc"})", python_func);
      }
      sampler.set_alpha(coef->getMapValue("alpha"));

      sampler.set_lambdas(coef->getMapValue("lambda_w0"),
                          coef->getMapValue("lambda_w1"),
                          coef->get_vector("lambda_w2"));

      sampler.set_mus(coef->getMapValue("mu_w0"),
                      coef->getMapValue("mu_w1"),
                      coef->get_vector("mu_w2"));
    }
    #endif

    // Update Zero Order (Bias) Parameter
    if (settings.zero_order) {
      const double w_old = coef->getw0();
      const double n = static_cast<double>(n_samples);

      if (is_mcmc) {
        #if !EXTERNAL_RELEASE
        coef->setw0(sampler.draw_w0(w_old, n, err.sum()));
        #endif
      } else {
        coef->setw0((err.sum() + w_old * n) / n);
      }
      // TODO(Immanuel) assert w0 is finite

      // update error
      err = err.array() + (w_old - coef->getw0());
    }

    // Update First (Linear) Order Parameter
    for (int j = 0; settings.first_order && j < n_features; ++j) {
      double chsqr = 0;
      double che = 0;
      const double w_old = coef->getw1().coeff(j);
      // TODO(Immanuel) don't recalculate che it's constant
      FirstOrderStats(j, weight, x, err, &chsqr, &che);
      double w_new = 0;
      if (is_mcmc) {
        #if !EXTERNAL_RELEASE
        w_new = sampler.draw_w1(w_old, chsqr, che);
        #endif
      } else {
        w_new = (che + w_old * chsqr) / (chsqr + settings.l2_reg_w1);
      }
      coef->getw1().coeffRef(j) = w_old + step_size * (w_new - w_old);
      FirstOrderErrUpdate(j, coef->getw1().coeff(j), w_old, x, &err);
    }

    // Update Second Order Parameter
    for (int f = 0; second_order && f < coef->getw2().rows(); ++f) {
      Vector q_cache = Qcache(f, x, coef->getw2());
      for (int j = 0; j < n_features; ++j) {
        double chsqr = 0;
        double che = 0;
        const double w_old = coef->getw2().coeff(f, j);
        SecondOrderStats(f, j, weight,
                         x, coef->getw2(), err,
                         q_cache, &chsqr, &che);
        double w_new = 0;
        if (is_mcmc) {
          #if !EXTERNAL_RELEASE
          w_new = sampler.draw_w2(f, w_old, chsqr, che);
          #endif
        } else {
          w_new = (che + w_old * chsqr) / (chsqr + settings.l2_reg_w2);
        }
        coef->getw2().coeffRef(f, j) = w_old + step_size * (w_new - w_old);
        SecondOrderErrAndQcacheUpdate(f, j, coef->getw2(), w_old,
                                      x, &err, &q_cache);
      }
    }

    #if !EXTERNAL_RELEASE
    // Update Third Order Parameter
    for (int f = 0; third_order && f < coef->getw3().rows(); ++f) {
      Vector q_cache = Qcache(f, x, coef->getw3());
      Vector q2_cache = Q2Cache(f, x, coef->getw3());
      for (int j = 0; j < n_features; ++j) {
        if (is_mcmc) CHECK(false) << "3'rd order not supported by mcmc";
        double chsqr = 0;
        double che = 0;
        const double w_old = coef->getw3().coeff(f, j);
        ThirdOrderStats(f, j, weight,
                        x, coef->getw3(), err,
                        q_cache, q2_cache, &chsqr, &che);
        const double
            w_new = (che + w_old * chsqr) / (chsqr + settings.l2_reg_w3);
        coef->getw3().coeffRef(f, j) = w_old + step_size * (w_new - w_old);
        ThirdOrderErrAndQcacheUpdate(f, j, coef->getw3(), w_old,
                                     x, &err, &q_cache, &q2_cache);
      }
    }
    #endif

    if (cb != nullptr && python_func != nullptr) {
      bool early_stop = false;
      if (is_mcmc) {
        #if !EXTERNAL_RELEASE
        std::stringstream ss;
        ss << "{";
        ss << "\"stage\":" << "\"early_stop\"" << ", ";
        ss << "\"params\":" << sampler.parameter_to_json();
        ss << "}";
        early_stop = cb(ss.str(), python_func);
        #endif
      } else {
        early_stop = cb("{}", python_func);
      }

      if (early_stop) {
        break;
      }
    }
  }
  #if !EXTERNAL_RELEASE
  if (is_mcmc) {
    if (cb != nullptr && python_func != nullptr) {
      cb(R"({"stage": "update_prediction"})", python_func);
    }

    if (third_order) {
      Predict(x,
              coef->getw3(), coef->getw2(), coef->getw1(), coef->getw0(),
              err);
    } else {
      Predict(x,
              coef->getw2(), coef->getw1(), coef->getw0(),
              err);
    }
    utils::streaming_mean(i, err, res);
  }
  #endif
}

void FirstOrderStats(const int col, constVectorRef cost, constSpMatRef x,
                     constVectorRef err, double* chsqr, double* che) {
  const bool no_cost = cost.size() == 0;
  *chsqr = *che = 0;
  for (constSpMatRef::InnerIterator it(x, col); it; ++it) {
    const int row = it.row();
    const double x_col_i = it.value();
    const double cost_i = no_cost ? 1 : cost.coeffRef(row);
    *chsqr += cost_i * x_col_i * x_col_i;
    *che += cost_i * x_col_i * err.coeffRef(row);
  }
}

void SecondOrderStats(const int layer, const int col, constVectorRef cost,
                      constSpMatRef x, constMatrixRef w2, constVectorRef err,
                      constVectorRef q_cache, double* chsqr, double* che) {
  const bool no_cost = cost.size() == 0;
  *chsqr = *che = 0;
  for (constSpMatRef::InnerIterator it(x, col); it; ++it) {
    const int row = it.row();
    const double x_col_i = it.value();
    const double cost_i = no_cost ? 1 : cost.coeffRef(row);
    const double q_i = q_cache.coeffRef(row);
    const double h_i = x_col_i * (q_i - w2.coeffRef(layer, col) * x_col_i);

    *chsqr += cost_i * h_i * h_i;
    *che += cost_i * h_i * err.coeffRef(row);
  }
}

Vector Qcache(const int f,
              constSpMatRef x,
              constVectorRef cost,
              constMatrixRef w) {
  const bool no_cost = cost.size() == 0;

  if (!no_cost) {
        CHECK_EQ(x.rows(), cost.size());
  }

  Vector q_cache = Vector::Zero(x.rows());
  for (int k = 0; k < x.cols(); ++k) {
    for (constSpMatRef::InnerIterator it(x, k); it; ++it) {
      const double x_kl = it.value();
      const int row = it.row();   // row index
      const double cost_ = no_cost ? 1.0 : cost.coeffRef(row);
      q_cache.coeffRef(row) += x_kl * w.coeffRef(f, k) * cost_;
    }
  }
  return q_cache;
}

Vector Qcache(const int f, constSpMatRef x, constMatrixRef w) {
  Vector q_cache = Vector::Zero(x.rows());
  for (int k = 0; k < x.cols(); ++k) {
    for (constSpMatRef::InnerIterator it(x, k); it; ++it) {
      const double x_kl = it.value();
      const int row = it.row();   // row index
      q_cache.coeffRef(row) += x_kl * w.coeffRef(f, k);
    }
  }
  return q_cache;
}

void FirstOrderErrUpdate(const int col, const double w_new, const double w_old,
                         constSpMatRef x, Vector* err) {
  for (constSpMatRef::InnerIterator it(x, col); it; ++it) {
    const int row = it.row();
    const double x_col_i = it.value();

    err->coeffRef(row) += (w_old - w_new) * x_col_i;
  }
}

void FirstOrderPredUpdate(const int col, const double w_new, const double w_old,
                          constSpMatRef x, Vector* y_pred) {
  for (constSpMatRef::InnerIterator it(x, col); it; ++it) {
    const int row = it.row();
    const double x_col_i = it.value();

    y_pred->coeffRef(row) += (w_new - w_old) * x_col_i;
  }
}

void SecondOrderErrAndQcacheUpdate(const int layer,
                                   const int col,
                                   constMatrixRef w2,
                                   const double w_old,
                                   constSpMatRef x,
                                   Vector* err,
                                   Vector* q_cache) {
  double const w_new = w2.coeffRef(layer, col);
  for (constSpMatRef::InnerIterator it(x, col); it; ++it) {
    const int row = it.row();
    const double x_col_i = it.value();

    const double q_i = q_cache->coeffRef(row);
    const double h_i = x_col_i * (q_i - w_old * x_col_i);

    q_cache->coeffRef(row) += (w_new - w_old) * x_col_i;
    err->coeffRef(row) += (w_old - w_new) * h_i;
  }
}

void SecondOrderPredAndQcacheUpdate(const int layer,
                                    const int col,
                                    constMatrixRef w2,
                                    const double w_old,
                                    constSpMatRef x,
                                    Vector* y_pred,
                                    Vector* q_cache) {
  double const w_new = w2.coeffRef(layer, col);
  for (constSpMatRef::InnerIterator it(x, col); it; ++it) {
    const int row = it.row();
    const double x_col_i = it.value();

    const double q_i = q_cache->coeffRef(row);
    const double h_i = x_col_i * (q_i - w_old * x_col_i);

    q_cache->coeffRef(row) += (w_new - w_old) * x_col_i;
    y_pred->coeffRef(row) += (w_new - w_old) * h_i;
  }
}

}  // namespace impl
}  // namespace cd
}  // namespace fastfm
