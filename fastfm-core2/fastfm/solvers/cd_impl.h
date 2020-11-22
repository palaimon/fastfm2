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

#ifndef FASTFM_CORE2_FASTFM_SOLVERS_CD_IMPL_H_
#define FASTFM_CORE2_FASTFM_SOLVERS_CD_IMPL_H_

#include "fastfm_impl.h"

namespace fastfm {
namespace cd {
namespace impl {

void Predict(constSpMatRef x,
             constMatrixRef w2,
             constVectorRef w1,
             const double w0,
             VectorRef res);

void Predict(constSpMatRef x,
             constMatrixRef w3,
             constMatrixRef w2,
             constVectorRef w1,
             const double w0,
             VectorRef res);

void FitSquareLoss(constSpMatRef x, constVectorRef y, constVectorRef cost,
                   SolverSettings settings, ModelParam* coef, VectorRef res,
                   fit_callback_t cb, python_function_t python_func);

void FitSquareLoss(constSpMatRef x, constVectorRef y, constVectorRef cost,
                   SolverSettings settings, ModelParam* coef,
                   fit_callback_t cb, python_function_t python_func);

void FitSquareLoss(constSpMatRef x, constVectorRef y, constVectorRef cost,
                   SolverSettings settings, ModelParam* coef);

void FirstOrderStats(const int col, constVectorRef cost, constSpMatRef x,
                     constVectorRef err, double* chsqr, double* che);

void SecondOrderStats(const int layer, const int col, constVectorRef cost,
                      constSpMatRef x, constMatrixRef w2, constVectorRef err,
                      constVectorRef q_cache, double* chsqr, double* che);

Vector Qcache(const int f, constSpMatRef x, constMatrixRef w);

Vector Qcache(const int f,
              constSpMatRef x,
              constVectorRef cost,
              constMatrixRef w);

void FirstOrderPredUpdate(const int col, const double w_new, const double w_old,
                          constSpMatRef x, Vector* y_pred);

// TODO(Immanuel): use calc error only on the fly
void FirstOrderErrUpdate(const int col, const double w_new, const double w_old,
                         constSpMatRef x, Vector* err);

void SecondOrderErrAndQcacheUpdate(const int layer,
                                   const int col,
                                   constMatrixRef w2,
                                   const double w_old,
                                   constSpMatRef x,
                                   Vector* err,
                                   Vector* q_cache);

void SecondOrderPredAndQcacheUpdate(const int layer,
                                    const int col,
                                    constMatrixRef w2,
                                    const double w_old,
                                    constSpMatRef x,
                                    Vector* err,
                                    Vector* q_cache);

}  // namespace impl
}  // namespace cd
}  // namespace fastfm

#endif  // FASTFM_CORE2_FASTFM_SOLVERS_CD_IMPL_H_
