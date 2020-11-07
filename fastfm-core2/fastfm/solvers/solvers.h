//
// Copyright [2020] <palaimon.io>
//

#ifndef FASTFM_CORE2_FASTFM_SOLVERS_SOLVERS_H_
#define FASTFM_CORE2_FASTFM_SOLVERS_SOLVERS_H_

#include <Eigen/Core>
#include "fastfm.h"
#include "fastfm_impl.h"

namespace fastfm {
namespace cd {
#define CD

void Predict(Model* m, Data* d);

void FitSquareLoss(Data* d,
                   Model* m,
                   Settings* s,
                   fit_callback_t cb,
                   python_function_t python_func);

void FitSquareLoss(Data* d, Model* m, Settings* s);

}  // namespace cd

// todo: add more solvers here for release =)
#if !EXTERNAL_RELEASE
#define SGD
#define RANKING
#define ICD
#define MCMC
#define IRLS
#endif  // EXTERNAL_RELEASE
}  // namespace fastfm

#endif  // FASTFM_CORE2_FASTFM_SOLVERS_SOLVERS_H_

