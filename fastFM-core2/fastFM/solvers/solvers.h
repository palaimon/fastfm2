//
// Created by Alex Joz on 12.06.2020.
//

#ifndef _SOLVERS_H_
#define _SOLVERS_H_


#include <Eigen/Core>
#include "fastfm.h"
#include "fastfm_impl.h"

namespace fastfm {
namespace cd {
#define CD

void Predict(Model* m, Data* d);

void FitSquareLoss(Data* d, Model* m, Settings* s, fit_callback_t cb, python_function_t python_func);

void FitSquareLoss(Data* d, Model* m, Settings* s);

}

//todo: add more solvers here for release =)
#if !EXTERNAL_RELEASE
    #define SGD
    #define RANKING
    #define ICD
    #define MCMC
    #define IRLS
#endif //EXTERNAL_RELEASE
}

#endif //_SOLVERS_H_

