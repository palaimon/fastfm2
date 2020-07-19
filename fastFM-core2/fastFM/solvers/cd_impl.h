//
// Created by ibayer on 22.10.15.
//

#ifndef FASTFM_CD_UTILS_H
#define FASTFM_CD_UTILS_H

#include "fastfm_impl.h"
#include "fastfm_internal.h"

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
                   fm_settings settings, fm_coef* coef, VectorRef res,
                   fit_callback_t cb, python_function_t python_func);

void FitSquareLoss(constSpMatRef x, constVectorRef y, constVectorRef cost,
                   fm_settings settings, fm_coef* coef,
                   fit_callback_t cb, python_function_t python_func);

void FitSquareLoss(constSpMatRef x, constVectorRef y, constVectorRef cost,
                   fm_settings settings, fm_coef* coef);

void logistic_error_weight(constVectorRef y, Vector* err, Vector* weight);

void FirstOrderStats(const int col, constVectorRef cost, constSpMatRef x,
                     constVectorRef err, double* chsqr, double* che);

void SecondOrderStats(const int layer, const int col, constVectorRef cost,
                      constSpMatRef x, constMatrixRef w2, constVectorRef err,
                      constVectorRef q_cache, double* chsqr, double* che);

void ThirdOrderStats(const int layer,
                     const int col,
                     constVectorRef cost,
                     constSpMatRef x,
                     constMatrixRef w3,
                     constVectorRef err,
                     constVectorRef q_cache,
                     constVectorRef q_sqr_cache,
                     double* chsqr,
                     double* che);

Vector Qcache(const int f, constSpMatRef x, constMatrixRef w);

Vector Qcache(const int f, constSpMatRef x, constVectorRef cost, constMatrixRef w);

Vector Q2Cache(const int f, constSpMatRef x, constMatrixRef w);

void FirstOrderPredUpdate(const int col, const double w_new, const double w_old,
                          constSpMatRef x, Vector* y_pred);

// TODO use calc error only on the fly
void FirstOrderErrUpdate(const int col, const double w_new, const double w_old,
                         constSpMatRef x, Vector* err);

void SecondOrderErrAndQcacheUpdate(const int layer, const int col, constMatrixRef w2,
                                   const double w_old, constSpMatRef x,
                                   Vector* err, Vector* q_cache);

void SecondOrderPredAndQcacheUpdate(const int layer, const int col, constMatrixRef w2,
                                    const double w_old, constSpMatRef x,
                                    Vector* err, Vector* q_cache);

void ThirdOrderErrAndQcacheUpdate(const int layer, const int col, constMatrixRef w,
                                  const double w_old, constSpMatRef x, Vector* err,
                                  Vector* q_cache, Vector* q_sqr_cache);

}
}
}


#endif //FASTFM_CD_UTILS_H
