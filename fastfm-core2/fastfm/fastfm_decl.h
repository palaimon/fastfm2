//
// Copyright [2020] <palaimon.io>
//

#ifndef FASTFM_CORE2_FASTFM_FASTFM_DECL_H_
#define FASTFM_CORE2_FASTFM_FASTFM_DECL_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SpMatRef = Eigen::Ref<SpMat>;
using constSpMatRef = const Eigen::Ref<const SpMat>;
using RowSpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using RowSpMatRef = Eigen::Ref<RowSpMat>;
using constRowSpMatRef = const Eigen::Ref<const RowSpMat>;
using Vector = Eigen::VectorXd;
using VectorRef = Eigen::Ref<Vector, 0, Eigen::OuterStride<>>;
using constVectorRef = const Eigen::Ref<const Vector, 0, Eigen::OuterStride<>>;
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixRef = Eigen::Ref<Matrix, 0, Eigen::OuterStride<>>;
using constMatrixRef = const Eigen::Ref<const Matrix, 0, Eigen::OuterStride<>>;
using outerStride = Eigen::OuterStride<>;

#endif  // FASTFM_CORE2_FASTFM_FASTFM_DECL_H_
