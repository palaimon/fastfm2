// Copyright (C) 2020 Palaimon GmbH
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
