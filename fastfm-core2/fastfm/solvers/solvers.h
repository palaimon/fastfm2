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

