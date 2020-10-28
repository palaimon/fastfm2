
#define LOGURU_REPLACE_GLOG 1
#include "loguru.hpp"

#include "solvers.h"
#include "cd_impl.h"


namespace fastfm {
namespace cd {

void Predict(Model* m, Data* d)
{
    Data::Impl* data = Internal::get_impl(d);
    Model::Impl* model = Internal::get_impl(m);

    const bool third_order = model->coef_->getw3().rows() > 0;

    if (third_order)
       impl::Predict(data->get_design_matrix_col_major(),
                     model->coef_->getw3(),
                     model->coef_->getw2(),
                     model->coef_->getw1(),
                     model->coef_->getw0(),
                     data->get_prediction());
    else
        impl::Predict(data->get_design_matrix_col_major(),
                      model->coef_->getw2(),
                      model->coef_->getw1(),
                      model->coef_->getw0(),
                      data->get_prediction());

}


void FitSquareLoss(Data* d, Model* m, Settings* s, fit_callback_t cb, python_function_t python_func) {
    Data::Impl* data = Internal::get_impl(d);
    Model::Impl* model = Internal::get_impl(m);
    Settings::Impl* settings = Internal::get_impl(s);

    const int n_samples = data->get_design_matrix_col_major().rows();
    const int n_features = data->get_design_matrix_col_major().cols();

    // Check that model parameter are consistent with data dimensions.
    CHECK_EQ(model->coef_->getw1().size(), n_features);
    if (model->coef_->getw2().size() > 0)
        CHECK_EQ(model->coef_->getw2().cols(), n_features);
    if (model->coef_->getw3().size() > 0)
        CHECK_EQ(model->coef_->getw3().cols(), n_features);


    // Check that data dimensions agree.
    CHECK_EQ(data->get_train_target().size(), n_samples);

    if (settings->settings_.solver == "mcmc"){
        CHECK_EQ(data->get_prediction().size(), n_samples);
        impl::FitSquareLoss(data->get_design_matrix_col_major(),
                            data->get_train_target(),
                            data->get_vector("cost"),
                            settings->settings_,
                            model->coef_,
                            data->get_prediction(),
                            cb, python_func);
    } else {
        impl::FitSquareLoss(data->get_design_matrix_col_major(),
                            data->get_train_target(),
                            data->get_vector("cost"),
                            settings->settings_,
                            model->coef_,
                            cb, python_func);
    }
}

void FitSquareLoss(Data* d, Model* m, Settings* s) {
    FitSquareLoss(d, m, s, nullptr, nullptr);
}

}
}
