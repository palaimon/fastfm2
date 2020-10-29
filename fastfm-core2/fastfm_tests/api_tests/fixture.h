#ifndef FASTFM_CATCH_FIXTURE_H
#define FASTFM_CATCH_FIXTURE_H


#include "fastfm.h"
#include "fastfm_impl.h"
#include "../fastfm_helpers.h"
#if !EXTERNAL_RELEASE
#include "solvers/fm_utils.h"
#endif


using namespace fastfm;

class FMExample{
public:
    FMExample() {

        // X = | 1 2 0 |
        //     | 0 3 0 |
        //     | 4 0 2 |
        //     | 4 5 0 |
        {
            Matrix tmp(4, 3);
            tmp <<  1, 2, 0,
                    0, 3, 0,
                    4, 0, 2,
                    4, 5, 0;
            x_ = tmp.sparseView();
            x_r_ = tmp.sparseView();
        }

        // w3 = |1 2 3|
        //      |4 5 6|
        w3_ = Matrix(2, 3);
        w3_ << 1, 2, 3,
                4, 5, 6;

        // w2 = |6 0 2|
        //      |5 1 0|
        w2_ = Matrix(2, 3);
        w2_ << 6, 0, 2,
                5, 1, 0;

        // w = [9 8 7]
        w1_ = Vector(3);
        w1_ << 9, 8, 7;

        w0_ = 2;

        coef_.setw0(w0_);
        coef_.setw1(w1_);
        coef_.setw2(w2_);
        coef_.setw3(w3_);

        settings_.iter = 50;

        settings_.zero_order = true;
        settings_.first_order = true;
        settings_.rank_w2 = 2;
        settings_.rank_w3 = 2;

        settings_.l2_reg_w1 = 0;
        settings_.l2_reg_w2 = 0;
        settings_.l2_reg_w3 = 0;

        pd_ = DataFactory().get();
        Data::Impl* data = Internal::get_impl(pd_);
        pm_ = ModelFactory().get();
        Model::Impl* model = Internal::get_impl(pm_);
        ps_ = new Settings();
        Settings::Impl* settings = Internal::get_impl(ps_);
        // Allocate space for the predictions.
        Vector y_pred = Vector::Zero(x_.rows());
        pd_ = DataFactory(x_, &y_pred).get();
        settings->settings_ = settings_;
        model->coef_ = &coef_;
    }
protected:
    SpMat x_;
    RowSpMat x_r_;
    Matrix w3_;
    Matrix w2_;
    Vector w1_;
    double w0_;
    ModelParam coef_;
    SolverSettings settings_;
    Data* pd_;
    Model* pm_;
    Settings* ps_;
    Vector y_naive;
};
#if !EXTERNAL_RELEASE
class IFMExample {
protected:
    IFMExample() {

        // X_c = | 1 0 0 |
        //       | 2 0 4 |
        {
            Matrix tmp(2, 3);
            tmp <<  1, 0, 0,
                    2, 0, 4;
            x_c_ = tmp.sparseView();
        }

        // X_i = | 1 2 |
        //       | 3 0 |
        //       | 0 6 |
        {
            Matrix tmp(3, 2);
            tmp <<  1, 2,
                    3, 0,
                    0, 6;
            x_i_ = tmp.sparseView();
        }

        // TODO Fix the all zero feature
        // X_naive = | 1 0 0 | 1 2 |
        //           | 1 0 0 | 3 0 |
        //           | 1 0 0 | 0 6 |
        //           | 2 0 4 | 1 2 |
        //           | 2 0 4 | 3 0 |
        //           | 2 0 4 | 0 6 |
        {
            Matrix tmp(6, 5);
            tmp <<  1, 0, 0, 1, 2,
                    1, 0, 0, 3, 0,
                    1, 0, 0, 0, 6,
                    2, 0, 4, 1, 2,
                    2, 0, 4, 3, 0,
                    2, 0, 4, 0, 6,
            x_naive_ = tmp.sparseView();
        }

        // X_pos = | 1 0 0 | 1 2 |
        //         | 1 0 0 | 0 6 |
        //         | 2 0 4 | 0 6 |
        {
            Matrix tmp(3, 5);
            tmp <<  1, 0, 0, 1, 2,
                    1, 0, 0, 0, 6,
                    2, 0, 4, 0, 6,
                    x_pos_ = tmp.sparseView();
        }

        // y_naive_ | 1 0 2 0 0 3 |
        y_naive_ = Vector(6);
        y_naive_ << 1, 0, 2, 0, 0, 3;

        // y_pos_ | 1 2 3 |
        y_pos_ = Vector(3);
        y_pos_ << 1, 2, 3;

        // Rescal target
        y_ifm_ = Vector(3);
        y_ifm_ << 1 * 2, 2 * (3. / 2), 3 * (4. / 3);

        // cost_naive_ | 2, 1, 3, 1, 1, 4 |
        cost_naive_ = Vector(6);
        cost_naive_ << 2, 1, 3, 1, 1, 4;

        // cost_pos_ | 2, 3, 4 |
        cost_pos_ = Vector(3);
        cost_pos_ << 2, 3, 4;


        // Rescal cost
        cost_ifm_ = Vector(3);
        cost_ifm_ = cost_pos_ - Vector::Ones(3);

        // w2_naive = |4 5 6 0 9 |
        //            |0 3 0 7 8 |
        w2_naive_ = Matrix(2, 5);
        w2_naive_ << 4, 5, 6, 0, 9,
                     0, 3, 0, 7, 8;

        // w2_c = |4 5 6|
        //        |0 3 0|
        w2_c_ = w2_naive_.leftCols(3);

        // w2_i = |0 9 |
        //        |7 8 |
        w2_i_ = w2_naive_.rightCols(2);

        // w1_naive = [11 22 33 44 55]
        w1_naive_ = Vector(5);
        w1_naive_ << 11, 22, 33, 44, 55;

        // w1_c = [11 22 33]
        w1_c_ = w1_naive_.topRows(3);

        // w1_i = [44 55]
        w1_i_ = w1_naive_.bottomRows(2);

        settings_.iter = 50;

        settings_.zero_order = true;
        settings_.first_order = true;
        settings_.rank_w2 = 2;

        settings_.l2_reg_w1 = 0;
        settings_.l2_reg_w2 = 0;

        coef_.setw0(0);
        coef_.setw1(w1_naive_);
        coef_.setw2(w2_naive_);

        settings_naive_.iter = 5;
        settings_naive_.zero_order = false;
        settings_naive_.first_order = true;
        settings_naive_.rank_w2 = 0;
        settings_naive_.rank_w3 = 0;
        settings_naive_.l2_reg_w1 = 1;

        coef_naive = fastfm::utils::InitFmCoef(settings_naive_, x_naive_.cols());
        coef_naive.setw1(w1_naive_);
    }

    // virtual void TearDown() {}

    SpMat x_naive_;
    SpMat x_pos_;
    SpMat x_c_;
    SpMat x_i_;

    Vector cost_naive_;
    Vector cost_pos_;
    Vector cost_ifm_;

    Vector y_naive_;
    Vector y_pos_;
    Vector y_ifm_;

    Matrix w2_naive_;
    Matrix w2_c_;
    Matrix w2_i_;

    Vector w1_naive_;
    Vector w1_c_;
    Vector w1_i_;

    double w0_;
    ModelParam coef_;
    SolverSettings settings_;

    ModelParam coef_naive;
    SolverSettings settings_naive_;

    const std::vector<double> v_empty_;

};
#endif //!EXTERNAL_RELEASE
#endif //FASTFM_CATCH_FIXTURE_H
