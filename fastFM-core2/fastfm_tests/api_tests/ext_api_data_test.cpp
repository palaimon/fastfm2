#include <Eigen/Dense>

#include "fastfm.h"
#include "fastfm.cpp"
#include "../3rdparty/catch/catch.hpp"

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using namespace fastfm;

TEST_CASE("Check default settings", "[check_default_settings]") {
    Settings* s = new Settings();
    REQUIRE(Approx(Internal::get_impl(s)->settings_.init_var_w2) == 0.1);

    // delete settings
    delete s;
}

TEST_CASE("Settings deserialization", "[deserialization]") {

    std::map<std::string, std::string> cppjson =
      {
              {"iter", "1000"},
              {"rng_seed", "567"},
              {"zero_order", "false"},
              {"first_order", "false"},
              {"l2_reg_w0", "0.10"},
              {"l2_reg_w1", "10.1"},
              {"l2_reg_w2", "20.2"},
              {"l2_reg_w3", "30.3"},
              {"init_var_w2", "0.25"},
              {"init_var_w3", "0.55"},
              {"step_size", "0.001"},
              {"decay", "0.002"},
              {"lazy_decay", "0.003"},
              {"clip_pred", "false"},
              {"clip_reg", "false"},
              {"lazy_reg", "false"}
      };

    Settings* s = new Settings(cppjson);

    REQUIRE(Internal::get_impl(s)->settings_.iter == 1000);
    REQUIRE(Internal::get_impl(s)->settings_.rng_seed == 567);
    REQUIRE_FALSE(Internal::get_impl(s)->settings_.zero_order);
    REQUIRE_FALSE(Internal::get_impl(s)->settings_.first_order);
    REQUIRE(Approx(Internal::get_impl(s)->settings_.l2_reg_w0) == 0.1);
    REQUIRE(Approx(Internal::get_impl(s)->settings_.l2_reg_w1) == 10.1);
    REQUIRE(Approx(Internal::get_impl(s)->settings_.l2_reg_w2) == 20.2);
    REQUIRE(Approx(Internal::get_impl(s)->settings_.l2_reg_w3) == 30.3);
    REQUIRE(Approx(Internal::get_impl(s)->settings_.init_var_w2) == 0.25);
    REQUIRE(Approx(Internal::get_impl(s)->settings_.init_var_w3) == 0.55);
    REQUIRE(Approx(Internal::get_impl(s)->settings_.step_size) == 0.001);
    REQUIRE(Approx(Internal::get_impl(s)->settings_.decay) == 0.002);
    REQUIRE(Approx(Internal::get_impl(s)->settings_.lazy_decay) == 0.003);
    REQUIRE_FALSE(Internal::get_impl(s)->settings_.clip_pred);
    REQUIRE_FALSE(Internal::get_impl(s)->settings_.clip_reg);
    REQUIRE_FALSE(Internal::get_impl(s)->settings_.lazy_reg);

    delete s;
}

TEST_CASE("Add parameter to Model", "[add_parameter]") {
    Model* m = new Model();

    // w3 = |1 2 3|
    //      |4 5 6|
    Matrix w3(2, 3);
    w3 << 1, 2, 3,
          4, 5, 6;

    // w2 = |6 0 2|
    //      |5 1 0|
    Matrix  w2(2, 3);
    w2 << 6, 0, 2,
          5, 1, 0;

    // w = [9 8 7]
    Matrix w1(1, 3);
    w1 << 9, 8, 7;

    Matrix w0(1, 1);
    w0 << 2;

    m->add_vector("w0", w0.data(), w0.size());
    m->add_vector("w1", w1.data(), w1.size());
    m->add_matrix("w2", w2.data(), w2.rows(), w2.cols(), true);
    m->add_matrix("w3", w3.data(), w3.rows(), w3.cols(), true);

    std::string keys = "one,ten,hundred";
    Vector y(3);
    y << 1.1, 10.2, 100.3;

    m->add_scalar_map(keys, y.data(), y.size());

    Vector l2(5);
    l2 << 1.1, 2.2, 3.3, 4.4, 5.5;

    m->add_vector("l2", l2.data(), l2.size());

    REQUIRE(Approx(Internal::get_impl(m)->coef_->getw0()) == w0.coeff(0, 0));
    REQUIRE(Approx(Internal::get_impl(m)->coef_->getw1().sum()) == w1.sum());
    REQUIRE(Approx(Internal::get_impl(m)->coef_->getw2().norm()) == w2.norm());
    REQUIRE(Approx(Internal::get_impl(m)->coef_->getw3().norm()) == w3.norm());

    REQUIRE(Approx(Internal::get_impl(m)->coef_->getMapValue("one").coeff(0)) == 1.1);
    REQUIRE(Approx(Internal::get_impl(m)->coef_->getMapValue("ten").coeff(0)) == 10.2);
    REQUIRE(Approx(Internal::get_impl(m)->coef_->getMapValue("hundred").coeff(0)) == 100.3);

    REQUIRE(Approx(Internal::get_impl(m)->coef_->getMapValue("one").size()) == 1);
    REQUIRE(Approx(Internal::get_impl(m)->coef_->getMapValue("ten").size()) == 1);
    REQUIRE(Approx(Internal::get_impl(m)->coef_->getMapValue("hundred").size()) == 1);

    REQUIRE(Approx(Internal::get_impl(m)->coef_->get_vector("l2").size()) == 5);
    REQUIRE(Approx(Internal::get_impl(m)->coef_->get_vector("l2").sum()) == 16.5);
    REQUIRE(Internal::get_impl(m)->coef_->get_vector("l2").data() == l2.data());

    delete m;
}

TEST_CASE("Default Settings", "[default_settings]") {
    Settings* s = new Settings();
    SolverSettings default_;
    REQUIRE(Approx(Internal::get_impl(s)->settings_.init_var_w2) == default_.init_var_w2);

    // delete settings
    delete s;
}

TEST_CASE("Data, add prediction", "[add_prediction]") {
// w = [9 8 7]
    Vector y(3);
    y << 9, 8, 7;
    Data* d = new Data();

    d->add_vector("y_pred", y.data(), y.size());
    REQUIRE(Approx(Internal::get_impl(d)->get_prediction().sum()) == y.sum());

    delete d;
}

TEST_CASE("Data, add target", "[add_target]") {
// w = [9 8 7]
    Vector y(3);
    y << 9, 8, 7;
    Data* d = new Data();

    d->add_vector("y_true", y.data(), y.size());
    REQUIRE(Approx(Internal::get_impl(d)->get_train_target().sum()) == y.sum());
    delete d;
}

TEST_CASE("Data, column major matrix", "[add_design_matrix_col_major]") {

    SpMat x;
    {
        Matrix tmp(4, 3);
        tmp <<  1, 2, 0,
                0, 3, 0,
                4, 0, 2,
                4, 5, 0;
        x = tmp.sparseView();
    }
    Data* d = new Data();

    d->add_sparse_matrix("x", x.valuePtr(), x.rows(), x.cols(), x.nonZeros(),
                         x.outerIndexPtr(), x.innerIndexPtr(), true);
    REQUIRE(Approx(Internal::get_impl(d)->get_design_matrix_col_major().norm()) == x.norm());
    delete d;
}

TEST_CASE("Data, row major matrix", "[add_design_matrix_row_major]") {

    RowSpMat x;
    {
        Matrix tmp(4, 3);
        tmp <<  1, 2, 0,
                0, 3, 0,
                4, 0, 2,
                4, 5, 0;
        x = tmp.sparseView();
    }
    Data* d = new Data();

    d->add_sparse_matrix("x", x.valuePtr(), x.rows(), x.cols(), x.nonZeros(),
                         x.outerIndexPtr(), x.innerIndexPtr(), false);
    REQUIRE(Approx(Internal::get_impl(d)->get_design_matrix_row_major().norm()) == x.norm());
    delete d;
}

TEST_CASE("Data, add vector", "[add_vector]") {
	Vector x(3);
	x << 1, 10.1, 101.01;
	Vector y(3);
	y << 2, 20.2, 202.02;
	Vector z(3);
	z << 3, 30.3, 303.03;
	Data* d = new Data();

    d->add_vector("x", x.data(), x.size());
    d->add_vector("y", y.data(), y.size());
    d->add_vector("z", z.data(), z.size());
	REQUIRE(Approx(Internal::get_impl(d)->get_vector("y").sum()) == y.sum());
	REQUIRE(Approx(Internal::get_impl(d)->get_vector("x").sum()) == x.sum());
	REQUIRE(Approx(Internal::get_impl(d)->get_vector("z").sum()) == z.sum());
	delete d;
}