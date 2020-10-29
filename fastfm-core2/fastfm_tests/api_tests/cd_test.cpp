#include <Eigen/Dense>

//#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch/catch.hpp"

#include "fixture.h"
#include "fastfm.h"

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;
using namespace fastfm;


TEST_CASE_METHOD(FMExample, "FMExample predict", "[API]") {
    // Minimal example to test the api predict function on a small example.

    // Allocate space for the predictions.
    Vector y_pred = Vector::Zero(x_.rows());

    // Object to hold train, test and validation data.
    Data* d = DataFactory(x_, &y_pred).get();

    // Object to hold the model parameter.
    Model* m = ModelFactory(&w0_, w1_, w2_).get();

    // Make predictions for split
    predict(m, d);

    // Set true reference predictions.
    Vector y_true(4);
    y_true << 37, 26, 148, 178;    

    // Test predictions against true values.
    for(int i=0; i<y_true.size(); i++)
        REQUIRE(y_true[i] == y_pred.coeff(i));
//        ASSERT_EQ(y_true[i], y_pred.coeff(i));

    // delete data, model
    delete d;
    delete m;
}

TEST_CASE_METHOD(FMExample, "FMExample fit", "[API]") {
    // Minimal example to test the api fit function on a small regression example.
    // TODO Test each combination of order individually. Only first, only second etc..

    // Allocate space for the targets & predictions.
    Vector y_true = Vector::Ones(x_.rows());
    Vector y_pred = Vector::Zero(x_.rows());

    // Object to hold train, test and validation data.
    auto d = DataFactory(x_, &y_pred, &y_true).get();

    // Change true coefficients
    w0_ += 0.5;
    w1_.setRandom();
    w2_.setRandom();

    // Object to hold the model parameter.
    auto m = ModelFactory(&w0_, w1_, w2_).get();

    double w0 = w0_;
    double w1_init_norm = w1_.norm();
    double w2_init_norm = w2_.norm();

    // Object to hold the model parameter.
    std::map<std::string, std::string> settings_ = {
            {"solver", "cd"},
            {"loss", "squared"}
    };
    Settings*s = new Settings(settings_);

    // Make predictions with the initial parameters.
    predict(m, d);
    double init_train_error = (y_pred - y_true).norm();

    fit(s, m, d);

    // Check that all parameter changed.
    REQUIRE_FALSE(w1_init_norm == w1_.norm());
    REQUIRE_FALSE(w2_init_norm == w2_.norm());
    REQUIRE_FALSE(w0 == w0_);

    // Make predictions with the updated parameters.
    predict(m, d);

    // Check that training error decreased.
    REQUIRE((y_pred - y_true).norm() < init_train_error);

    // delete data, model, settings
    delete d;
    delete m;
    delete s;
}
