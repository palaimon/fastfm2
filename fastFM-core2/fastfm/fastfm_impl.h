//
// Created by ibayer on 22.10.15.
//

#ifndef FASTFM_FASTFM_IMPL_H
#define FASTFM_FASTFM_IMPL_H

#define LOGURU_REPLACE_GLOG 1
#include "loguru.hpp"

#include <random> // std::mt19937
#include <memory>
#include "fastfm.h"
#include "fastfm_decl.h"

#include <unordered_set>
#include <unordered_map>
#include <iostream>

#define LOGURU_REPLACE_GLOG 1
#include "loguru.hpp"

namespace fastfm {

struct SolverSettings {

    std::string loss = "<empty>";
    std::string solver = "<empty>";

    int iter = 50;
    int rng_seed = 123;

    bool zero_order = true;
    bool first_order = true;

    int rank_w2 = 0;
    int rank_w3 = 0;

    double l2_reg_w0 = 0;
    double l2_reg_w1 = 0;
    double l2_reg_w2 = 0;
    double l2_reg_w3 = 0;

    double impl_reg = 1;
    bool first_order_impl_reg = true;

    std::vector<double> group_l2_reg_w2;

    double init_var_w2 = .1;
    double init_var_w3 = .1;

    // sgd specific
    double step_size = 0.01;
    double decay = 0.01;
    double lazy_decay = 0;
    int n_epoch = 10;
    bool clip_pred = true;
    bool clip_reg = true;
    bool lazy_reg = true;
};

class Evaluator {
    public:
    virtual void eval() = 0;
};

class ModelParam {
private:
    double w0 = 0;
    double *w0map;
    Vector w1;
    Eigen::Map< Vector > *w1map;
    Matrix w2;
    Eigen::Map< Matrix > *w2map;
    Matrix w3;
    Eigen::Map< Matrix > *w3map;

    Eigen::Map<Vector>* mapValues;
    std::unordered_map<std::string, VectorRef> mapValuesBlocks;
    std::unordered_map<std::string, Eigen::Map<Vector>> vectors;
    Vector dummy;

public:
    ModelParam() : w0map(NULL), w1map(NULL), w2map(NULL), w3map(NULL), mapValues(NULL) {
    }

    ModelParam(const SolverSettings& settings, const int n_features) : w0map(NULL), w1map(NULL),
                                                                       w2map(NULL), w3map(NULL),
                                                                       mapValues(NULL) {
        std::mt19937 mt_rand(settings.rng_seed);

        w0 = 0;
        w1 = Vector::Zero(n_features);

        {
            w2 = Matrix(settings.rank_w2, n_features);
            std::normal_distribution<double> distribution(0, std::sqrt(settings.init_var_w2));
            for (int i = 0; i < settings.rank_w2; ++i) {
                for (int j = 0; j < n_features; ++j) {
                    w2.coeffRef(i, j) = distribution(mt_rand);
                }
            }
        }

        {
            w3 = Matrix(settings.rank_w3, n_features);
            std::normal_distribution<double> distribution(0, std::sqrt(settings.init_var_w3));
            for (int i = 0; i < settings.rank_w3; ++i) {
                for (int j = 0; j < n_features; ++j) {
                    w3.coeffRef(i, j) = distribution(mt_rand);
                }
            }
        }
    }

    void setw0(double in) {
        if (w0map != NULL)
            *w0map = in;
        else
            w0 = in;
    }

    void setw0_ptr(double* in) {
        w0map = in;
    }

    void setw1(double* data, int rows) {
        if (w1map != NULL)
            delete w1map;
        w1map = new Eigen::Map<Vector>(data, rows);
    }

    void setw1(VectorRef vector) {
        if (w1map != NULL)
            delete w1map;
        w1map = new Eigen::Map<Vector>(vector.data(), vector.rows());
    }

    void setw2(double* data, int rows, int cols) {
        if (w2map != NULL)
            delete w2map;
        w2map = new Eigen::Map<Matrix>(data, rows, cols);
    }

    void setw2(MatrixRef matrix) {
        if (w2map != NULL)
            delete w2map;
        w2map = new Eigen::Map<Matrix>(matrix.data(), matrix.rows(), matrix.cols());
    }

    void setw3(double* data, int rows, int cols) {
        if (w3map != NULL)
            delete w3map;
        w3map = new Eigen::Map<Matrix>(data, rows, cols);
    }

    void setw3(MatrixRef matrix) {
        if (w3map != NULL)
            delete w3map;
        w3map = new Eigen::Map<Matrix>(matrix.data(), matrix.rows(), matrix.cols());
    }

    void setMapValues(const std::string& keys, double* values, size_t size) {
        delete mapValues;
        mapValues = new Eigen::Map<Vector>(values, size);

        auto shift = 0;
        std::string key;
        std::istringstream tokenStream(keys);
        while (std::getline(tokenStream, key, ',')) {
            mapValuesBlocks.emplace(key, mapValues->segment(shift, 1));
            ++shift;
        }
        // check that string-parsed total size equals to real memory size
        CHECK(size == shift);
    }

    double getw0() {
        if (w0map)
            return *w0map;
        return w0;
    }

    double *getw0_ptr() {
        return w0map;
    }

    VectorRef getw1() {
        if (w1map != NULL)
            return *w1map;
        return w1;
    }

    constVectorRef getw1() const {
        if (w1map != NULL)
            return *w1map;
        return w1;
    }

    VectorRef getw1block(const unsigned int rowstart, const unsigned int rownum) {
        if (w1map != NULL)
            return Eigen::Map< Vector, 0, Eigen::OuterStride<> >(&(*w1map)(rowstart, 0), rownum, Eigen::OuterStride<>(w1map->outerStride()));
        return Eigen::Map< Vector, 0, Eigen::OuterStride<> >(&w1(rowstart, 0), rownum, Eigen::OuterStride<>(w1.outerStride()));
    }
    
    constVectorRef getw1block(const unsigned int rowstart, const unsigned int rownum) const {
        if (w1map != NULL)
            return Eigen::Map< const Vector, 0, Eigen::OuterStride<> >(&(*w1map)(rowstart, 0), rownum, Eigen::OuterStride<>(w1map->outerStride()));
        return Eigen::Map< const Vector, 0, Eigen::OuterStride<> >(&w1(rowstart, 0), rownum, Eigen::OuterStride<>(w1.outerStride()));
    }

    MatrixRef getw2() {
        if (w2map != NULL)
            return *w2map;
        return w2;
    }

    constMatrixRef getw2() const {
        if (w2map != NULL)
            return *w2map;
        return w2;
    }

    MatrixRef getw2block(const unsigned int rowstart, const unsigned int colstart, const unsigned int rownum, const unsigned int colnum) {
        if (w2map != NULL) {
            if ((rownum == 0 && colnum == 0) || rowstart+1 > w2map->rows() || colstart+1 > w2map->cols())
                return Eigen::Map< Matrix, 0, Eigen::OuterStride<> >(w2map->data(), 0, 0, Eigen::OuterStride<>(0));
            return Eigen::Map< Matrix, 0, Eigen::OuterStride<> >(&(*w2map)(rowstart, colstart), rownum, colnum, Eigen::OuterStride<>(w2map->outerStride()));
        }
        return w2.block(rowstart, colstart, rownum, colnum);
    }

    constMatrixRef getw2block(const unsigned int rowstart, const unsigned int colstart, const unsigned int rownum, const unsigned int colnum) const {
        if (w2map != NULL) {
            if ((rownum == 0 && colnum == 0) || rowstart+1 > w2map->rows() || colstart+1 > w2map->cols())
                return Eigen::Map< const Matrix, 0, Eigen::OuterStride<> >(w2map->data(), 0, 0, Eigen::OuterStride<>(0));
            return Eigen::Map< const Matrix, 0, Eigen::OuterStride<> >(&(*w2map)(rowstart, colstart), rownum, colnum, Eigen::OuterStride<>(w2map->outerStride()));
        }
        return w2.block(rowstart, colstart, rownum, colnum);
    }

    MatrixRef getw3() {
        if (w3map != NULL)
            return *w3map;
        return w3;
    }

    constMatrixRef getw3() const {
        if (w3map != NULL)
            return *w3map;
        return w3;
    }

    VectorRef getMapValue(const std::string& k) {
        return mapValuesBlocks.at(k);
    }

    void add_vector(const std::string& name, double* data, size_t size) {
        // Store only unique keys
        auto res = vectors.emplace(name, Eigen::Map<Vector>(data, size));
        // Check if construction was successful
        CHECK(res.second);
    }

    VectorRef get_vector(const std::string& name) {
        if (has_vector(name)) {
            return vectors.at(name);
        } else {
            return dummy;
        }
    }

    bool has_vector(const std::string& name) const{
        return vectors.count(name) > 0;
    }
};

class Data::Impl{
 private:
    Eigen::Map<Vector> y_train;
    Eigen::Map<Vector> y_pred;
    Eigen::Map<Matrix> y_recs;

    std::unordered_map<std::string, Eigen::Map<SpMat>> x_;
    std::unordered_map<std::string, Eigen::Map<RowSpMat>> x_row_;
    std::unordered_map<std::string, Eigen::Map<Vector>> vectors;
    Vector dummy;

 public:
    bool has_col_major() const {
        return x_.size() > 0;
    }

    bool is_ranking() const {
        return y_recs.size() > 0;
    }

    bool check_row_major_train(){
            CHECK_GT(x_row_.size(), 0);
            CHECK_EQ(x_row_.at("x").rows(), y_train.size());
        return true;
    }

    bool check_icd_train(){
            CHECK_EQ(x_.size(), 3);
            CHECK_EQ(x_.at("x").rows(), y_train.size());
            CHECK_EQ(x_.at("x").cols(), x_.at("x_c").cols() + x_.at("x_i").cols());
        return true;
    }

    bool check_col_major_train(){
            CHECK_GT(x_.size(), 0);
            CHECK_EQ(x_.at("x").rows(), y_train.size());
        return true;
    }

    Eigen::Map<Vector> get_train_target() const
    {
        return y_train;
    }
    void wrap_train_target_memory(double* data, int n_samples)
    {
        // C++ placement operator. Does not allocate memory, but runs the constructor of the class on the memory pointer provided.
        new (&y_train) Eigen::Map<Vector>(data, n_samples);
    }

    Eigen::Map<Vector> get_prediction() const
    {
        return y_pred;
    }

    Eigen::Map<Matrix> get_recs() const
    {
        return y_recs;
    }

    void wrap_pred_memory(double* data, int n_samples)
    {
        // C++ placement operator. Does not allocate memory, but runs the constructor of the class on the memory pointer provided.
        new (&y_pred) Eigen::Map<Vector>(data, n_samples);
    }
    void wrap_pred_memory(double* data, const int n_rows, const int n_cols)
    {
        // C++ placement operator. Does not allocate memory, but runs the constructor of the class on the memory pointer provided.
        new (&y_recs) Eigen::Map<Matrix>(data, n_rows, n_cols);
    }

    int wrap_design_matrix_col_major(const std::string& name, double* data, int n_samples, int n_features, int nnz,
                                     int* outer, int* inner)
    {
        auto res = x_.emplace(name, Eigen::Map<SpMat>(n_samples, n_features, nnz, outer, inner, data));
        CHECK(res.second);
        return x_.size();
    }

    int wrap_design_matrix_row_major(const std::string& name, double* data, int n_samples, int n_features, int nnz,
                                     int* outer, int* inner)
    {
        auto res = x_row_.emplace(name, Eigen::Map<RowSpMat>(n_samples, n_features, nnz, outer, inner, data));
        CHECK(res.second);
        return x_row_.size();
    }

    Eigen::Map<SpMat> get_design_matrix_col_major() const
    {
        return x_.at("x");
    }

    Eigen::Map<RowSpMat> get_design_matrix_row_major() const
    {
        return x_row_.at("x");
    }

    Eigen::Map<SpMat> get_design_matrix_context_col_major() const
    {
        return x_.at("x_c");
    }

    Eigen::Map<RowSpMat> get_design_matrix_context_row_major() const
    {
        return x_row_.at("x_c");
    }

    Eigen::Map<SpMat> get_design_matrix_item_col_major() const
    {
        return x_.at("x_i");
    }

    Eigen::Map<RowSpMat> get_design_matrix_item_row_major() const
    {
        return x_row_.at("x_i");
    }

    void add_vector(const std::string& name, double* data, size_t size) {
        // Store only unique keys
        auto res = vectors.emplace(name, Eigen::Map<Vector>(data, size));
        // Check if construction was successful
        CHECK(res.second);
    }

    VectorRef get_vector(const std::string& name) {
        if (has_vector(name)) {
            return vectors.at(name);
        } else {
            return dummy;
        }
    }

    bool has_vector(const std::string& name) const{
        return vectors.count(name) > 0;
    }

    Impl() : y_train(NULL, 0), y_pred(NULL, 0), y_recs(NULL, 0, 0) {}
};

class Settings::Impl{
 public:
    SolverSettings settings_;

    Impl() = default;

    explicit Impl(const std::map<std::string, std::string>& settings_map)
    {
        for (const auto& item : settings_map) {
            if (item.first == "solver"){
                settings_.solver = item.second;
            }
            else if (item.first == "loss"){
                settings_.loss = item.second;
            }
//
            else if (item.first == "iter"){
                settings_.iter = std::stoi(item.second);
            }
            else if (item.first == "rng_seed"){
                settings_.rng_seed = std::stoi(item.second);
            }
//
            else if (item.first == "zero_order"){
                std::istringstream(item.second) >> std::boolalpha >> settings_.zero_order;
            }
            else if (item.first == "first_order"){
                std::istringstream(item.second) >> std::boolalpha >> settings_.first_order;
            }
//
            else if (item.first == "l2_reg_w0"){
                settings_.l2_reg_w0 = std::stod(item.second);
            }
            else if (item.first == "l2_reg_w1"){
                settings_.l2_reg_w1 = std::stod(item.second);
            }
            else if (item.first == "l2_reg_w2"){
                settings_.l2_reg_w2 = std::stod(item.second);
            }
            else if (item.first == "l2_reg_w3"){
                settings_.l2_reg_w3 = std::stod(item.second);
            }
//
            else if (item.first == "init_var_w2"){
                settings_.init_var_w2 = std::stod(item.second);
            }
            else if (item.first == "init_var_w3"){
                settings_.init_var_w3 = std::stod(item.second);
            }
//
            //sgd
            else if (item.first == "step_size"){
                settings_.step_size = std::stod(item.second);
            }
            else if (item.first == "decay"){
                settings_.decay = std::stod(item.second);
            }
            else if (item.first == "lazy_decay"){
                settings_.lazy_decay = std::stod(item.second);
            }
            else if (item.first == "n_epoch"){
                settings_.n_epoch = std::stoi(item.second);
            }
            else if (item.first == "clip_pred"){
                std::istringstream(item.second) >> std::boolalpha >> settings_.clip_pred;
            }
            else if (item.first == "clip_reg"){
                std::istringstream(item.second) >> std::boolalpha >> settings_.clip_reg;
            }
            else if (item.first == "lazy_reg"){
                std::istringstream(item.second) >> std::boolalpha >> settings_.lazy_reg;
            }
            else {
                    LOG(ERROR) << "Parameter " << item.first << " is not supported.";
                CHECK(false);
            }
        }
    }
};

class Model::Impl{
 public:
    ModelParam* coef_;
};

class Internal{
 public:
    static Model::Impl* get_impl(Model* m){
        return m->mImpl;
    }
    static Settings::Impl* get_impl(Settings* s){
        return s->mImpl;
    }
    static Data::Impl* get_impl(Data* d){
        return d->mImpl;
    }
};

}
#endif //FASTFM_FASTFM_IMPL_H
