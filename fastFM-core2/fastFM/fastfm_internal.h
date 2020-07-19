#ifndef FASTFM_INTERNAL_H
#define FASTFM_INTERNAL_H

#define LOGURU_REPLACE_GLOG 1
#include "loguru.hpp"

#include "fastfm.h"
#include "fastfm_impl.h"
#include "fastfm_decl.h"
#include <iostream>

#include <unordered_map>

namespace fastfm {

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
    fm_settings settings_;

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
            else if (item.first == "step_size"){
                settings_.step_size = std::stod(item.second);
            }
            else if (item.first == "decay"){
                settings_.decay = std::stod(item.second);
            }
            else if (item.first == "lazy_decay"){
                settings_.lazy_decay = std::stod(item.second);
            }
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
            else if (item.first == "init_var_w2"){
                settings_.init_var_w2 = std::stod(item.second);
            }
            else if (item.first == "init_var_w3"){
                settings_.init_var_w3 = std::stod(item.second);
            }
            else if (item.first == "iter"){
                settings_.iter = std::stoi(item.second);
            }
            else if (item.first == "rng_seed"){
                settings_.rng_seed = std::stoi(item.second);
            }
            else if (item.first == "zero_order"){
                std::istringstream(item.second) >> std::boolalpha >> settings_.zero_order;
            }
            else if (item.first == "first_order"){
                std::istringstream(item.second) >> std::boolalpha >> settings_.first_order;
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
    int a;
    fm_coef* coef_;
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

#endif // FASTFM_INTERNAL_H

