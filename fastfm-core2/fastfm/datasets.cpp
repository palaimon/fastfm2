//
// Created by ibayer on 01.08.17.
//

#define LOGURU_REPLACE_GLOG 1
#include "loguru.hpp"

#include "cd_impl.h"

#include <unordered_map>
#include "datasets.h"

#include <iostream>


namespace fastfm {
namespace utils {

using Triplet = Eigen::Triplet<double>;
using TripletList = std::vector<Eigen::Triplet<double>>;


DataGenerator::DataGenerator(int n_samples, std::vector<int> group_sizes, std::vector<int> rank,
                             std::vector<double> stddev, int rng_seed):
        n_samples_(n_samples), n_features_(0), group_sizes_(group_sizes), rank_(rank), stddev_(stddev), rng_seed_(rng_seed){
    create_design_matrix();
    create_model_parameter();
};

void DataGenerator::create_design_matrix() {

    std::mt19937 mt_rand =std::mt19937(rng_seed_);
    TripletList triplet_list;
    std::vector<SpMat> x_list;
    int column_offset = 0;
    std::uniform_int_distribution<int> rand_int_sampler(0,9);

    for (int size: group_sizes_){
        SpMat x_tmp;
        int n_cols = n_samples_ / size;

        // Assign the levels in random order.
        std::vector<int>cols(n_samples_);
        std::iota(cols.begin(), cols.end(), 0);
        std::shuffle(cols.begin(), cols.end(), mt_rand);
        for (int i=0; i < n_samples_; ++i){

            // Use random integer for dense feature vector values.
            const int value = ((n_cols == 1) ? rand_int_sampler(mt_rand) : 1);
            const int col_index =  column_offset + (cols[i] % n_cols);

            triplet_list.push_back(Triplet(i, col_index, value));
            if (col_index + 1 > n_features_) n_features_ = col_index + 1;
        }
        column_offset += n_cols;
    }

    x_ = SpMat(n_samples_, n_features_);
    x_.reserve(triplet_list.size());
    x_.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

void DataGenerator::create_model_parameter(){
    CHECK_GT(rank_.size(), 0);
    CHECK_GT(stddev_.size(), 0);

    std::mt19937 mt_rand =std::mt19937(rng_seed_);

    // In case of uniform stddev for all interations.
    if (rank_.size() > stddev_.size()) {
        CHECK_EQ(stddev_.size(), 1);
        const double tmp = stddev_[0];
        stddev_.resize(rank_.size());
        std::fill(stddev_.begin(), stddev_.end(), tmp);
    }

    if (rank_[0] > 0) {
       std::normal_distribution<> normal(0, stddev_[0]);
       w0_ = normal(mt_rand);
    } else {
        w0_ = 0;
    }
    if (rank_.size() == 1) return;

    if (rank_[1] > 0) {
        std::normal_distribution<> normal(0, stddev_[1]);
        w1_.resize(n_features_);
        for (int i=0; i < n_features_; ++i) w1_[i] = normal(mt_rand);
    }
    if (rank_.size() == 2) return;

    if (rank_[2] > 0) {
        std::normal_distribution<> normal(0, stddev_[2]);
        w2_.resize(rank_[2], n_features_);
        for (int i=0; i < w2_.rows(); ++i)
            for (int j=0; j < w2_.cols(); ++j)
                w2_.coeffRef(i, j) = normal(mt_rand);
    }
    if (rank_.size() == 3) return;

    if (rank_[3] > 0) {
        std::normal_distribution<> normal(0, stddev_[3]);
        w3_.resize(rank_[3], n_features_);
        for (int i=0; i < w3_.rows(); ++i)
            for (int j=0; j < w3_.cols(); ++j)
                w3_.coeffRef(i, j) = normal(mt_rand);
    }
            CHECK_EQ(rank_.size(), 4);
}

Vector DataGenerator::y_reg(double noise_stddev) {
    std::mt19937 mt_rand =std::mt19937(rng_seed_);
    Vector y(x_.rows());
    y.setZero();
    fastfm::cd::impl::Predict(x_, w3_, w2_, w1_, w0_, y);

    std::normal_distribution<> normal(0, noise_stddev);
    for (int i=0; i < y.rows(); ++i) y[i] += normal(mt_rand);

    return  y;
}

Vector DataGenerator::y_class(double noise_stddev, double flip_proba, double class_ratio) {
    std::mt19937 mt_rand =std::mt19937(rng_seed_);
    Vector y = y_reg(noise_stddev);

    std::uniform_real_distribution<double> uniform(0.0,1.0);
    // Sigmoid Transform + class threshold
    for (int i=0; i < y.rows(); ++i){
        const double sigmoid = 1. / (1 + exp(- y[i]));

        // Assign correct class only if label shouldn't be flipped.
        double label =  ((sigmoid > class_ratio) & (flip_proba < uniform(mt_rand))) ? 0 : 1;
        y[i] = label;
    }
    return  y;
}

RecDataGenerator::RecDataGenerator(int n_context, int n_item, int rank, int n_context_features, int n_item_features,
                                   int n_active_features, std::vector<double> stddev, int rng_seed):
    n_context_(n_context), n_item_(n_item), rank_(rank), n_context_features_(n_context_features),
    n_item_features_(n_item_features), n_active_features_(n_active_features), stddev_(stddev), rng_seed_(rng_seed){

    n_features_ = n_context_features_ + n_item_features_;

    CHECK_GT(n_item_, n_active_features_);

    create_design_matrix();
    create_model_parameter();
    create_target();
};

void RecDataGenerator::create_design_matrix() {
    std::mt19937 mt_rand =std::mt19937(rng_seed_);

    // sample x_c
    {
        TripletList triplet_list_c;
        std::uniform_int_distribution<int> rand_context_sampler(0, n_context_features_ -1);
        for (int i=0; i < n_context_; ++i){
            for (int j=0; j < n_active_features_; ++j){
                const int col = rand_context_sampler(mt_rand);
                triplet_list_c.push_back(Triplet(i, col, 1));
            }
        }
        x_c_ = SpMat(n_context_, n_context_features_);
        x_c_.reserve(triplet_list_c.size());
        x_c_.setFromTriplets(triplet_list_c.begin(), triplet_list_c.end());
    }

    // sample x_i
    {
        TripletList triplet_list_i;
        std::uniform_int_distribution<int> rand_item_sampler(0, n_item_features_ -1);
        for (int i=0; i < n_item_; ++i){
            for (int j=0; j < n_active_features_; ++j){
                const int col = rand_item_sampler(mt_rand);
                triplet_list_i.push_back(Triplet(i, col, 1));
            }
        }
        x_i_ = SpMat(n_item_, n_item_features_);
        x_i_.reserve(triplet_list_i.size());
        x_i_.setFromTriplets(triplet_list_i.begin(), triplet_list_i.end());
    }
}

void RecDataGenerator::create_model_parameter(){
            CHECK_GT(stddev_.size(), 0);

    std::mt19937 mt_rand =std::mt19937(rng_seed_);

    {
        std::normal_distribution<> normal(0, stddev_[1]);
        w1_.resize(n_features_);
        for (int i=0; i < n_features_; ++i) w1_[i] = normal(mt_rand);
    }

    {
        std::normal_distribution<> normal(0, stddev_[2]);
        w2_.resize(rank_, n_features_);
        for (int i=0; i < w2_.rows(); ++i)
            for (int j=0; j < w2_.cols(); ++j)
                w2_.coeffRef(i, j) = normal(mt_rand);
    }
}

void RecDataGenerator::create_target() {
    std::mt19937 mt_rand =std::mt19937(rng_seed_);

    TripletList triplet_list;

    int n_ratings = 5;

    std::uniform_int_distribution<int> rand_item_sampler(0, n_item_ -1);
    for (int i=0; i < n_context_; ++i){
        for (int j=0; j < n_ratings; ++j){
            const int col = rand_item_sampler(mt_rand);
            triplet_list.push_back(Triplet(i, col, 1));
        }
    }

    r_ = SpMat(n_context_, n_item_);
    r_.reserve(triplet_list.size());
    r_.setFromTriplets(triplet_list.begin(), triplet_list.end());
}



}
}
