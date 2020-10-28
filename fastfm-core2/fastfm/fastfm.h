
// Created by ibayer on 21.02.17.
//

#ifndef FASTFM_FASTFM_H
#define FASTFM_FASTFM_H

#include <memory>
#include <map>

namespace fastfm {

/** @brief Class encapsulating training settings.
 *
 * Please note that the class only specifies the interface, not implementation.
 * Contains all settings needed to train a model.
 * All fastfm `fit` functions accept Ptr\<Settings\> as parameter.
 */
class Settings {
public:
    Settings();
    /** Constructor in the case of solver settings string serialization
      *
      * \param settings_map <name>-<value> map of strings
      */
    explicit Settings(const std::map<std::string, std::string>& settings_map);
    ~Settings();
    class Impl;
private:
    // non copyable
    Settings(const Settings&);
	Settings& operator=(const Settings&);

    friend class Internal;
    Impl* mImpl;
};

/** @brief Class encapsulating training model.
 *
 * Please note that the class only specifies the interface, not implementation.
 * Contains all model parameters.
 * All fastfm `fit` and `predict` functions accept Ptr\<Model\> as parameter.
 */
class Model {
public:
    Model();
    ~Model();
    /** @brief  Vector expression mapping an existing array of data.
     *
     * Use name `w0` or `w1` for default FM model parameters.
     * Row major vector is expected, so for scalar param `w0` function requires size == 1.
     * `w1` - floating-point vector of `<number_of_rows>` size.
     *
     * @param name name of model parameter
     * @param data pointer to the array location to map the memory
     * @param size number of rows of the vector expression
     */
    void add_vector(const std::string& name, double* data, size_t size);


    /** @brief Matrix expression mapping an existing array of data.
     *
     * For 2d FM model parameters supported names are `w2` and `w3`
     *
     * @param name name of model parameter
     * @param data pointer to the array location to map the memory
     * @param rows number of rows of the matrix expression
     * @param cols number of columns of the matrix expression
     * @param rowMajor unused, rowMajor matrix expected!!
     */
    void add_matrix(const std::string& name, double* data, size_t rows, size_t cols, bool rowMajor);


    /** @brief Packed scalars mapping an existing array of data.
     *
     * Used for mapping of additional named scalar values packed together in a contiguous array.
     * Keys should be string formatted like 'key1,key2,keyN', where `key*` - parameter name. Do not use spaces.
     *
     * Correct keys example:
     * 'alpha,beta,gamma,mu', while `size` should equal to the number of parameters. In this case size==4.
     * `values` should point to a floating-point rowMajor vector of `<number_of_params>` size.
     *
     * @param keys string of comma separated names
     * @param values pointer to the array location to map the memory
     * @param size number of packed values
     */
    void add_scalar_map(const std::string& keys, double* values, size_t size);

    class Impl;
private:
    // non copyable
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    friend class Internal;
    friend class ModelFactory;
    Impl* mImpl;
};


/** @brief Class encapsulating training data.
 *
 * Please note that the class only specifies the interface, not implementation.
 * Contains all data to train the model.
 * All fastfm `fit` and `predict` functions accept Ptr\<Data\> as parameter.
 */
class Data {

public:
    Data();
    ~Data();

    /** @brief  Vector expression mapping an existing array of data.
     *
     * Use name `y_true`, `y_pred` or `y_train_pred` for default FM data arrays.
     * Row major vector is expected.
     *
     * @param name name of data parameter
     * @param data pointer to the array location to map the memory
     * @param size number of samples
     */
    // TODO should use all const, requires changing SpMat to SpConstMat everywhere.
    void add_vector(const std::string& name, double* data, const size_t size);


    /** @brief Matrix expression mapping an existing array of data.
     *
     * For 2d FM data parameters currently supported name is `y_rec` only.
     *
     * @param name name of data parameter
     * @param data pointer to the array location to map the memory
     * @param rows number of rows of the matrix expression
     * @param cols number of columns of the matrix expression
     * @param rowMajor unused, rowMajor matrix expected!!
     */
    void add_matrix(const std::string& name, double* data, size_t rows, size_t cols, bool rowMajor);


    /** @brief Sparse Matrix expression mapping an existing array of data.
     *
     * For sparse FM data parameters currently supported names are: `x`, `x_c`, `x_i`
     *
     * @param name name of data parameter
     * @param data pointer to the array location to map the memory
     * @param rows number of samples
     * @param cols number of features
     * @param nnz number of non-zeros of each column (resp. row).
     * @param outer stores the row (resp. column) indices of the non-zeros.
     * @param inner stores the col (resp. row) indices of the non-zeros
     * @param col_major True if storage order is column major
     */
    void add_sparse_matrix(const std::string& name, double* data, size_t rows, size_t cols,
                           int nnz, int* outer, int* inner, bool col_major);
    class Impl;
private:
    // non copyable
    Data(const Data&) = delete;
    Data& operator=(const Data&) = delete;

    friend class Internal;
    friend class DataFactory;
    Impl* mImpl;   
};


//!Python function type
typedef void* python_function_t;

//!Fit callback type
typedef bool (*fit_callback_t)(std::string json_in, python_function_t python_func);

//! Fits a model using the specified settings, data and callback.
/*!
  \param s the settings that specify how the model should be trained.
  \param m the initial model parameter.
  \param d the data required for the selected settings.
  \sa QTstyle_Test(), ~QTstyle_Test(), testMeToo() and publicVar()
  TODO add link to the regression tests.
*/
void fit(Settings* s, Model* m, Data* d, fit_callback_t cb, python_function_t python_func);

//! Fits a model without a callback progress function.
void fit(Settings* s, Model* m, Data* d);

//! Make predictions with a trained model for the given data.
/*!
  \param m the model parameter.
  \param d the data required to make the predictions.
  \sa QTstyle_Test(), ~QTstyle_Test(), testMeToo() and publicVar()
  TODO add link to the regression tests.
  TODO can we make the Model argument const?
*/
void predict(Model* m, Data* d);
}
#endif //FASTFM_FASTFM_H
