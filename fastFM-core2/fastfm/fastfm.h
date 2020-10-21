
// Created by ibayer on 21.02.17.
//

#ifndef FASTFM_FASTFM_H
#define FASTFM_FASTFM_H

#include <memory>
#include <map>

namespace fastfm {
//    TODO add enumerate for split.
/*! \brief This class contains all settings needed to train a model.
*
*  Detailed description starts here.
*/
class Settings {
public:
    Settings();
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

/*! \brief This class contains all model parameter.
*
*  Detailed description starts here.
*/
class Model {
public:
    Model();
    ~Model();
    // A row major matrix is expected.
    void add_vector(const std::string& name, double* data, size_t size);
    void add_matrix(const std::string& name, double* data, size_t rows, size_t cols, bool rowMajor);
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


/*! \brief This class contains all data needed to train a model.
*
*  Detailed description starts here.
*/
class Data {

public:
    Data();
    ~Data();
    // TODO should use all const, requires changing SpMat to SpConstMat everywhere.
    void add_vector(const std::string& name, double* data, const size_t size);
    void add_matrix(const std::string& name, double* data, size_t rows, size_t cols, bool rowMajor);
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
  \param s the settings that specifiy how the model should be trained.
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
