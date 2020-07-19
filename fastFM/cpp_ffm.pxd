# Author: Immanuel Bayer
# License: BSD 3 clause
#distutils: language=c++

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.map cimport map as cpp_map

cdef extern from "../fastFM-core2/fastFM/fastfm.h" namespace "fastfm":

    cdef cppclass Settings:
        Settings()
        Settings(cpp_map[string, string] settings)

    cdef cppclass Model:
        Model()
        void add_vector(const string name, double* data, size_t size)
        void add_matrix(const string name, double* data, size_t rows, size_t cols,
                        bool row_major)
        void add_scalar_map(const string keys, double* values, size_t size)


    cdef cppclass Data:
        Data()
        void add_vector(const string name, double* data, const size_t size)
        void add_matrix(const string name, double* data,
                        size_t rows, size_t cols, bool row_major)
        void add_sparse_matrix(const string name, double* data,
                               size_t rows, size_t cols, int nnz,
                               int* outer, int* inter, bool col_major)

    ctypedef void* python_function_t
    ctypedef bool (*fit_callback_t)(string json_in, python_function_t python_func)

    cdef void fit(Settings* s, Model* m, Data* d,
                  fit_callback_t callback, python_function_t python_callback_func)
    cdef void predict(Model* m, Data* d)
