# cython: language_level=3

# Author: Immanuel Bayer
# License: BSD 3 clause

import json

cimport cpp_ffm
from cpp_ffm cimport Settings, Data, Model
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.map cimport map as cpp_map

import scipy.sparse as sp

cimport numpy as np
import numpy as np

cdef bytes to_c_str(text):
     if isinstance(text, bytes):
         return <bytes> text
     else:
         return text.encode('ascii')

cdef unicode to_py_str(text):
    if isinstance(text, unicode):
        return <unicode> text
    else:
        return text.decode('ascii')


cdef Model* _model_factory(np.ndarray[np.float64_t, ndim = 1] w_0,
        np.ndarray[np.float64_t, ndim = 1] w,
        np.ndarray[np.float64_t, ndim = 2] V):

    cdef Model *m = new Model()
    rank = V.shape[0]
    n_features = V.shape[1]

    m.add_vector(to_c_str("w0"), <double*> w_0.data, 1)
    m.add_vector(to_c_str("w1"), <double*> w.data, n_features)
    m.add_matrix(to_c_str("w2"), <double*> V.data, rank, n_features, True)

    return m


cdef _add_sparse_matrix(name, Data* d, X):
    # get attributes from csc scipy
    n_features = X.shape[1]
    n_samples = X.shape[0]
    nnz = X.count_nonzero()

    if not (sp.isspmatrix_csc(X) or sp.isspmatrix_csr(X)):
        raise Exception("matrix format is not supported")

    cdef np.ndarray[int, ndim=1, mode='c'] inner = X.indices
    cdef np.ndarray[int, ndim=1, mode='c'] outer = X.indptr
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] data = X.data

    d.add_sparse_matrix(to_c_str(name), &data[0], n_samples, n_features, nnz,
                        &outer[0], &inner[0], sp.isspmatrix_csc(X))


def ffm_top_k_retrieval(np.ndarray[np.float64_t, ndim = 1] w_0,
                        np.ndarray[np.float64_t, ndim = 1] w,
                        np.ndarray[np.float64_t, ndim = 2] V,
                        C, I, top_k):
    assert top_k <= I.shape[0], "top_k can't be larger then number of items"
    n_features = len(w)
    n_context = C.shape[0]
    assert n_features == V.shape[1]
    assert C.shape[1] + I.shape[1] == n_features

    # allocate memory for predictions
    n_samples = n_context * top_k
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] y =\
         np.zeros((n_samples, 2), dtype=np.float64)

    m = _model_factory(w_0, w, V)

    cdef Data *d = new Data()
    _add_sparse_matrix("x_c", d, C)
    _add_sparse_matrix("x_i", d, I)
    d.add_matrix(to_c_str("y_rec"),
                 <double *> y.data, y.shape[0], y.shape[1], True)

    cpp_ffm.predict(m, d)

    del m
    del d

    pos = y[:, 0].reshape((int(y.shape[0] / top_k), top_k)).astype(int)
    scores = y[:, 1].reshape((int(y.shape[0] / top_k), top_k))

    return pos, scores


def ffm_predict(np.ndarray[np.float64_t, ndim = 1] w_0,
        np.ndarray[np.float64_t, ndim = 1] w,
        np.ndarray[np.float64_t, ndim = 2] V, X):
    n_samples, n_features = X.shape
    assert n_features == len(w)
    assert n_features == V.shape[1]

    # allocate memory for predictions
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] y =\
         np.zeros(n_samples, dtype=np.float64)

    m = _model_factory(w_0, w, V)

    cdef Data *d = new Data()
    _add_sparse_matrix("x", d, X)
    d.add_vector(to_c_str("y_pred"), &y[0], n_samples)

    cpp_ffm.predict(m, d)

    del m
    del d

    return y


cdef bool fit_callback_wrapper(string json_in, void* python_function):
    """
    The main piece of the glue between Python, Cython and C++.
    It wraps the python function so it can be used in C++ space.
    """
    f = (<object>python_function)
    params = json.loads(to_py_str(json_in))

    if f is not None:
        try:
            return f(params)
        except Exception as e:
            print(str(e))

    return False

def ffm_fit(np.ndarray[np.float64_t, ndim = 1] w_0,
            np.ndarray[np.float64_t, ndim = 1] w,
            np.ndarray[np.float64_t, ndim = 2] V,
            X, np.ndarray[np.float64_t, ndim = 1] y,
            np.ndarray[np.float64_t, ndim = 1] y_train_pred_=None,
            str keys=None, np.ndarray[np.float64_t, ndim = 1] values=None,
            np.ndarray[np.float64_t, ndim = 1] lambda_w2=None,
            np.ndarray[np.float64_t, ndim = 1] mu_w2=None,
            C=None, I=None,
            np.ndarray[np.float64_t, ndim = 1] cost=None,
            np.ndarray[np.float64_t, ndim = 1] x_c_cost=None,
            np.ndarray[np.float64_t, ndim = 1] x_i_cost=None,
            dict settings=None,
            callback=None):

    assert isinstance(settings, dict)
    n_samples = X.shape[0]

    if y is not None:
        assert n_samples == len(y) # test shapes

    #cdef Settings* s = new Settings(json.dumps(settings).encode())

    cdef cpp_map[string, string] strmap

    # py-cpp inconsistencies
    # remove unused
    if "l2_reg" in settings:         # used on py side only
        del settings["l2_reg"]
    if "init_stdev" in settings:     # used on py side only
        del settings["init_stdev"]
    if "random_state" in settings:   # used on py side only
        del settings["random_state"]
    if "rank" in settings:           # derived from V shape
        del settings["rank"]
    if "copy_X" in settings:         # inherited/unused
        del settings["copy_X"]

    # map that differs
    if "l2_reg_w" in settings:
        settings["l2_reg_w1"] = settings.pop("l2_reg_w")
    if "l2_reg_V" in settings:
        settings["l2_reg_w2"] = settings.pop("l2_reg_V")
    if "n_iter" in settings:
        settings['iter'] = settings.pop('n_iter')
    if settings['loss'] != 'bpr' and "step_size" in settings:
        settings["decay"] = str(-float(settings.pop("step_size")))

    for i in settings.iterkeys():
        strmap[to_c_str(i)] = to_c_str(settings[i])
    cdef Settings* s = new Settings(strmap)

    m = _model_factory(w_0, w, V)
    if keys is not None and values is not None:
        m.add_scalar_map(to_c_str(keys), <double*> values.data, values.size)

    if lambda_w2 is not None:
        m.add_vector(to_c_str("lambda_w2"),
                     <double*> lambda_w2.data, lambda_w2.size)
    if mu_w2 is not None:
        m.add_vector(to_c_str("mu_w2"),
                     <double*> mu_w2.data, mu_w2.size)

    cdef Data *d = new Data()
    _add_sparse_matrix("x", d, X)

    if y_train_pred_ is not None:
        d.add_vector(to_c_str("y_train_pred"),
                     <double*> y_train_pred_.data, n_samples)

    if cost is not None:
        d.add_vector(to_c_str("cost"),
                     <double*> cost.data, cost.size)
    if x_c_cost is not None:
        d.add_vector(to_c_str("x_c_cost"),
                     <double*> x_c_cost.data, x_c_cost.size)
    if x_i_cost is not None:
        d.add_vector(to_c_str("x_i_cost"),
                     <double*> x_i_cost.data, x_i_cost.size)

    if y is not None:
        d.add_vector(to_c_str("y_true"), &y[0], X.shape[0])

    if C is not None and I is not None:
        _add_sparse_matrix("x_c", d, C)
        _add_sparse_matrix("x_i", d, I)
        if settings['loss'] == 'bpr':
            assert X.shape[0] == C.shape[0]
            assert X.shape[1] == I.shape[1]
        if settings['loss'] == 'icd':
            assert X.shape[1] == C.shape[1] + I.shape[1]

    cpp_ffm.fit(s, m, d, fit_callback_wrapper, (<void*> callback))

    del d
    del m
    del s

    return w_0, w, V
