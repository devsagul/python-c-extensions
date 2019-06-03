#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdio.h>
#include <Python.h>
#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>

void Chello(void)
{
    printf("Hello, World!");
}

static PyObject* hello_world(PyObject* self, PyObject* args)
{
    Chello();
    return Py_None;
}

unsigned long long Ccollatz(unsigned long long n)
{
    if (!n || (n == 1))
        return 0;
    if (!(n & 1))
        return 1 + Ccollatz(n >> 1);
    return 2 + Ccollatz(((n << 1) + n + 1) >> 1);
}

static PyObject* collatz(PyObject* self, PyObject* args)
{
    unsigned long long n;

    if(!PyArg_ParseTuple(args, "K", &n))
        return NULL;
    return Py_BuildValue("K", Ccollatz(n));
}

double Cdrawdown(double *arr, long int n)
{
    double maxdd;
    double peak;
    double cur;
    double dif;

    maxdd = peak = *arr;
    while (n > 0) {
        cur = *arr;
        peak = peak > cur ? peak : cur;
        dif = peak - cur;
        maxdd = maxdd > dif ? maxdd : dif;
        arr++;
        n--;
    }
    return maxdd;
}

static PyObject* drawdown(PyObject* self, PyObject* args)
{
    PyObject *input;
    PyArrayObject *arr;
    double *dptr;
    long int *dims;
    long int nd;

    if(!PyArg_ParseTuple(args, "O", &input))
        return NULL;

    arr = PyArray_FROM_OTF(input, NPY_DOUBLE, 0);
    if (arr == NULL)
        return NULL;

    nd = PyArray_NDIM(arr);
    dims = PyArray_DIMS(arr);
    if ((nd != 1) || (dims[0] < 1)) {
        Py_XDECREF(arr);
        return NULL;
    }
    dptr = (double *)PyArray_DATA(arr);

    Py_DECREF(arr);
    return Py_BuildValue("d", Cdrawdown(dptr, dims[0]));
}

static PyMethodDef methods[] = {
    { "hello_world", hello_world, METH_NOARGS, "Prints Hello World!" },
    { "collatz", collatz, METH_VARARGS, "Returns collatz chain length"},
    { "maxdrawdown", drawdown, METH_VARARGS,
        "Returns the maximum draw down of the given time series" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef helloModule = {
    PyModuleDef_HEAD_INIT,
    "helloModule",
    "Module for greeting the World",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_helloModule(void)
{
    import_array();
    return PyModule_Create(&helloModule);
}