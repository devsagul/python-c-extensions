# Расширения для numpy на С

Для оценки стратегии в трейдинге среди прочих используется метрика recovery factor, которая определяется как отношение заработка к maximum draw down. Maximum draw down есть величина "максимальной просадки", т.е. максимальная разница между предыдущим максимумом временного ряда и текущим значением.

Необходимость реализации этой метрики для произвольного временного ряда навела меня на мысль о возможности написания собственного расширения для numpy и сравнения скорости выполнения различных подходов к решению данной задачи.

## Нативный Python

Прежде чем кидаться с головой в написание расширений на C, рассмотрим несколько решений, которые мы можем написать на Python.

### Python: наивный подход

Сначала определимся с тем, как же вообще вычисляется целевая метрика. Для этого напишем простую функцию в python, которая принимает итератор с индексацией (объект, подобный встроенному классу `list`; необходимы определения методов `__iter__` и `__getitem__`) чисел с плавающей точкой и возвращает число с плавающей точкой. Здесь и далее будем считать, что в качестве аргумента нам подается ряд, содержащий кумулятивную функцию заработка, т.е. если в ходе реализации нашей стратегии сделки приносили нам следующие величины: [1, 2, 5, -3, -7, 13], то на аргументом нашей функции должен стать ряд [1, 3, 8, 5, -2, 11]. Таким образом, последний член этого ряда и есть полный заработок нашей стратегии. Полученная функция:

```Python
def recovery_factor_naive(X):
    maxdrawdown = 0
    peak = float("-inf")
    for x in X:
        peak = max(peak, x)
        maxdrawdown = max(maxdrawdown, peak - x)
    return x / maxdrawdown
```

### Python: продвинутый подход

Следующая реализация этой функции на самом деле была самой первой, которую я написал. Признаюсь честно, сначала я был горд, что смог найти такое "изящное" решение. Это было связано с тем, что я не сразу понял, как быстро и эффективно посчитать это значение "в прямую" и пришел к необходимости редуцирования двух отображений. Сейчас мне кажется, что наивный подход значительно более удобный и читаемый, но у нас будет возможность проверить на практике, какой из них все-таки быстрее. Реализация более сложным методом в Python:

```Python
import itertools

def recovery_factor_advanced(X):
    maxdrawdown = max(
                      map(
                          lambda x: x[0] - x[1],
                          zip(
                              itertools.accumulate(X, max),
                              X
                          )
                      )
                  )
    return X[-1] / maxdrawdown
```

Здесь требуются некоторые пояснения. [itertools.accumulate](https://docs.python.org/3.3/library/itertools.html#itertools.accumulate) делает то же, что и reduce, но возвращает генератор со всеми промежуточными значениями свертки. Таким образом, применение его к исходному ряду и функции max даст нам ряд с текущими максимумами в исходном ряду. Т.е. получим отображение [1, 3, 8, 5, -2, 11] -> [1, 3, 8, 8, 8, 11].

zip не нуждается в представлении, после его применения просто получаем множество пар (текущий максимум, текущее значение). Теперь осталось лишь отобразить ряд данных пар на разность между первым и вторым значением в паре, свернуть при помощи функции max, и мы получим искомую "максимальную просадку". Повторюсь, на момент написания функции, это решение казалось мне удачным :)

### NumPy: нативный подход

Здесь все попроще: нужно просто взять аккумулятив функции np.maximum на исходном ряде и вычесть значения ряда, а затем взять максимум.

```Python
import numpy as np

def recovery_factor_numpy(X):
    maxdrawdown = np.max(np.maximum.accumulate(X) - X)
    return X[-1] / maxdrawdown
```

## Расширения для Python, написанные на C

### Hello, World!

До текущего дня я только лишь читал о написании расширений для Python в C, так что начать придется с малого. В целом, для написания расширения необходимо: написать функцию C, которую необходимо реализовать, написать для нее байндинг, который отвечает за получение аргументов из вызова в Python и приводить полученное значение для возврата из вызова, сформировать описание функций в модуле и описание самого модуля, описать функцию инициализации модуля и предоставить файл `setup.py` для установки полученного модуля. Начнем с функции, выводящей "Hello, World!"

Функция на C

```C
void Chello(void)
{
    printf("Hello, World!");
}
```

Байндинг (понятно, что на самом деле функцию Chello можно было не описывать, а просто поместить ее тело внутрь hello_world, но так продемострировать концепкцую байндинга в дальнейшем будет проще):

```C
static PyObject* hello_world(PyObject* self, PyObject* args)
{
    Chello();
    return Py_None;
}
```
Зарегистрируем методы нашего модуля; структура с нулями определяет конец массива.

```C
static PyMethodDef methods[] = {
    { "hello_world", hello_world, METH_NOARGS, "Prints Hello World!" },
    { NULL, NULL, 0, NULL }
};
```

Определим наш модуль:

```C
static struct PyModuleDef helloModule = {
    PyModuleDef_HEAD_INIT,
    "helloModule",
    "Module for greeting the World",
    -1,
    methods
};
```

Определим функцию инициализации модуля:

```C
PyMODINIT_FUNC PyInit_helloModule(void)
{
    return PyModule_Create(&helloModule);
}
```

Таким образом наш файл `hello.c` имеет вид:

```C
#include <stdio.h>
#include <Python.h>

void Chello(void)
{
    printf("Hello, World!\n");
}

static PyObject* hello_world(PyObject* self, PyObject* args)
{
    Chello();
    return Py_None;
}

static PyMethodDef methods[] = {
    { "hello_world", hello_world, METH_NOARGS, "Prints Hello World!" },
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
    return PyModule_Create(&helloModule);
}
```

Теперь `setup.py`

```Python
from distutils.core import setup, Extension
setup(name = 'helloModule', version = '1.0',  \
   ext_modules = [Extension('helloModule', ['hello.c'])])
```

Выполняем `python setup.py build` и `python setup.py install` для сборки модуля, запускаем python и убеждаемся в работе:

```Python
>>>import helloModule
>>>helloModule.hello_world()
Hello, World!
```

### Работа с аргументами

Написание функции, которая не принимает никаких аргументов, — не самое увлекательное занятие. Напишем функцию, которая вычисляет длину цепочки Коллатца (Collatz), которая определяется следующим образом:

Длина цепочки для числа 1 равна 0. Длина цепочки для любого четного числа равна 1 плюс длина цепочки для этого числа, деленного на 2. Длина цепочки для нечетного числа n, отличного от 1 равна 1 плюс длина цепочки от 3 * n + 1.

Сама функция (доопределим ее в 0, чтобы покрыть все множество неотрицательных целых):

```C
unsigned long long Ccollatz(unsinged long long)
{
    if ((n == 0) || (n == 1))
        return 0;
    if ((n % 2) == 0)
        return 1 + Ccollatz(n / 2);
    return 1 + Ccollatz(3 * n + 1);
}
```

Байндинг (K используется для указания на unsigned long long):

```C
static PyObject* collatz(PyObject* self, PyObject* args)
{
    unsigned long long n;

    if(!PyArg_ParseTuple(args, "K", &n))
        return NULL;
    return Py_BuildValue("K", Ccollatz(n));
}
```

Полный обновленный файл `hello.c` (с оптимизациями для вычисления длины цепочки):

```C
#include <stdio.h>
#include <Python.h>

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

static PyMethodDef methods[] = {
    { "hello_world", hello_world, METH_NOARGS, "Prints Hello World!" },
    { "collatz", collatz, METH_VARARGS, "Returns collatz chain length"},
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
    return PyModule_Create(&helloModule);
}
```

Убеждаемся в работе:

```Python
>>>import helloModule
>>>helloModule.collatz(27)
111
```

Все работает!

### Работа с numpy-массивами

Во-первых, для работы с массивами в функцию инициализации необхдимо добавить инструкцию `import_array();`

Здесь достаточно много деталей, о которых необходимо писать отдельный пост, так что на данный момент я просто оставлю пример функции, работающей с массивами типа double. Для получения деталей кури [документацию](https://docs.scipy.org/doc/numpy/reference/c-api.html).
В итоге получим `hello.c`:

```C
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

    maxdd = peak = cur = *arr;
    while (n > 0) {
        cur = *arr;
        peak = peak > cur ? peak : cur;
        dif = peak - cur;
        maxdd = maxdd > dif ? maxdd : dif;
        arr++;
        n--;
    }
    return cur / maxdd;
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
```

И `setup.py`, в котором я указал также флаг gcc для компиляции с оптимизациями третьего уровня:

```Python
import numpy as np
from distutils.core import setup, Extension

setup(name = 'helloModule', version = '1.0',  \
   include_dirs = [np.get_include()], \
   ext_modules = [Extension('helloModule', ['hello.c'],  extra_compile_args = ["-O3"])], \
  )

```

Проверяем:

```Python
>>> import numpy as np
>>> import helloModule as hm
>>> hm.maxdrawdown(np.array([1, 2, 0, 3]))
2
```

## Сравение подходов

Ок, теперь осталось протестировать все 4 подхода. Предположим, что у нас есть торговая стратегия, доход на каждом шаге которой есть случайная величина с нормальным распределением с параметрами \mu = 0.1, \sigma = 1.

Файл `benchmark.py`:

```Python
import time
import numpy as np
import helloModule as hm
import itertools
import math as m
import cProfile

def recovery_factor_naive(X):
    maxdrawdown = 0
    peak = float("-inf")
    for x in X:
        peak = max(peak, x)
        maxdrawdown = max(maxdrawdown, peak - x)
    return x / maxdrawdown

def recovery_factor_advanced(X):
    maxdrawdown = max(
                      map(
                          lambda x: x[0] - x[1],
                          zip(
                              itertools.accumulate(X, max),
                              X
                          )
                      )
                  )
    return X[-1] / maxdrawdown

def recovery_factor_numpy(X):
    maxdrawdown = np.max(np.maximum.accumulate(X) - X)
    return X[-1] / maxdrawdown

def recovery_factor_c(X):
    return X[-1] / hm.maxdrawdown(X)

if __name__ == "__main__":
    cProfile.run("""
for _ in range(10 ** 5):
    X = np.cumsum(np.random.normal(loc=0.01, size=10 ** 3))
    a = recovery_factor_naive(X.copy())
    b = recovery_factor_advanced(X.copy())
    c = recovery_factor_numpy(X.copy())
    d = recovery_factor_c(X.copy())
    assert m.isclose(a, b, abs_tol=0.0001)
    assert m.isclose(a, c, abs_tol=0.0001)
    assert m.isclose(a, d, abs_tol=0.0001)
    """)
```

Я получил следующие результаты (кумулятивное время на каждый из вызовов в секундах):

`recovery_factor_naive`: 79.035

`recovery_factor_advanced`: 70.731

`recovery_factor_numpy`: 2.944

`recovery_factor_c`: 0.475

Если мы за единицу возьмем время выполнения `recovery_factor_numpy`, как наиболее целесообразного подхода (с точки зрения скорости написания, гибкоски расширения, проверки условий и т.д.), то время выполнения `recovery_factor_naive` составит 26.846 единиц, `recovery_factor_advanced` — 24.025 единиц, а `recovery_factor_c` — 0.161.

## Выводы

Стоит отметить четыре пункта:

1. Реализация функции на C не обложена в полной мере всеми необходимыми проверками и не возбуждает необходимые исключения в случае нарушения условий использования.

2. Сама функция оказалась достаточно простой, хотя изначально я предполагал ее реализацию несколько более трудоемкой. Конечно, написание собственного расширения — слишком трудоемкий процесс для такой простой функции.

3. В данном посте я не адресовал некоторые другие способы построения интерфейсов к функциям, написанным на C \(например, [SWIG](http://swig.org/tutorial.html) и [Cython](https://cython.org/)\).

4. Я не углублялся в особенности оптимизации компиляции как python и numpy, так и расширения на C, что может сильно повлиять на производительность.

Так или иначе, реализация расширения для Python в целом и numpy в частности представляет большой интерес с точки зрения повышения производительности программ.