# 20 - C与Python互操作

## 概述

Python可以通过多种方式调用C代码，用于性能优化或复用现有C库。本节介绍ctypes、cffi和Python C扩展三种方法。

---

## 1. ctypes（最简单的方式）

### C库代码

```c
// mylib.c
#include <math.h>

// 简单函数
int add(int a, int b) {
    return a + b;
}

double fast_sqrt(double x) {
    return sqrt(x);
}

// 数组操作
void array_multiply(double *arr, int n, double factor) {
    for (int i = 0; i < n; i++) {
        arr[i] *= factor;
    }
}

// 结构体
typedef struct {
    double x;
    double y;
} Point;

double point_distance(Point *a, Point *b) {
    double dx = a->x - b->x;
    double dy = a->y - b->y;
    return sqrt(dx * dx + dy * dy);
}

// 字符串
const char* get_greeting(const char *name) {
    static char buf[256];
    snprintf(buf, sizeof(buf), "你好, %s!", name);
    return buf;
}
```

### 编译共享库

```bash
# Linux
gcc -shared -fPIC -o mylib.so mylib.c -lm

# macOS
gcc -shared -fPIC -o mylib.dylib mylib.c

# Windows
gcc -shared -o mylib.dll mylib.c -lm
```

### Python调用

```python
import ctypes
import os
import platform

# 加载库
if platform.system() == 'Windows':
    lib = ctypes.CDLL('./mylib.dll')
else:
    lib = ctypes.CDLL('./mylib.so')

# ========== 基本类型 ==========

# 设置参数和返回类型
lib.add.argtypes = [ctypes.c_int, ctypes.c_int]
lib.add.restype = ctypes.c_int

result = lib.add(3, 4)
print(f"add(3, 4) = {result}")

# 浮点数
lib.fast_sqrt.argtypes = [ctypes.c_double]
lib.fast_sqrt.restype = ctypes.c_double
print(f"sqrt(2) = {lib.fast_sqrt(2.0)}")

# ========== 数组 ==========

import numpy as np

# numpy数组传递
arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
lib.array_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_int,
    ctypes.c_double
]
lib.array_multiply.restype = None

lib.array_multiply(arr, len(arr), 2.5)
print(f"数组结果: {arr}")

# ========== 结构体 ==========

class Point(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_double),
        ('y', ctypes.c_double),
    ]

lib.point_distance.argtypes = [ctypes.POINTER(Point), ctypes.POINTER(Point)]
lib.point_distance.restype = ctypes.c_double

p1 = Point(0.0, 0.0)
p2 = Point(3.0, 4.0)
dist = lib.point_distance(ctypes.byref(p1), ctypes.byref(p2))
print(f"距离: {dist}")

# ========== 字符串 ==========

lib.get_greeting.argtypes = [ctypes.c_char_p]
lib.get_greeting.restype = ctypes.c_char_p

msg = lib.get_greeting(b"Python")
print(msg.decode('utf-8'))
```

---

## 2. cffi（更灵活的方式）

### C代码

```c
// math_ops.c
#include <stdlib.h>
#include <math.h>

typedef struct {
    double *data;
    int size;
} Array;

Array* array_new(int size) {
    Array *arr = (Array *)malloc(sizeof(Array));
    arr->data = (double *)calloc(size, sizeof(double));
    arr->size = size;
    return arr;
}

void array_free(Array *arr) {
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

double array_sum(Array *arr) {
    double sum = 0;
    for (int i = 0; i < arr->size; i++)
        sum += arr->data[i];
    return sum;
}

void array_set(Array *arr, int index, double value) {
    if (index >= 0 && index < arr->size)
        arr->data[index] = value;
}
```

### Python使用cffi

```python
from cffi import FFI
import os

ffi = FFI()

# 声明C接口
ffi.cdef("""
    typedef struct {
        double *data;
        int size;
    } Array;

    Array* array_new(int size);
    void array_free(Array *arr);
    double array_sum(Array *arr);
    void array_set(Array *arr, int index, double value);
""")

# 编译并加载
# 方式1: 预编译的库
C = ffi.dlopen('./math_ops.so')

# 方式2: 在线编译
# C = ffi.verify("""
#     #include "math_ops.c"
# """, libraries=['m'])

# 使用
arr = C.array_new(10)
for i in range(10):
    C.array_set(arr, i, float(i + 1))

print(f"求和: {C.array_sum(arr)}")  # 55.0
C.array_free(arr)
```

---

## 3. Python C扩展（最高性能）

### 扩展模块代码

```c
// mymodule.c
#include <Python.h>
#include <math.h>

// 函数：计算数组均值
static PyObject* py_mean(PyObject *self, PyObject *args) {
    PyObject *list_obj;

    if (!PyArg_ParseTuple(args, "O", &list_obj))
        return NULL;

    if (!PyList_Check(list_obj)) {
        PyErr_SetString(PyExc_TypeError, "需要列表参数");
        return NULL;
    }

    Py_ssize_t n = PyList_Size(list_obj);
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "列表不能为空");
        return NULL;
    }

    double sum = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyList_GetItem(list_obj, i);
        sum += PyFloat_AsDouble(item);
    }

    return PyFloat_FromDouble(sum / n);
}

// 函数：快速阶乘
static PyObject* py_factorial(PyObject *self, PyObject *args) {
    int n;
    if (!PyArg_ParseTuple(args, "i", &n))
        return NULL;

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "负数无阶乘");
        return NULL;
    }

    long long result = 1;
    for (int i = 2; i <= n; i++)
        result *= i;

    return PyLong_FromLongLong(result);
}

// 方法表
static PyMethodDef MyMethods[] = {
    {"mean", py_mean, METH_VARARGS, "计算列表均值"},
    {"factorial", py_factorial, METH_VARARGS, "计算阶乘"},
    {NULL, NULL, 0, NULL}
};

// 模块定义
static struct PyModuleDef mymodule = {
    PyModuleDef_HEAD_INIT,
    "mymodule",    // 模块名
    NULL,          // 文档
    -1,            // 每个解释器的状态大小
    MyMethods
};

// 模块初始化
PyMODINIT_FUNC PyInit_mymodule(void) {
    return PyModule_Create(&mymodule);
}
```

### 构建脚本

```python
# setup.py
from setuptools import setup, Extension

module = Extension(
    'mymodule',
    sources=['mymodule.c'],
    libraries=['m'],  # 链接数学库
)

setup(
    name='mymodule',
    version='1.0',
    ext_modules=[module],
)
```

```bash
# 编译
python setup.py build_ext --inplace

# 使用
python -c "import mymodule; print(mymodule.mean([1,2,3,4,5]))"
```

---

## 4. 方法对比

| 方法 | 性能 | 易用性 | 依赖 | 适用场景 |
|------|------|--------|------|----------|
| ctypes | 中 | 最简单 | 无 | 快速调用已有C库 |
| cffi | 中 | 简单 | cffi包 | 需要更灵活的接口 |
| C扩展 | 最高 | 复杂 | Python头文件 | 性能关键模块 |

---

## 5. 性能对比

```python
import time
import ctypes
import mymodule  # C扩展

# Python实现
def py_factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# ctypes
lib = ctypes.CDLL('./mymodule.so')
lib.factorial.argtypes = [ctypes.c_int]
lib.factorial.restype = ctypes.c_longlong

n = 20
iterations = 100000

# 测试
t = time.time()
for _ in range(iterations):
    py_factorial(n)
print(f"Python: {time.time()-t:.3f}s")

t = time.time()
for _ in range(iterations):
    lib.factorial(n)
print(f"ctypes: {time.time()-t:.3f}s")

t = time.time()
for _ in range(iterations):
    mymodule.factorial(n)
print(f"C扩展:  {time.time()-t:.3f}s")
```

---

## 要点总结

1. ctypes最简单，适合快速调用已有C库
2. cffi提供更灵活的接口定义
3. C扩展性能最高，但开发复杂度也最高
4. 使用numpy+ctypes可以高效传递数组数据
5. C扩展中必须正确处理Python引用计数和异常
6. 生产环境推荐使用cython或pybind11简化C扩展开发
