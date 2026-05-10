# Python调试与测试基础


## 🐛 Python 调试与测试基础


pdb 交互式调试器（断点/单步/变量查看）、doctest 文档测试、unittest 单元测试框架、TestCase 断言方法、测试发现与运行。


## pdb 调试器


```
// ========== pdb 基本使用 ==========
import pdb

def divide(a, b):
    result = a / b
    return result

# 方式 1: 设置断点
def calculate():
    x = 10
    y = 0
    pdb.set_trace()          # 在此处暂停,进入交互式调试
    result = divide(x, y)
    return result

# 方式 2: python -m pdb script.py
# 方式 3: breakpoint() (Python 3.7+)
# breakpoint()  # 等价于 pdb.set_trace(), 但可通过环境变量控制
```


```
// ========== pdb 常用命令 ==========
# 运行控制:
#   n (next)     — 执行下一行 (不进入函数)
#   s (step)     — 进入函数内部
#   c (continue) — 继续执行到下一个断点
#   r (return)   — 执行到当前函数返回
#   unt (until)  — 执行到指定行

# 断点:
#   b 行号        — 设置断点
#   b 函数名      — 在函数处断点
#   cl 断点号     — 清除断点
#   disable/enable 断点号

# 变量查看:
#   p 变量名      — 打印变量
#   pp 变量名     — 漂亮打印
#   l (list)      — 显示源代码
#   ll (longlist) — 显示当前函数全部源码
#   a (args)      — 打印当前函数参数

# 其他:
#   !语句         — 执行 Python 语句
#   h (help)      — 帮助
#   q (quit)      — 退出调试器

# 示例流程:
# > script.py(10)calculate()
# -> result = divide(x, y)
# (Pdb) p x           # 10
# (Pdb) p y           # 0
# (Pdb) s             # 进入 divide 函数
# > script.py(3)divide()
# -> result = a / b
# (Pdb) p a, b        # 10 0
# (Pdb) q             # 退出调试
```


## doctest 文档测试


```
// ========== doctest ==========
# 在文档字符串中写测试用例

def add(a, b):
    """返回 a 和 b 的和。

    >>> add(2, 3)
    5
    >>> add(-1, 1)
    0
    >>> add(0, 0)
    0
    """
    return a + b

def multiply(a, b):
    """返回 a 和 b 的乘积。

    >>> multiply(3, 4)
    12
    >>> multiply(0, 5)
    0
    >>> multiply(-2, 3)
    -6
    """
    return a * b

# 运行 doctest:
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)  # 显示所有测试结果
    # 无输出 = 全部通过

# 命令行运行:
# python -m doctest script.py -v
# python -m doctest README.md      # 也可以测试 .md 文件!

# 边缘情况:
def divide(a, b):
    """返回 a / b。

    >>> divide(10, 2)
    5.0
    >>> divide(5, 0)
    Traceback (most recent call last):
        ...
    ZeroDivisionError: division by zero
    """
    return a / b

# 注意: 浮点数用 doctest 需近似比较
# ELLIPSIS 指令 ... 可匹配任意内容
```


## unittest 基础


```
// ========== unittest TestCase ==========
import unittest

def add(a, b):
    return a + b

def is_even(n):
    return n % 2 == 0

class TestMath(unittest.TestCase):
    """测试数学函数"""

    # setUp: 每个测试前执行
    def setUp(self):
        print("setup...")
        self.test_data = [1, 2, 3, 4, 5]

    # tearDown: 每个测试后执行
    def tearDown(self):
        print("teardown...")

    # 测试方法必须以 test_ 开头
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)

    def test_add_float(self):
        self.assertAlmostEqual(add(0.1, 0.2), 0.3)
        # assertAlmostEqual 用于浮点数比较

    def test_is_even(self):
        self.assertTrue(is_even(4))
        self.assertFalse(is_even(5))
        self.assertTrue(is_even(0))

    def test_list_length(self):
        self.assertEqual(len(self.test_data), 5)

# 运行:
if __name__ == "__main__":
    unittest.main()

# 命令行运行:
# python -m unittest test_math.py
# python -m unittest test_math.TestMath
# python -m unittest test_math.TestMath.test_add
# python -m unittest discover  # 发现所有测试
```


## unittest 断言方法


```
// ========== 常用断言 ==========
import unittest

class TestAssertions(unittest.TestCase):

    def test_assertions(self):
        # 相等性:
        self.assertEqual(a, b)          # a == b
        self.assertNotEqual(a, b)       # a != b

        # 布尔值:
        self.assertTrue(x)              # bool(x) is True
        self.assertFalse(x)             # bool(x) is False

        # 比较:
        self.assertGreater(a, b)        # a > b
        self.assertGreaterEqual(a, b)   # a >= b
        self.assertLess(a, b)           # a < b
        self.assertLessEqual(a, b)      # a <= b

        # 容器:
        self.assertIn(item, container)  # item in container
        self.assertNotIn(item, container)
        self.assertIsNone(x)            # x is None
        self.assertIsNotNone(x)

        # 类型:
        self.assertIsInstance(obj, cls) # isinstance(obj, cls)
        self.assertNotIsInstance(obj, cls)

        # 浮点数:
        self.assertAlmostEqual(0.1+0.2, 0.3, places=5)

        # 异常:
        with self.assertRaises(ValueError):
            int("not_a_number")

        # 异常消息:
        with self.assertRaisesRegex(ValueError, "invalid literal"):
            int("xyz")

        # 日志:
        with self.assertLogs("mylogger", level="INFO") as log:
            logging.getLogger("mylogger").info("test")
            self.assertIn("test", log.output[0])
```


## unittest 进阶


```
// ========== setUpClass / tearDownClass ==========
import unittest

class TestDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """所有测试前执行一次 (创建数据库连接)"""
        cls.connection = create_connection()
        cls.connection.connect()

    @classmethod
    def tearDownClass(cls):
        """所有测试后执行一次"""
        cls.connection.close()

    def setUp(self):
        """每个测试前执行 (准备数据)"""
        self.connection.begin_transaction()

    def tearDown(self):
        """每个测试后执行 (清理数据)"""
        self.connection.rollback()

// ========== 跳过测试 ==========
import unittest

class TestSkip(unittest.TestCase):

    @unittest.skip("跳过此测试")
    def test_skip(self):
        pass

    @unittest.skipIf(sys.version_info < (3, 10), "需要 Python 3.10+")
    def test_new_feature(self):
        pass

    @unittest.skipUnless(sys.platform == "linux", "仅 Linux")
    def test_linux_only(self):
        pass

// ========== 子测试 ==========
class TestParametrize(unittest.TestCase):

    def test_even(self):
        for value, expected in [(2, True), (3, False), (0, True)]:
            with self.subTest(value=value):
                self.assertEqual(is_even(value), expected)
```


## 测试组织与运行


```
// ========== 项目测试结构 ==========
# project/
# ├── src/myproject/
# │   ├── __init__.py
# │   ├── calculator.py
# │   └── utils.py
# └── tests/
#     ├── __init__.py
#     ├── test_calculator.py
#     └── test_utils.py

// ========== test_calculator.py ==========
import unittest
from src.myproject.calculator import add, multiply

class TestCalculator(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_multiply(self):
        self.assertEqual(multiply(3, 4), 12)

# ========== 运行测试 ==========
# 运行全部:
# python -m unittest discover -s tests -v

# 运行单个文件:
# python -m unittest tests/test_calculator.py

# 运行单个测试类:
# python -m unittest tests.test_calculator.TestCalculator

# 运行单个测试方法:
# python -m unittest tests.test_calculator.TestCalculator.test_add

# 使用 pytest (更现代):
# pip install pytest
# pytest tests/ -v

// ========== 测试覆盖率 ==========
# 安装: pip install coverage
# 运行: coverage run -m unittest discover
# 报告: coverage report -m
# HTML: coverage html

# .coveragerc:
# [run]
# source = src
# omit = */tests/*
```


> **Note:** 💡 调试与测试要点: (1) breakpoint() 设置断点 (Python 3.7+); (2) pdb 命令: n/s/c/p/l/q; (3) doctest 写文档同时做测试; (4) unittest TestCase 断言方法 assertEqual/assertTrue/assertRaises; (5) python -m unittest discover 自动发现测试; (6) coverage 衡量测试覆盖率。


## 练习


<!-- Converted from: 83_Python调试与测试基础.html -->
