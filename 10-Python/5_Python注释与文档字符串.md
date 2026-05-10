# Python注释与文档字符串


## 💬 Python 注释与文档字符串


单行注释 #、多行注释、docstring 规范、__doc__ 属性、文档生成工具。


## 单行注释与多行注释


```
// ========== 单行注释 # ==========
// Python 用 # 表示单行注释
// # 之后的内容都被解释器忽略

# 这是单行注释
name = "Alice"  # 行尾注释（代码后空两格再写 #）

// ========== 多行注释 ==========
// Python 没有专门的多行注释语法
// 可用连续 # 或三引号字符串（不被赋值时）

# 方式一: 连续 #
# 这是多行
# 注释内容

// 方式二: 三引号（字符串未被使用，相当于注释）
"""
这是多行内容
用三引号包裹
解释器会创建字符串但不使用
"""
print("Hello")  # 上面 """ ... """ 不影响执行

// ========== 批量注释快捷键 ==========
// VSCode: Ctrl + /  (选中行后按)
// PyCharm: Ctrl + /
// 批量添加/取消 # 注释

// ========== 注释最佳实践 ==========
// 好注释解释 WHY，不是 WHAT
// 代码本身应该清晰到能自解释 WHAT

# 不好的注释 (解释 WHAT):
x = x + 1  # x 加 1

# 好的注释 (解释 WHY):
# 补偿时区差异，用户本地时间比 UTC 早 8 小时
offset = current_hour + 8

// 删除代码用删除，不要注释掉
// 注释掉的代码会腐烂，让人困惑
// 版本控制历史里有旧代码
```


## 文档字符串 Docstring


```
// ========== Docstring 是什么 ==========
// 模块、函数、类、方法的第一条语句
// 用三引号 """ 包裹
// 通过 obj.__doc__ 访问
// 被 Python 解释器识别并保留

// ========== 模块级 Docstring ==========
"""本模块提供用户认证相关功能。

包含用户注册、登录、令牌验证等函数。
依赖: hashlib, jwt, datetime
"""

// ========== 函数 Docstring ==========
def add(a, b):
    """返回两个数的和。

    Args:
        a: 第一个加数
        b: 第二个加数

    Returns:
        两数之和
    """
    return a + b

// ========== 类 Docstring ==========
class User:
    """表示系统用户。

    Attributes:
        name: 用户名
        email: 电子邮箱
        is_active: 账户是否激活
    """

    def __init__(self, name, email):
        """初始化 User 实例。

        Args:
            name: 用户名
            email: 电子邮箱
        """
        self.name = name
        self.email = email
        self.is_active = True

// ========== 访问 Docstring ==========
print(add.__doc__)     # "返回两个数的和。..."
print(User.__doc__)    # "表示系统用户。..."
help(add)              # 显示格式化的文档
help(User)             # 显示类及其方法的文档
```


> **Note:** 💡 三种常用 docstring 风格: (1) Google 风格 — 简洁直观，最流行; (2) reStructuredText — Sphinx 默认，适合大型项目; (3) NumPy 风格 — 数据科学领域常用。推荐 Google 风格，兼顾可读性和结构化。


## Docstring 风格对比


```
// ========== Google 风格 (推荐) ==========
def parse_data(text):
    """解析文本数据为字典列表。

    Args:
        text (str): 原始文本内容，每行一个记录。

    Returns:
        list[dict]: 解析后的字典列表。

    Raises:
        ValueError: 文本格式不正确时抛出。
    """
    pass

// ========== reStructuredText 风格 ==========
def parse_data(text):
    """解析文本数据为字典列表。

    :param text: 原始文本内容，每行一个记录。
    :type text: str
    :returns: 解析后的字典列表。
    :rtype: list[dict]
    :raises ValueError: 文本格式不正确。
    """
    pass

// ========== NumPy 风格 ==========
def parse_data(text):
    """解析文本数据为字典列表。

    Parameters
    ----------
    text : str
        原始文本内容，每行一个记录。

    Returns
    -------
    list[dict]
        解析后的字典列表。

    Raises
    ------
    ValueError
        文本格式不正确时抛出。
    """
    pass
```


## 实用工具与技巧


```
// ========== help() 函数 ==========
help(print)      # 查看 print 函数的文档
help(list)       # 查看列表的帮助文档
help(str.upper)  # 查看字符串 upper 方法的文档

// ========== __doc__ 属性 ==========
print(print.__doc__)     # 直接打印文档字符串
print(len.__doc__)       # "Return the number of items..."

// ========== 自动生成文档 ==========
// Sphinx: 最流行的 Python 文档生成器
// 从 docstring 生成 HTML/PDF 文档

pip install sphinx
sphinx-quickstart
sphinx-apidoc -o docs/ src/
make html

// ========== 类型提示与文档配合 ==========
// 类型提示让文档更精确
def calculate_total(prices: list[float], discount: float = 0) -> float:
    """计算折后总价。

    Args:
        prices: 商品价格列表
        discount: 折扣率 (0-1)，默认无折扣

    Returns:
        折后总金额
    """
    return sum(prices) * (1 - discount)

// ========== TODO 注释规范 ==========
# TODO: 后续需要添加缓存机制
# FIXME: 边界情况 n=0 时结果不正确
# HACK: 临时绕过第三方库的 bug，下个版本移除
# XXX: 这里逻辑复杂，建议重构
```


> **Note:** 💡 注释与 docstring 原则: (1) 注释解释 WHY，代码表达 WHAT; (2) 每个模块、公共函数、类都要有 docstring; (3) 保持 docstring 与代码同步更新; (4) 用 # TODO / # FIXME 标记待办事项; (5) 不要注释显而易见的内容。


## 练习


<!-- Converted from: 5_Python注释与文档字符串.html -->
