# MATLAB 与 Python 混合编程

## 一、从 MATLAB 调用 Python

### 1.1 基本调用

```matlab
% MATLAB R2014b+ 内置支持调用 Python
% 检查 Python 环境
pe = pyenv;
fprintf('Python 版本: %s\n', char(pe.Version));
fprintf('Python 路径: %s\n', char(pe.Executable));

% 指定 Python 版本（需要重启 MATLAB）
% pyenv('Version', '3.10');
% pyenv('Version', 'C:\Python310\python.exe');

% 调用 Python 模块
os = py.importlib.import_module('os');
cwd = os.getcwd();
fprintf('当前目录: %s\n', char(cwd));

% 列出目录
files = os.listdir('.');
fprintf('文件列表:\n');
for i = 1:length(files)
    fprintf('  %s\n', char(files{i}));
end
```

### 1.2 调用 Python 函数

```matlab
% 数学运算
result = py.math.sqrt(2);
fprintf('sqrt(2) = %s\n', char(result));

result = py.math.sin(py.math.pi / 6);
fprintf('sin(π/6) = %s\n', char(result));

% 字符串操作
s = py.str('Hello, MATLAB!');
upper_s = s.upper();
fprintf('大写: %s\n', char(upper_s));

% 列表操作
py_list = py.list({1, 2, 3, 4, 5});
py_list.append(6);
fprintf('Python 列表: %s\n', char(py_list));

% 字典
py_dict = py.dict(pyargs('name', 'MATLAB', 'version', 'R2024b'));
fprintf('字典: %s\n', char(py_dict));

% numpy 调用
np = py.importlib.import_module('numpy');
arr = np.array({{1, 2, 3}, {4, 5, 6}});
fprintf('numpy 数组: %s\n', char(arr));
fprintf('形状: %s\n', char(arr.shape));
fprintf('均值: %s\n', char(np.mean(arr)));
```

### 1.3 类型转换

```matlab
% Python -> MATLAB
py_list = py.list({1, 2, 3, 4, 5});
matlab_array = double(py_list);      % Python list -> MATLAB double

py_int = py.int(42);
matlab_val = double(py_int);          % Python int -> MATLAB double

py_str = py.str('hello');
matlab_str = char(py_str);            % Python str -> MATLAB char

% MATLAB -> Python
matlab_matrix = [1 2 3; 4 5 6];
py_array = py.numpy.array(matlab_matrix);  % MATLAB matrix -> numpy array

matlab_list = {1, 'two', 3.0};
py_list = py.list(matlab_list);        % MATLAB cell -> Python list

matlab_dict = containers.Map({'a', 'b'}, {1, 2});
% MATLAB Map -> Python dict (需要手动转换)
py_dict = py.dict(pyargs('a', 1, 'b', 2));

% 处理 Python tuple
py_tuple = py.tuple({1, 2, 3});
matlab_cell = cell(py_tuple);          % Python tuple -> MATLAB cell

% 处理 numpy 数组
np = py.importlib.import_module('numpy');
py_arr = np.array({{1.5, 2.5, 3.5}});
matlab_arr = double(py_arr);           % numpy array -> MATLAB double
```

### 1.4 调用 Python 类

```matlab
% 创建 Python 类的实例
% 假设有一个 Python 模块 mymodule.py:
% class Calculator:
%     def __init__(self, value):
%         self.value = value
%     def add(self, x):
%         return self.value + x

% mymod = py.importlib.import_module('mymodule');
% calc = mymod.Calculator(10);
% result = calc.add(5);
% fprintf('10 + 5 = %d\n', int64(result));

% 使用 Python 的 collections 模块
collections = py.importlib.import_module('collections');
counter = collections.Counter(py.list({'a', 'b', 'a', 'c', 'b', 'a'}));
fprintf('Counter: %s\n', char(counter));

% 使用 Python 的 itertools
itertools = py.importlib.import_module('itertools');
combs = py.list(itertools.combinations(py.list({1, 2, 3, 4}), 2));
fprintf('组合: %s\n', char(combs));
```

---

## 二、实用 Python 库集成

### 2.1 NumPy 集成

```matlab
np = py.importlib.import_module('numpy');

% 创建数组
a = np.array({{1, 2, 3}, {4, 5, 6}});
fprintf('数组: %s\n', char(a));
fprintf('形状: %s\n', char(a.shape));

% 数组操作
b = np.zeros(py.tuple({3, 4}));
c = np.ones(py.tuple({2, 3}));
d = np.random.rand(py.tuple({3, 3}));

% 线性代数
A_np = np.array({{1, 2}, {3, 4}});
B_np = np.array({{5, 6}, {7, 8}});
C_np = np.matmul(A_np, B_np);
fprintf('矩阵乘积: %s\n', char(C_np));

% 特征值
eigvals = np.linalg.eigvals(A_np);
fprintf('特征值: %s\n', char(eigvals));

% 返回到 MATLAB 验证
A_matlab = double(A_np);
B_matlab = double(B_np);
C_matlab = A_matlab * B_matlab;
fprintf('MATLAB 验证:\n');
disp(C_matlab);
```

### 2.2 Pandas 集成

```matlab
pd = py.importlib.import_module('pandas');

% 创建 DataFrame
data = py.dict(pyargs( ...
    'Name', py.list({'Alice', 'Bob', 'Charlie', 'Diana'}), ...
    'Age', py.list({25, 30, 35, 28}), ...
    'Score', py.list({85.5, 92.3, 78.9, 95.1})));
df = pd.DataFrame(data);
fprintf('DataFrame:\n');
disp(df);

% 读取 CSV（通过 pandas）
% df_csv = pd.read_csv('data.csv');
% summary = df_csv.describe();
% fprintf('统计摘要:\n');
disp(df.describe());

% 数据筛选
% filtered = df.loc(df['Age'] > py.int(28));

% 转为 MATLAB table
% names = cell(df{'Name'});
% ages = double(df{'Age'});
% T = table(names', ages', 'VariableNames', {'Name', 'Age'});
```

### 2.3 Matplotlib 集成

```matlab
plt = py.importlib.import_module('matplotlib.pyplot');
np = py.importlib.import_module('numpy');

% 生成数据
x = np.linspace(0, 2*np.pi, 100);
y = np.sin(x);

% 绘图
plt.figure();
plt.plot(x, y, 'b-', 'LineWidth', py.int(2));
plt.xlabel('x');
plt.ylabel('sin(x)');
plt.title('Python Matplotlib from MATLAB');
plt.grid(py.bool_(true));
plt.show();

% 也可以使用 MATLAB 原生绘图功能
x_matlab = double(x);
y_matlab = double(y);
figure;
plot(x_matlab, y_matlab, 'r-', 'LineWidth', 2);
title('MATLAB 原生绘图（数据来自 Python）');
```

---

## 三、MATLAB Engine for Python

### 3.1 安装与启动

```python
# Python 端调用 MATLAB
# 首先安装 MATLAB Engine
# cd "matlabroot/extern/engines/python"
# python setup.py install

import matlab.engine

# 启动 MATLAB 引擎
eng = matlab.engine.start_matlab()

# 或后台启动
eng = matlab.engine.start_matlab('-desktop')  # 启动桌面
eng = matlab.engine.start_matlab('-nodesktop')  # 无桌面模式
```

### 3.2 基本操作

```python
import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()

# 调用 MATLAB 函数
result = eng.sqrt(2.0)
print(f"sqrt(2) = {result}")

# 创建 MATLAB 数组
arr = matlab.double([[1, 2, 3], [4, 5, 6]])
print(f"数组: {arr}")

# 调用 MATLAB 内置函数
result = eng.sin(matlab.double([0, 3.14159/6, 3.14159/2]))
print(f"sin([0, π/6, π/2]) = {result}")

# MATLAB 矩阵运算
A = matlab.double([[1, 2], [3, 4]])
B = matlab.double([[5, 6], [7, 8]])
C = eng.mtimes(A, B)  # 矩阵乘法
print(f"A * B = {C}")

# 调用 MATLAB 脚本
# eng.my_script(nargout=0)

# 调用自定义函数
# result = eng.my_function(1.0, 2.0, nargout=1)

# 设置/获取工作区变量
eng.workspace['x'] = 10.0
eng.workspace['y'] = 20.0
result = eng.eval('x + y')
print(f"x + y = {result}")

eng.quit()
```

### 3.3 NumPy 与 MATLAB 数组互转

```python
import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()

# NumPy -> MATLAB
np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
mat_array = matlab.double(np_array.tolist())
result = eng.det(mat_array)  # 行列式

# MATLAB -> NumPy
mat_result = eng.magic(3)
np_result = np.array(mat_result)
print(f"Magic(3):\n{np_result}")

# 处理复数
np_complex = np.array([[1+2j, 3+4j]])
mat_complex = matlab.double(np_complex.tolist(), is_complex=True)
eng.workspace['z'] = mat_complex
abs_z = eng.eval('abs(z)')
print(f"|z| = {np.array(abs_z)}")

eng.quit()
```

### 3.4 异步调用

```python
import matlab.engine

eng = matlab.engine.start_matlab()

# 异步调用
future = eng.sqrt(42.0, background=True)

# 做其他事情...
print("正在计算...")

# 获取结果（阻塞）
result = future.result()
print(f"结果: {result}")

# 取消异步任务
# future.cancel()

eng.quit()
```

---

## 四、进程间通信方案

### 4.1 通过文件交换数据

```matlab
% MATLAB 端保存数据
data = rand(100, 5);
save('shared_data.mat', 'data');

% Python 端读取
% import scipy.io
% mat = scipy.io.loadmat('shared_data.mat')
% data = mat['data']
```

### 4.2 TCP/IP 通信

```matlab
% MATLAB 作为 TCP 服务器
t = tcpserver('localhost', 3000);
configureCallback(t, 'byte', 10, @(src, evt) readData(src));

function readData(src)
    data = read(src, src.NumBytesAvailable, 'double');
    result = sum(data);
    write(src, result, 'double');
end

% Python 客户端
% import socket, struct
% sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
% sock.connect(('localhost', 3000))
% sock.sendall(struct.pack('5d', 1, 2, 3, 4, 5))
% result = struct.unpack('d', sock.recv(8))[0]
```

---

## 五、调用系统命令

```matlab
% 执行系统命令
[status, result] = system('python --version');
fprintf('Python 版本: %s\n', result);

% 运行 Python 脚本
[status, result] = system('python my_script.py arg1 arg2');
if status == 0
    disp(result);
else
    fprintf('执行失败: %s\n', result);
end

% 使用 ! 操作符
!python my_script.py

% 使用 pyrunfile（R2022a+）
pyrunfile('my_script.py');

% 传递变量给 Python
x = 42;
result = pyrun("y = x ** 2; result = str(y)", "result", x=x);
fprintf('结果: %s\n', char(result));
```
