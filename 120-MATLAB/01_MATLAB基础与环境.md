# MATLAB 基础与环境

## 一、MATLAB 简介

MATLAB（Matrix Laboratory）是由 MathWorks 公司开发的高性能数值计算和可视化软件，广泛应用于工程计算、控制系统设计、信号处理、图像处理、金融分析等领域。

### 1.1 核心特点

- **矩阵为基本数据类型**：MATLAB 中所有数据都以矩阵形式存储，标量是 1×1 矩阵，向量是 1×n 或 n×1 矩阵
- **交互式环境**：支持命令行即时计算，也可编写脚本和函数
- **丰富的工具箱**：Signal Processing、Image Processing、Optimization、Deep Learning 等
- **强大的可视化能力**：二维、三维图形绘制功能完善
- **与外部语言接口**：支持调用 C/C++、Fortran、Python、Java 代码

### 1.2 版本演进

| 版本 | 年份 | 重要特性 |
|------|------|----------|
| R2016a | 2016 | 实时脚本（Live Script） |
| R2019b | 2019 | 参数/返回值声明 |
| R2021a | 2021 | MATLAB Online 改进 |
| R2023a | 2023 | 字符串增强、新图表 |
| R2024b | 2024 | Python 集成改进 |

---

## 二、MATLAB 桌面环境

### 2.1 主要窗口组件

启动 MATLAB 后，默认界面包含以下核心组件：

| 组件 | 位置 | 功能 |
|------|------|------|
| 命令窗口（Command Window） | 中央下方 | 执行命令、显示输出 |
| 工作区（Workspace） | 右侧面板 | 查看当前变量及其值 |
| 当前文件夹（Current Folder） | 左侧面板 | 文件浏览器 |
| 命令历史（Command History） | 左侧面板 | 记录已执行的命令 |
| 编辑器（Editor） | 独立窗口/标签 | 编写脚本和函数 |
| 路径管理器（Path Manager） | 对话框 | 管理搜索路径 |

### 2.2 自定义布局

```matlab
% 通过菜单：Home > Layout 自定义窗口布局
% 也可通过命令重置布局
% desktop = com.mathworks.mlservices.MatlabDesktopServices.getDesktop;
% desktop.restoreLayout('Default');
```

常用布局预设：
- **Default**：标准布局
- **Two Column**：双栏布局
- **All Tabbed**：所有面板以标签形式显示

### 2.3 系统预设

```matlab
% 打开预设对话框
% 菜单：Home > Preferences

% 常用预设项：
% - Fonts: 修改字体和字号
% - Colors: 修改语法高亮颜色
% - Keyboard Shortcuts: 自定义快捷键
% - MATLAB > General > Source Control: 版本控制集成
```

---

## 三、命令窗口与基本操作

### 3.1 命令执行

```matlab
% 基本算术运算
2 + 3           % 加法，ans = 5
10 - 4          % 减法，ans = 6
3 * 7           % 乘法，ans = 21
15 / 4          % 除法，ans = 3.7500
2 ^ 10          % 幂运算，ans = 1024

% 分号抑制输出
x = 10;         % 不显示结果
y = 20          % 显示结果：y = 20

% 多条命令同行，用逗号或分号分隔
a = 1, b = 2; c = 3

% 续行符 ...
result = 1 + 2 + 3 + ...
         4 + 5 + 6;
```

### 3.2 常用命令

```matlab
% 信息查询
clc             % 清除命令窗口显示
clear           % 清除工作区所有变量
clear x y       % 清除指定变量
close all       % 关闭所有图形窗口
clf             % 清除当前图形

% 帮助系统
help plot       % 查看函数简要帮助
doc plot        % 在帮助浏览器中打开详细文档
lookfor eig     % 搜索与关键字相关的函数
type eig        % 查看函数源代码（内置函数不可用）
which eig       % 显示函数所在路径
```

### 3.3 输出格式控制

```matlab
% format 控制数值显示格式
format short    % 短格式（默认），4位小数
pi              % ans = 3.1416
format long     % 长格式，15位小数
pi              % ans = 3.141592653589793
format shortE   % 科学计数法，短格式
12345           % ans = 1.2345e+04
format longE    % 科学计数法，长格式
format bank     % 银行格式，2位小数
1234.5678       % ans = 1234.57
format rat      % 有理数近似
pi              % ans = 355/113
format compact  % 紧凑输出（减少空行）
format loose     % 宽松输出
```

---

## 四、变量与数据类型

### 4.1 变量命名规则

```matlab
% 合法变量名
myVar = 1;
data_2024 = 100;
MAX_VALUE = 255;

% 不合法的变量名（会报错）
% 2data = 1;       % 不能以数字开头
% my-var = 1;      % 不能包含连字符
% my var = 1;      % 不能包含空格
% for = 1;         % 不能使用保留字

% 检查变量名是否合法
isvarname('myVar')      % 返回 1（合法）
isvarname('2data')      % 返回 0（不合法）

% MATLAB 区分大小写
A = 1;
a = 2;   % A 和 a 是不同的变量
```

### 4.2 基本数据类型

```matlab
% ==================== 数值类型 ====================
% double（默认类型，双精度浮点数）
x = 3.14;
class(x)        % 'double'
whos x          % 显示变量详细信息

% single（单精度浮点数）
y = single(3.14);
class(y)        % 'single'

% 整数类型
a = int8(100);      % 8位有符号整数，范围 -128~127
b = int16(1000);    % 16位有符号整数
c = int32(100000);  % 32位有符号整数
d = int64(1e12);    % 64位有符号整数

% 无符号整数
e = uint8(255);     % 8位无符号整数，范围 0~255
f = uint16(65535);  % 16位无符号整数

% 类型转换
val = double(int8(50));  % int8 -> double
val = int32(3.9);        % 截断为 3，不四舍五入

% ==================== 字符与字符串 ====================
% 字符数组（char）
name = 'MATLAB';
class(name)     % 'char'
length(name)    % 6

% 字符串标量（string，R2016b+）
str = "Hello, World!";
class(str)      % 'string'

% 字符串操作
fullfile('home', 'user', 'data')    % 路径拼接：'home/user/data'
contains('MATLAB', 'LAB')           % true
startsWith('Hello', 'He')           % true
endsWith('file.txt', '.txt')        % true
strrep('Hello World', 'World', 'MATLAB')  % 'Hello MATLAB'
split('a,b,c', ',')                 % 字符串分割

% ==================== 逻辑类型 ====================
flag = true;
class(flag)     % 'logical'
a = [1 2 3];
mask = a > 1;   % [0 1 1]，逻辑数组

% ==================== 其他类型 ====================
% 复数
z = 3 + 4i;
real(z)         % 实部：3
imag(z)         % 虚部：4
abs(z)          % 模：5
angle(z)        % 相角

% 日期时间
t = datetime('now');
datestr(now)    % 日期字符串
datevec(now)    % 日期向量
```

### 4.3 特殊常量

```matlab
pi              % 圆周率 π = 3.14159...
i, j            % 虚数单位 √(-1)
inf             % 正无穷
-inf            % 负无穷
NaN             % 非数（Not a Number）
eps             % 机器精度（约 2.22e-16）
realmax         % 最大正浮点数
realmin         % 最小正浮点数
```

---

## 五、工作区管理

### 5.1 变量查看与操作

```matlab
% 查看变量
who             % 列出变量名
whos            % 列出变量详细信息（大小、类型、字节数）

% 示例输出
% Name      Size            Bytes  Class     Attributes
% x         1x1                 8  double
% A         3x3                72  double

% 变量的保存与加载
save mydata.mat         % 保存所有变量到 .mat 文件
save mydata.mat x y     % 只保存指定变量
save mydata.txt x -ascii % 以文本格式保存

load mydata.mat          % 加载 .mat 文件中的所有变量
load mydata.mat x        % 只加载指定变量
S = load('mydata.mat');  % 加载为结构体

% 清除变量
clear                   % 清除所有变量
clear x y               % 清除指定变量
clearvars -except x     % 保留 x，清除其余变量
```

### 5.2 路径管理

```matlab
% 查看路径
pwd                 % 显示当前工作目录
cd                  % 切换目录
ls / dir            % 列出目录内容

% 添加路径
addpath('my_folder')            % 添加文件夹到搜索路径
addpath('my_folder', '-begin')  % 添加到路径开头（优先）
addpath('my_folder', '-end')    % 添加到路径结尾

% 管理路径
pathtool            % 打开路径管理器 GUI
restoredefaultpath  % 恢复默认路径

% 创建和删除文件夹
mkdir my_folder     % 创建文件夹
rmdir my_folder     % 删除空文件夹
delete file.txt     % 删除文件
```

### 5.3 Live Script

Live Script 是 MATLAB R2016a 引入的交互式文档格式，将代码、输出、格式化文本和图像整合在同一界面中。

```matlab
% 创建 Live Script
% 菜单：Home > New > Live Script
% 保存为 .mlx 格式

% Live Script 主要功能：
% 1. 代码分段执行（分节，用 %% 创建）
% 2. 实时显示输出和图形
% 3. 内嵌富文本、公式（LaTeX）、图像
% 4. 支持交互式控件（滑块、下拉菜单）
```

分节示例（在 Live Script 中）：
```matlab
%% 第一节：数据生成
x = linspace(0, 2*pi, 100);
y = sin(x);

%% 第二节：绘图
figure;
plot(x, y, 'b-', 'LineWidth', 2);
xlabel('x');
ylabel('sin(x)');
title('正弦函数');
```

---

## 六、常用技巧与最佳实践

### 6.1 性能优化

```matlab
% 预分配数组（避免动态扩展）
n = 10000;
tic;
for i = 1:n
    A(i) = i^2;   % 动态扩展，慢
end
toc

tic;
A = zeros(1, n);   % 预分配
for i = 1:n
    A(i) = i^2;
end
toc

% 向量化代替循环
tic;
x = 1:1000000;
y = x.^2;          % 向量化操作，快
toc
```

### 6.2 调试技巧

```matlab
% 设置断点
% 在编辑器中点击行号左侧的横线，或使用：
dbstop in myFunction at 25    % 在 myFunction 第25行设置断点
dbstop if error               % 出错时自动进入调试模式

% 调试命令
dbcont    % 继续执行
dbstep    % 单步执行
dbstep in % 进入函数
dbquit    % 退出调试模式
dbstack   % 查看函数调用栈
dbup      % 切换到上层工作区
dbdown    % 切换到下层工作区

% 条件断点
% 右键断点 > Set/Modify Condition
% 输入条件表达式，如：i > 500
```

### 6.3 发布（Publish）

```matlab
% 将脚本发布为 HTML、PDF、Word 等格式
% 使用格式化注释：
%% 标题
% 普通文本段落
%
% *粗体* 和 _斜体_
%
% 有序列表：
% # 第一项
% # 第二项
%
% 无序列表：
% * 项目一
% * 项目二
%
% 内嵌代码：|variable_name|
%
% LaTeX 公式：$E = mc^2$
%
% <https://www.mathworks.com 链接文字>

% 发布命令
publish('myscript.m', 'html');
publish('myscript.m', 'pdf');
```
