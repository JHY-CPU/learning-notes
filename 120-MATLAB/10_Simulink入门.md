# Simulink 入门

## 一、Simulink 基础

### 1.1 Simulink 简介

Simulink 是 MATLAB 的附加产品，提供图形化建模和仿真环境，广泛用于控制系统、信号处理、通信系统等领域的动态系统仿真。

**核心概念：**

| 概念 | 说明 |
|------|------|
| 模型（Model） | 由模块和连线组成的系统描述 |
| 模块（Block） | 基本功能单元，实现特定运算 |
| 连线（Line） | 传递信号的通道 |
| 信号（Signal） | 模块间传递的数据 |
| 子系统（Subsystem） | 封装的模块组合 |
| 模型引用（Model Reference） | 引用其他模型 |

### 1.2 启动与界面

```matlab
% 启动 Simulink
simulink                    % 打开 Simulink 启动页面
open_system('modelName')    % 打开已有模型

% 创建新模型
new_system('myModel');      % 创建空模型
open_system('myModel');     % 打开
save_system('myModel');     % 保存

% 界面主要区域
% 1. 模型窗口：搭建系统
% 2. 库浏览器：查找和拖拽模块
% 3. 模型浏览器：管理子系统层次
% 4. 诊断查看器：查看仿真警告和错误
```

---

## 二、常用模块库

### 2.1 Sources（信号源）

```matlab
% 常用信号源模块
% - Constant: 恒定值
% - Step: 阶跃信号
% - Ramp: 斜坡信号
% - Sine Wave: 正弦波
% - Clock: 仿真时间
% - From Workspace: 从工作区读取
% - From File: 从文件读取
% - Random Number: 随机数
% - Band-Limited White Noise: 带限白噪声
% - Signal Builder: 自定义信号波形

% 通过 MATLAB 命令设置模块参数
% set_param('myModel/Sine Wave', 'Amplitude', '2', 'Frequency', '10');
```

### 2.2 Sinks（输出/显示）

```matlab
% 常用输出模块
% - Scope: 示波器（最常用）
% - Display: 数值显示
% - To Workspace: 输出到工作区变量
% - To File: 输出到文件
% - XY Graph: XY 图
% - Stop Simulation: 条件停止仿真
% - Out1: 输出端口（子系统输出）
```

### 2.3 Math Operations（数学运算）

```matlab
% 常用数学模块
% - Sum: 加减运算
% - Product: 乘除运算
% - Gain: 增益（乘常数）
% - Integrator: 积分器
% - Derivative: 微分器
% - Trigonometric Function: 三角函数
% - Math Function: 幂、对数等
% - Abs: 绝对值
% - Rounding Function: 取整
% - Dot Product: 点积
```

### 2.4 Continuous（连续系统）

```matlab
% 连续系统模块
% - Transfer Fcn: 传递函数 G(s) = N(s)/D(s)
% - State-Space: 状态空间模型 dx/dt = Ax + Bu, y = Cx + Du
% - Integrator: 1/s 积分
% - Derivative: s 微分
% - Transport Delay: 传输延迟
% - Zero-Pole: 零极点形式传递函数

% 示例：二阶系统传递函数
% G(s) = 1/(s^2 + 2*0.5*1*s + 1)
% 在 Transfer Fcn 模块中设置：
% Numerator coefficients: [1]
% Denominator coefficients: [1 1 1]
```

### 2.5 Discrete（离散系统）

```matlab
% 离散系统模块
% - Discrete Transfer Fcn: 离散传递函数
% - Discrete State-Space: 离散状态空间
% - Unit Delay: 单位延迟 z^(-1)
% - Discrete-Time Integrator: 离散积分器
% - Zero-Order Hold: 零阶保持器
% - Discrete Filter: 离散滤波器
```

---

## 三、模型创建示例

### 3.1 简单 PID 控制器

```matlab
% 程序化创建 Simulink 模型
modelName = 'pid_control';

% 创建新模型
new_system(modelName);
open_system(modelName);

% 添加模块
add_block('simulink/Sources/Step', [modelName '/Reference']);
add_block('simulink/Math Operations/Sum', [modelName '/Sum']);
add_block('simulink/Continuous/Transfer Fcn', [modelName '/Plant']);
add_block('simulink/Sinks/Scope', [modelName '/Scope']);

% PID 控制器需要手动搭建或使用 PID Controller 模块
% add_block('simulink/Continuous/PID Controller', [modelName '/PID']);

% 设置参数
set_param([modelName '/Reference'], 'Time', '1', 'After', '1');
set_param([modelName '/Plant'], 'Numerator', '[1]', 'Denominator', '[1 2 1]');
set_param([modelName '/Sum'], 'Inputs', '+-');

% 连接模块
add_line(modelName, 'Reference/1', 'Sum/1');
add_line(modelName, 'Sum/1', 'Plant/1');
add_line(modelName, 'Plant/1', 'Scope/1');
add_line(modelName, 'Plant/1', 'Sum/2');  % 反馈

% 保存
save_system(modelName);
fprintf('模型 %s 创建完成\n', modelName);
```

### 3.2 RLC 电路仿真

```matlab
% RLC 串联电路: L*di/dt + R*i + (1/C)*∫i dt = V(t)
% 状态变量: x1 = i (电流), x2 = v_C (电容电压)

% 等效状态空间:
% dx1/dt = (V - R*x1 - x2) / L
% dx2/dt = x1 / C

% 参数
R = 10;     % 电阻 (Ohm)
L = 0.1;    % 电感 (H)
C = 1e-4;   % 电容 (F)

% 在 Simulink 中搭建：
% Sources -> Step (阶跃电压)
% Continuous -> State-Space
A = [-R/L, -1/L; 1/C, 0];
B = [1/L; 0];
C_out = [R, 0];    % 输出为电阻两端电压
D = 0;

% 或用 Transfer Fcn 形式
% V_R(s)/V(s) = R*s / (L*s^2 + R*s + 1/C)
num = [R, 0];
den = [L, R, 1/C];
fprintf('传递函数分子: [%s]\n', num2str(num));
fprintf('传递函数分母: [%s]\n', num2str(den));

% 频率响应分析
sys = tf(num, den);
figure;
bode(sys);
title('RLC 电路 Bode 图');

figure;
step(sys);
title('RLC 电路阶跃响应');
```

---

## 四、仿真参数设置

### 4.1 仿真配置

```matlab
% 获取和设置仿真参数
modelName = 'myModel';

% 求解器设置
set_param(modelName, ...
    'Solver', 'ode45', ...              % 求解器类型
    'StartTime', '0', ...               % 起始时间
    'StopTime', '10', ...               % 终止时间
    'MaxStep', '0.01', ...              % 最大步长
    'RelTol', '1e-3', ...               % 相对误差
    'AbsTol', '1e-6');                  % 绝对误差

% 求解器选择指南
% 变步长求解器：
%   ode45: 非刚性问题首选（Dormand-Prince 4/5阶）
%   ode23: 非刚性，低精度（Bogacki-Shampine 2/3阶）
%   ode113: 非刚性，多步法
%   ode15s: 刚性问题（NDF/BDF）
%   ode23s: 刚性（Rosenbrock）
%   ode23t: 中等刚性
%   ode23tb: 刚性低精度
%
% 固定步长求解器：
%   ode5: 固定步长 Dormand-Prince
%   ode4: 固定步长 4 阶 Runge-Kutta
%   ode3: 固定步长 Bogacki-Shampine

% 通过 MATLAB 命令运行仿真
simOut = sim(modelName, 'StopTime', '20');
% 或
simOut = sim(modelName, ...
    'StopTime', '20', ...
    'Solver', 'ode45', ...
    'MaxStep', '0.001');
```

### 4.2 数据导入导出

```matlab
% 配置数据导入/导出
set_param(modelName, ...
    'SaveOutput', 'on', ...             % 保存输出
    'OutputSaveName', 'yout', ...       % 输出变量名
    'SaveState', 'on', ...              % 保存状态
    'StateSaveName', 'xout', ...        % 状态变量名
    'SaveTime', 'on', ...               % 保存时间
    'TimeSaveName', 'tout');            % 时间变量名

% From Workspace 模块输入数据
t_input = (0:0.01:10)';
u_input = sin(2*pi*t_input);
inputData = [t_input, u_input];  % 第一列时间，第二列数据
set_param('myModel/From Workspace', 'VariableName', 'inputData');

% 运行仿真
simOut = sim(modelName);
tout = simOut.tout;
yout = simOut.yout;
figure;
plot(tout, yout);
title('仿真结果');
```

---

## 五、示波器（Scope）配置

```matlab
% Scope 模块配置
% 通过 GUI 设置：
% - 坐标轴范围
% - 线条样式和颜色
% - 数据记录
% - 布局（多通道显示）

% 程序化控制 Scope
% scope_h = get_param('myModel/Scope', 'Handle');
% set_param('myModel/Scope', 'NumInputPorts', '3');  % 3个输入

% 使用 To Workspace 代替 Scope 获取数据
% 设置 To Workspace 模块：
% Variable name: simout
% Save format: Array (或 Structure, Structure with Time)

% 仿真后绘图
set_param('myModel/To Workspace', 'SaveFormat', 'Timeseries');
simOut = sim(modelName);
data = simOut.get('simout');
figure;
plot(data.Time, data.Data, 'b-', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('输出');
title('仿真结果（To Workspace）');
grid on;
```

---

## 六、子系统

### 6.1 创建子系统

```matlab
% 方法1：选中模块后创建子系统
% 选中要封装的模块 -> 右键 -> Create Subsystem
% 或 Ctrl+G

% 方法2：程序化创建
modelName = 'subsystem_demo';
new_system(modelName);
open_system(modelName);

% 添加模块
add_block('simulink/Sources/Sine Wave', [modelName '/Sine']);
add_block('simulink/Math Operations/Gain', [modelName '/Gain']);
add_block('simulink/Sinks/Scope', [modelName '/Scope']);

% 创建子系统
add_block('simulink/Ports & Subsystems/Subsystem', [modelName '/Filter']);

% 连接
add_line(modelName, 'Sine/1', 'Filter/1');
add_line(modelName, 'Filter/1', 'Gain/1');
add_line(modelName, 'Gain/1', 'Scope/1');

save_system(modelName);

% 封装子系统（Mask）
% 右键子系统 -> Mask -> Create Mask
% 可设置参数对话框、图标、初始化代码等
```

### 6.2 条件执行子系统

```matlab
% 使能子系统（Enabled Subsystem）
% 内含 Enable 端口，当控制信号 > 0 时执行

% 触发子系统（Triggered Subsystem）
% 内含 Trigger 端口，沿触发或电平触发

% 函数调用子系统（Function-Call Subsystem）
% 由 Stateflow 或 S-Function 触发

% 启用/触发组合子系统
% 同时含 Enable 和 Trigger 端口
```

---

## 七、S-Function

```matlab
% S-Function 允许用 MATLAB/C/C++/Fortran 编写自定义模块

% MATLAB S-Function 模板
% 文件名：my_sfun.m
function [sys, x0, str, ts] = my_sfun(t, x, u, flag)
    switch flag
        case 0  % 初始化
            [sys, x0, str, ts] = mdlInitializeSizes;
        case 1  % 计算导数
            sys = mdlDerivatives(t, x, u);
        case 2  % 更新离散状态
            sys = mdlUpdate(t, x, u);
        case 3  % 计算输出
            sys = mdlOutputs(t, x, u);
        case {4, 9}  % 下一步仿真时间 / 仿真结束
            sys = [];
        otherwise
            error(['Unhandled flag = ', num2str(flag)]);
    end
end

function [sys, x0, str, ts] = mdlInitializeSizes()
    sizes = simsizes;
    sizes.NumContStates  = 0;     % 连续状态数
    sizes.NumDiscStates  = 0;     % 离散状态数
    sizes.NumOutputs     = 1;     % 输出数
    sizes.NumInputs      = 1;     % 输入数
    sizes.DirFeedthrough = 1;     % 直接馈通
    sizes.NumSampleTimes = 1;     % 采样时间数
    sys = simsizes(sizes);
    x0  = [];                     % 初始状态
    str = [];
    ts  = [0 0];                  % 连续采样时间
end

function sys = mdlOutputs(t, x, u)
    sys = u^2;  % 输出 = 输入的平方
end

function sys = mdlDerivatives(t, x, u)
    sys = [];
end
```
