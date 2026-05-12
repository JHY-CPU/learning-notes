# MATLAB 在数学建模中的应用

## 一、曲线拟合

### 1.1 多项式拟合

```matlab
% 生成实验数据
x = linspace(0, 4, 30)';
y = 2.5*exp(-0.6*x) + 0.8*randn(size(x));

% 不同阶数多项式拟合
degrees = [1, 3, 5, 8];
figure;
plot(x, y, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
hold on;
labels = {'数据点'};
for deg = degrees
    p = polyfit(x, y, deg);
    x_fit = linspace(0, 4, 200)';
    y_fit = polyval(p, x_fit);
    plot(x_fit, y_fit, 'LineWidth', 1.5);
    labels{end+1} = sprintf('%d阶多项式', deg);
end
legend(labels, 'Location', 'northeast');
title('不同阶数多项式拟合');
xlabel('x'); ylabel('y');

% 拟合优度评估
for deg = degrees
    p = polyfit(x, y, deg);
    y_pred = polyval(p, x);
    SS_res = sum((y - y_pred).^2);
    SS_tot = sum((y - mean(y)).^2);
    R2 = 1 - SS_res / SS_tot;
    RMSE = sqrt(mean((y - y_pred).^2));
    fprintf('阶数 %d: R² = %.4f, RMSE = %.4f\n', deg, R2, RMSE);
end
```

### 1.2 fit 自定义拟合

```matlab
% 使用 fit 函数进行非线性拟合
x = linspace(0, 5, 50)';
y = 3*exp(-0.8*x) .* cos(2*x) + 0.3*randn(size(x));

% 定义拟合类型
ft = fittype('a*exp(-b*x)*cos(c*x)', ...
    'independent', 'x', ...
    'coefficients', {'a', 'b', 'c'});

% 拟合
opts = fitoptions('Method', 'NonlinearLeastSquares', ...
    'StartPoint', [1 1 1], ...
    'Lower', [0 0 0]);
[fitresult, gof] = fit(x, y, ft, opts);

% 结果
fprintf('拟合参数:\n');
fprintf('  a = %.4f\n', fitresult.a);
fprintf('  b = %.4f\n', fitresult.b);
fprintf('  c = %.4f\n', fitresult.c);
fprintf('  R² = %.4f\n', gof.rsquare);

figure;
plot(fitresult, x, y);
legend('数据', '拟合曲线');
title(sprintf('非线性拟合: R² = %.4f', gof.rsquare));

% 使用 Curve Fitter App（交互式）
% cftool(x, y);
```

### 1.3 最小二乘法手动实现

```matlab
% 线性最小二乘: y = X*beta
% 解: beta = (X'*X) \ (X'*y)

x = linspace(0, 5, 30)';
y = 2 + 3*x - 0.5*x.^2 + randn(size(x));

% 构建设计矩阵（3阶多项式）
X = [ones(size(x)), x, x.^2, x.^3];

% 最小二乘解
beta = (X'*X) \ (X'*y);
fprintf('最小二乘系数: ');
fprintf('%.4f ', beta);
fprintf('\n');

% 预测
x_new = linspace(0, 5, 200)';
X_new = [ones(size(x_new)), x_new, x_new.^2, x_new.^3];
y_pred = X_new * beta;

figure;
plot(x, y, 'ko', 'MarkerSize', 6); hold on;
plot(x_new, y_pred, 'r-', 'LineWidth', 2);
legend('数据', '最小二乘拟合');
title('线性最小二乘法');

% 加权最小二乘
weights = 1 ./ (0.1 + x);  % 权重
W = diag(weights);
beta_weighted = (X'*W*X) \ (X'*W*y);
```

---

## 二、优化问题

### 2.1 无约束优化

```matlab
% fminsearch：Nelder-Mead 单纯形法
f = @(x) (x(1)-2)^2 + (x(2)-3)^2 + (x(1)-2)*(x(2)-3);
x0 = [0, 0];
[x_opt, fval] = fminsearch(f, x0);
fprintf('最优解: x = [%.4f, %.4f], f(x) = %.6f\n', x_opt(1), x_opt(2), fval);

% fminunc：拟牛顿法（需要 Optimization Toolbox）
options = optimoptions('fminunc', 'Display', 'iter', ...
    'Algorithm', 'quasi-newton');
[x_opt, fval, exitflag, output] = fminunc(f, x0, options);

% Rosenbrock 函数（经典测试函数）
rosenbrock = @(x) 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
[x_opt, fval] = fminsearch(rosenbrock, [-1, 1]);
fprintf('Rosenbrock 最优解: [%.6f, %.6f], f = %.2e\n', ...
    x_opt(1), x_opt(2), fval);
```

### 2.2 有约束优化（fmincon）

```matlab
% 目标函数
fun = @(x) (x(1) - 1)^2 + (x(2) - 2.5)^2;

% 初始点
x0 = [0, 0];

% 不等式约束: A*x <= b
% x(1) + x(2) <= 4
% x(1) - x(2) <= 1
A = [1 1; 1 -1];
b = [4; 1];

% 等式约束: Aeq*x = beq
% 无等式约束
Aeq = [];
beq = [];

% 变量上下界
% 0 <= x(1) <= 5, 0 <= x(2) <= 5
lb = [0, 0];
ub = [5, 5];

% 非线性约束（可选）
nonlcon = [];  % 无非线性约束

% 求解
options = optimoptions('fmincon', 'Display', 'iter', ...
    'Algorithm', 'interior-point');
[x_opt, fval, exitflag] = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, ...
    nonlcon, options);

fprintf('最优解: x = [%.4f, %.4f]\n', x_opt(1), x_opt(2));
fprintf('最优值: f(x) = %.6f\n', fval);

% 非线性约束示例
% 约束: x(1)^2 + x(2)^2 <= 4（圆内）
function [c, ceq] = myNonlcon(x)
    c = x(1)^2 + x(2)^2 - 4;    % c <= 0
    ceq = [];                     % 无等式约束
end
```

### 2.3 线性规划（linprog）

```matlab
% min f'*x, subject to: A*x <= b, Aeq*x = beq, lb <= x <= ub

% 示例：生产计划优化
% min -5*x1 - 4*x2 （最大化利润 = 最小化负利润）
% 6*x1 + 4*x2 <= 24  （原材料约束）
% x1 + 2*x2 <= 6     （工时约束）
% x1, x2 >= 0

f = [-5; -4];                    % 目标函数系数
A = [6 4; 1 2];                  % 不等式约束矩阵
b = [24; 6];                     % 不等式约束右端
Aeq = [];                        % 无等式约束
beq = [];
lb = [0; 0];                     % 非负约束
ub = [];

[x_opt, fval, exitflag] = linprog(f, A, b, Aeq, beq, lb, ub);
fprintf('最优生产方案: x1 = %.2f, x2 = %.2f\n', x_opt(1), x_opt(2));
fprintf('最大利润: %.2f\n', -fval);  % 取负号还原

% 整数规划
% intcon = [1, 2];  % x1, x2 为整数
% [x_int, fval_int] = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub);
```

### 2.4 非线性规划与全局优化

```matlab
% 全局搜索
gs = GlobalSearch('Display', 'iter');
fun = @(x) (x(1)^2 + x(2) - 11)^2 + (x(1) + x(2)^2 - 7)^2;  % Himmelblau
problem = createOptimProblem('fmincon', ...
    'objective', fun, ...
    'x0', [0, 0], ...
    'lb', [-5, -5], ...
    'ub', [5, 5]);
[x_global, fval_global] = run(gs, problem);
fprintf('全局最优: x = [%.4f, %.4f], f = %.6f\n', ...
    x_global(1), x_global(2), fval_global);

% 遗传算法
% ga_options = optimoptions('ga', 'Display', 'iter', ...
%     'PopulationSize', 100, 'MaxGenerations', 200);
% [x_ga, fval_ga] = ga(fun, 2, [], [], [], [], [-5 -5], [5 5], [], ga_options);

% 多目标优化：Pareto 前沿
% fun_multi = @(x) [(x(1)-1)^2 + (x(2)-1)^2; ...
%                    (x(1)+1)^2 + (x(2)+1)^2];
% [x_pareto, fval_pareto] = gamultiobj(fun_multi, 2);
```

---

## 三、统计分析

### 3.1 描述性统计

```matlab
% 生成数据
rng(42);
data = randn(1000, 3) * diag([1 2 3]) + [0 5 -2];

% 基本统计量
fprintf('均值: %s\n', num2str(mean(data)));
fprintf('标准差: %s\n', num2str(std(data)));
fprintf('中位数: %s\n', num2str(median(data)));
fprintf('偏度: %s\n', num2str(skewness(data)));
fprintf('峰度: %s\n', num2str(kurtosis(data)));

% 分位数
prctile(data, [25 50 75])  % 四分位数

% 使用 tabulate 统计频数
categories = {'A', 'B', 'A', 'C', 'B', 'A', 'A', 'C', 'B', 'A'};
tabulate(categories);

% 使用 summary（R2024b+）
T = array2table(data, 'VariableNames', {'X1', 'X2', 'X3'});
summary_stats = grpstats(T, {}, {'mean', 'std', 'min', 'max'});
```

### 3.2 假设检验

```matlab
% t 检验：单样本
data = 5.2 + 0.8*randn(30, 1);  % 均值约 5.2
[h, p, ci, stats] = ttest(data, 5);  % 检验均值是否为 5
fprintf('单样本 t 检验: h=%d, p=%.4f\n', h, p);
fprintf('置信区间: [%.4f, %.4f]\n', ci(1), ci(2));

% t 检验：双样本
group1 = randn(50, 1) + 0.5;
group2 = randn(50, 1) - 0.3;
[h, p, ci, stats] = ttest2(group1, group2);  % 独立双样本
fprintf('双样本 t 检验: h=%d, p=%.4f\n', h, p);

% 配对 t 检验
before = randn(30, 1);
after = before + 0.5 + 0.2*randn(30, 1);
[h, p] = ttest(before, after);
fprintf('配对 t 检验: h=%d, p=%.4f\n', h, p);

% 卡方检验
observed = [50, 30, 20];
expected = [33.3, 33.3, 33.3];
[h, p, stats] = chi2gof(1, 'Freq', observed, 'Expected', expected);
fprintf('卡方检验: h=%d, p=%.4f\n', h, p);

% Kolmogorov-Smirnov 检验（正态性检验）
data = randn(100, 1);
[h, p] = kstest(data);
fprintf('KS 检验（正态性）: h=%d, p=%.4f\n', h, p);

% ANOVA（方差分析）
group1 = randn(30, 1);
group2 = randn(30, 1) + 1;
group3 = randn(30, 1) + 2;
[p, tbl, stats] = anova1([group1, group2, group3]);
fprintf('ANOVA p 值: %.4f\n', p);
```

### 3.3 回归分析

```matlab
% 线性回归
rng(42);
x1 = randn(100, 1);
x2 = randn(100, 1);
y = 3 + 2*x1 - 1.5*x2 + 0.5*randn(100, 1);

% fitlm：线性模型
tbl = table(x1, x2, y);
mdl = fitlm(tbl, 'y ~ x1 + x2');
disp(mdl);

% 获取系数
coefficients = mdl.Coefficients;
fprintf('系数:\n');
disp(coefficients);

% 预测
y_pred = predict(mdl, tbl);
R2 = mdl.Rsquared.Ordinary;
fprintf('R² = %.4f\n', R2);

% 残差图
figure;
plotResiduals(mdl, 'fitted');
title('残差图');

% 逐步回归
mdl_step = stepwiselm(tbl, 'y ~ x1 + x2', 'Verbose', 2);

% 逻辑回归（分类）
X = randn(100, 2);
y_logistic = (X(:,1) + X(:,2) > 0);
tbl_logistic = table(X(:,1), X(:,2), y_logistic, ...
    'VariableNames', {'X1', 'X2', 'Y'});
mdl_logistic = fitglm(tbl_logistic, 'Y ~ X1 + X2', ...
    'Distribution', 'binomial');
```

---

## 四、蒙特卡洛方法

### 4.1 蒙特卡洛积分

```matlab
% 计算 π 的近似值
N = 1e6;
x = rand(N, 1);
y = rand(N, 1);
inside = sum(x.^2 + y.^2 <= 1);
pi_est = 4 * inside / N;
fprintf('蒙特卡洛估计 π = %.6f (误差: %.2e)\n', pi_est, abs(pi_est - pi));

% 可视化
figure;
idx = x.^2 + y.^2 <= 1;
scatter(x(idx), y(idx), 1, 'b', 'filled'); hold on;
scatter(x(~idx), y(~idx), 1, 'r', 'filled');
axis equal;
title(sprintf('蒙特卡洛估计 π ≈ %.4f', pi_est));

% 蒙特卡洛积分: ∫f(x)dx ≈ (b-a)/N * Σf(xi)
f = @(x) exp(-x.^2);
N_mc = 1e6;
x_mc = rand(N_mc, 1) * 2;  % [0, 2]
integral_est = 2 * mean(f(x_mc));
integral_exact = integral(f, 0, 2);
fprintf('蒙特卡洛积分: %.6f (精确值: %.6f, 误差: %.2e)\n', ...
    integral_est, integral_exact, abs(integral_est - integral_exact));
```

### 4.2 随机模拟

```matlab
% 模拟掷骰子
N = 1e5;
rolls = randi(6, N, 1);
fprintf('均值: %.4f (理论: 3.5)\n', mean(rolls));

% 模拟布朗运动
N_paths = 100;
N_steps = 500;
dt = 0.01;
t = (0:N_steps) * dt;
dW = sqrt(dt) * randn(N_steps, N_paths);
W = [zeros(1, N_paths); cumsum(dW, 1)];

figure;
plot(t, W, 'b-', 'LineWidth', 0.5);
xlabel('时间'); ylabel('W(t)');
title(sprintf('布朗运动 (%d 条路径)', N_paths));

% 期权定价（Black-Scholes 蒙特卡洛）
S0 = 100;       % 初始股价
K = 105;        % 行权价
r = 0.05;       % 无风险利率
sigma = 0.2;    % 波动率
T = 1;          % 到期时间
N_sim = 1e6;

Z = randn(N_sim, 1);
ST = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z);
payoff = max(ST - K, 0);
price = exp(-r*T) * mean(payoff);
fprintf('蒙特卡洛期权价格: %.4f\n', price);

% Black-Scholes 解析解
d1 = (log(S0/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T));
d2 = d1 - sigma*sqrt(T);
price_exact = S0*normcdf(d1) - K*exp(-r*T)*normcdf(d2);
fprintf('BS 解析解: %.4f\n', price_exact);
```

---

## 五、图论与网络

```matlab
% 创建图
s = [1 1 2 2 3 3 4 5];
t = [2 3 3 4 4 5 5 6];
weights = [1 2 1 3 2 1 4 2];
G = graph(s, t, weights);

% 可视化
figure;
plot(G, 'EdgeLabel', G.Edges.Weight, 'NodeFontSize', 14, ...
    'MarkerSize', 10, 'LineWidth', 2);
title('加权无向图');

% 最短路径
[dist, path, pred] = shortestpath(G, 1, 6);
fprintf('最短路径: %s\n', num2str(path));
fprintf('最短距离: %.1f\n', dist);

% 最小生成树
T_mst = minspantree(G);
fprintf('最小生成树权重: %.1f\n', sum(T_mst.Edges.Weight));

% 度、中心性
deg = degree(G);         % 度
bw = centrality(G, 'betweenness');  % 介数中心性
fprintf('度: %s\n', num2str(deg'));
fprintf('介数中心性: %s\n', num2str(bw'));

% 有向图
G_digraph = digraph(s, t, weights);
figure;
plot(G_digraph, 'EdgeLabel', G_digraph.Edges.Weight);
title('有向图');

% 邻接矩阵
A_adj = adjacency(G);
disp('邻接矩阵:');
disp(full(A_adj));
```

---

## 六、模糊综合评价

```matlab
% 模糊综合评价模型
% 因素集 U = {u1, u2, u3}
% 评语集 V = {优, 良, 中, 差}

% 权重向量
W = [0.5, 0.3, 0.2];

% 模糊评价矩阵（每行一个因素，每列一个评语等级）
R = [0.4 0.3 0.2 0.1;
     0.3 0.4 0.2 0.1;
     0.2 0.3 0.3 0.2];

% 模糊综合评价（加权平均型）
B = W * R;
fprintf('综合评价结果: %s\n', num2str(B));
fprintf('评价等级: %s\n', num2str({'优','良','中','差'}));
[~, best] = max(B);
fprintf('最佳等级: %s\n', {'优','良','中','差'}{best});

% 最大最小型
B_minmax = zeros(1, size(R, 2));
for j = 1:size(R, 2)
    B_minmax(j) = max(min(W', R(:, j)));
end
fprintf('最大最小型结果: %s\n', num2str(B_minmax));
```

---

## 七、灵敏度分析

```matlab
% 参数灵敏度分析
model = @(params) params(1)*exp(-params(2)) + params(3);

% 基准参数
p0 = [2.5, 0.8, 0.5];
y0 = model(p0);

% 单参数灵敏度
n_points = 50;
figure;
param_names = {'幅值 A', '衰减 λ', '偏置 B'};
for i = 1:3
    p_range = linspace(p0(i)*0.5, p0(i)*1.5, n_points);
    y_range = zeros(n_points, 1);
    for j = 1:n_points
        p_temp = p0;
        p_temp(i) = p_range(j);
        y_range(j) = model(p_temp);
    end
    subplot(1, 3, i);
    plot(p_range, y_range, 'b-', 'LineWidth', 2);
    hold on;
    plot(p0(i), y0, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    xlabel(param_names{i}); ylabel('输出');
    title(sprintf('灵敏度: %s', param_names{i}));
    grid on;
end
sgtitle('参数灵敏度分析');
```
