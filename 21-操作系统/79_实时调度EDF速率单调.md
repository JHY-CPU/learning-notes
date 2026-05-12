# 79_实时调度EDF速率单调

## 核心概念

- **实时调度**：在实时系统中，保证任务在截止期限前完成的调度策略。
- **硬实时（Hard Real-Time）**：必须在截止期限前完成，否则系统失败（如飞行控制）。
- **软实时（Soft Real-Time）**：偶尔超过截止期限可容忍（如视频播放）。
- 两种经典算法：**EDF（Earliest Deadline First，最早截止期限优先）**和**RMS（Rate Monotonic Scheduling，速率单调调度）**。
- 408 考点：EDF 和 RMS 的原理，可调度性判断。

## 原理分析

### 周期性实时任务模型

每个实时任务 $T_i$ 由以下参数描述：
- $P_i$：任务周期（每隔 $P_i$ 时间到达一次）。
- $e_i$：任务执行时间（每次需要 $e_i$ 时间完成）。
- $D_i$：截止期限（到达后 $D_i$ 内必须完成，通常 $D_i = P_i$）。

**CPU 利用率**：

$$U = \sum_{i=1}^{n} \frac{e_i}{P_i}$$

### EDF（最早截止期限优先）

**规则**：每次选择截止期限最近的就绪任务执行。**动态优先级**。

**特点**：
- 适用于**抢占式**调度。
- 理论上最优：如果存在任何调度方案能使所有任务满足截止期限，EDF 一定能找到。
- 可调度条件：$U \leq 1$（CPU 利用率不超过 100%）。

**示例**：任务 $T_1(P=4, e=1)$，$T_2(P=5, e=2)$，$T_3(P=10, e=3)$。

$U = 1/4 + 2/5 + 3/10 = 0.25 + 0.40 + 0.30 = 0.95 < 1$，可调度。

| 时间 | $T_1$ 截止 | $T_2$ 截止 | $T_3$ 截止 | 执行 |
|------|-----------|-----------|-----------|------|
| 0 | 4 | 5 | 10 | $T_1$（最近截止） |
| 1 | 4 | 5 | 10 | $T_2$（截止5） |
| 2 | 4 | 5 | 10 | $T_2$（完成） |
| 3 | 4 | - | 10 | $T_3$（截止10） |
| 4 | 8 | - | 10 | $T_3$（完成1单位） |
| 5 | 8 | 10 | 10 | $T_1$（新周期到达，截止8） |
| 6 | 8 | 10 | - | $T_2$（新周期，截止10） |
| 7 | 8 | 10 | - | $T_2$（完成） |
| 8 | 12 | 10 | - | $T_3$（截止10，剩余1） |
| 9 | 12 | 10 | - | $T_3$（完成） |

### RMS（速率单调调度）

**规则**：任务周期越短，优先级越高。**静态优先级**（优先级在运行前确定，不变）。

**特点**：
- 适用于**不可抢占**或**可抢占**调度。
- 优先级固定，实现简单。
- 可调度条件（充分条件）：

$$U \leq n(2^{1/n} - 1)$$

当 $n \to \infty$ 时，$n(2^{1/n} - 1) \approx \ln 2 \approx 0.693$。

| 任务数 $n$ | RMS 上界 $n(2^{1/n}-1)$ |
|-----------|----------------------|
| 1 | 1.000 |
| 2 | 0.828 |
| 3 | 0.780 |
| 4 | 0.757 |
| 5 | 0.743 |
| $\infty$ | 0.693 |

**示例**：$T_1(P=4, e=1)$，$T_2(P=5, e=2)$，$T_3(P=10, e=3)$。

$U = 0.95$。RMS 上界（$n=3$）$= 0.780$。$0.95 > 0.780$，RMS 充分条件不满足，但**不一定不可调度**（充分非必要）。

优先级：$T_1$（周期最短，最高） > $T_2$ > $T_3$（周期最长，最低）。

### EDF vs RMS 对比

| 特性 | EDF | RMS |
|------|-----|-----|
| 优先级类型 | 动态（随截止期限变化） | 静态（基于周期，不变） |
| 可调度条件 | $U \leq 1$（充要） | $U \leq n(2^{1/n}-1)$（充分） |
| 实现复杂度 | 较高（需排序截止期限） | 较低（固定优先级） |
| 适用场景 | 软实时、硬实时 | 硬实时（可预测性强） |
| 最优性 | 动态优先级中最优 | 静态优先级中最优 |
| 开销 | 每次调度需重新排序 | 一次分配优先级即可 |

## 直观理解

- **EDF** = 医院急诊：谁的病最急（截止最近）先治谁。病情变化随时调整。
- **RMS** = 快递公司：快递越紧急（周期越短）越优先送。优先级提前定好不变。

## 知识关联

- 与 25 CPU 调度：实时调度是 CPU 调度的特殊情况。
- 与 31-33 死锁：实时系统的可调度性分析与死锁避免有相似思路。
- 与 80 工作集与抖动：实时系统中抖动是致命的。
- 408 考查：计算 CPU 利用率，判断是否满足 EDF/RMS 的可调度条件。

## 代码实现

### EDF 与 RMS 实时调度模拟

```c
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

/*
 * EDF（最早截止期限优先）与 RMS（速率单调调度）模拟
 *
 * EDF: 动态优先级，每次选截止期限最近的任务
 * RMS: 静态优先级，周期越短优先级越高
 */

#define N 3        /* 任务数 */
#define SIM_TIME 30 /* 模拟时间 */

typedef struct {
    int period;     /* 周期 P */
    int exec;       /* 执行时间 e */
    int deadline;   /* 当前截止期限 */
    int remaining;  /* 剩余执行时间 */
    int next_arrive;/* 下次到达时间 */
} Task;

/* 计算 CPU 利用率 */
double calc_utilization(Task tasks[]) {
    double u = 0.0;
    for (int i = 0; i < N; i++)
        u += (double)tasks[i].exec / tasks[i].period;
    return u;
}

/* RMS 可调度充分条件 */
double rms_bound(int n) {
    return n * (pow(2.0, 1.0/n) - 1.0);
}

/* 重置任务状态 */
void reset_tasks(Task tasks[]) {
    for (int i = 0; i < N; i++) {
        tasks[i].deadline = tasks[i].period;
        tasks[i].remaining = 0;
        tasks[i].next_arrive = 0;
    }
}

/* EDF 调度模拟 */
void edf_schedule(Task tasks[]) {
    reset_tasks(tasks);
    int idle = 0;

    printf("EDF 调度过程:\n");
    printf("时间 | 执行任务 | 说明\n");
    printf("-----|---------|------\n");

    for (int t = 0; t < SIM_TIME; t++) {
        /* 检查是否有新任务到达 */
        for (int i = 0; i < N; i++) {
            if (t == tasks[i].next_arrive) {
                tasks[i].remaining = tasks[i].exec;
                tasks[i].deadline = t + tasks[i].period;
                tasks[i].next_arrive = t + tasks[i].period;
            }
        }

        /* 选择截止期限最近的就绪任务 */
        int selected = -1;
        int earliest_dd = SIM_TIME * 10;
        for (int i = 0; i < N; i++) {
            if (tasks[i].remaining > 0 && tasks[i].deadline < earliest_dd) {
                earliest_dd = tasks[i].deadline;
                selected = i;
            }
        }

        if (selected >= 0) {
            tasks[selected].remaining--;
            printf(" %2d  |   T%d    | 截止期=%d, 剩余=%d\n",
                   t, selected, tasks[selected].deadline,
                   tasks[selected].remaining);
        } else {
            printf(" %2d  |   空闲  | \n", t);
            idle++;
        }
    }
    printf("\nCPU 空闲时间: %d / %d\n", idle, SIM_TIME);
}

/* RMS 调度模拟 */
void rms_schedule(Task tasks[]) {
    reset_tasks(tasks);
    int idle = 0;

    printf("RMS 调度过程:\n");
    printf("时间 | 执行任务 | 说明\n");
    printf("-----|---------|------\n");

    /* RMS 静态优先级：周期越短优先级越高 */
    int priority[N];  /* priority[i] = 任务i的优先级(0最高) */
    for (int i = 0; i < N; i++) {
        int rank = 0;
        for (int j = 0; j < N; j++)
            if (tasks[j].period < tasks[i].period) rank++;
        priority[i] = rank;
    }

    for (int t = 0; t < SIM_TIME; t++) {
        for (int i = 0; i < N; i++) {
            if (t == tasks[i].next_arrive) {
                tasks[i].remaining = tasks[i].exec;
                tasks[i].next_arrive = t + tasks[i].period;
            }
        }

        /* 选优先级最高的就绪任务 */
        int selected = -1;
        int best_pri = N + 1;
        for (int i = 0; i < N; i++) {
            if (tasks[i].remaining > 0 && priority[i] < best_pri) {
                best_pri = priority[i];
                selected = i;
            }
        }

        if (selected >= 0) {
            tasks[selected].remaining--;
            printf(" %2d  |   T%d    | 优先级=%d, 剩余=%d\n",
                   t, selected, priority[selected],
                   tasks[selected].remaining);
        } else {
            printf(" %2d  |   空闲  | \n", t);
            idle++;
        }
    }
    printf("\nCPU 空闲时间: %d / %d\n", idle, SIM_TIME);
}

int main() {
    /* 定义三个周期任务 */
    Task tasks[N] = {
        {4, 1, 0, 0, 0},   /* T0: P=4, e=1 */
        {5, 2, 0, 0, 0},   /* T1: P=5, e=2 */
        {10, 3, 0, 0, 0},  /* T2: P=10, e=3 */
    };

    double U = calc_utilization(tasks);
    double bound = rms_bound(N);

    printf("=== 实时调度算法模拟 ===\n\n");
    printf("任务参数:\n");
    for (int i = 0; i < N; i++)
        printf("  T%d: 周期=%d, 执行时间=%d, e/P=%.2f\n",
               i, tasks[i].period, tasks[i].exec,
               (double)tasks[i].exec / tasks[i].period);

    printf("\nCPU 利用率 U = %.2f\n", U);
    printf("EDF 可调度条件: U <= 1.0 → %s\n", U <= 1.0 ? "满足" : "不满足");
    printf("RMS 充分条件:   U <= %.3f → %s\n\n",
           bound, U <= bound ? "满足" : "不满足");

    edf_schedule(tasks);
    printf("\n");
    rms_schedule(tasks);

    return 0;
}
```
