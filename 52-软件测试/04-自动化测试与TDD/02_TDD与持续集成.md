# TDD与持续集成


## 一、测试驱动开发（TDD）


TDD（Test-Driven Development）是一种先写测试、再写代码、最后重构的开发方法论。


### 1.1 红-绿-重构循环


| 阶段 | 含义 | 动作 |
| --- | --- | --- |
| **Red（红）** | 写一个失败的测试 | 写测试用例，运行确认失败（因为还没有实现代码） |
| **Green（绿）** | 写最少的代码让测试通过 | 编写实现代码，不追求完美，只求测试通过 |
| **Refactor（重构）** | 优化代码结构 | 消除重复、改善命名、提取方法，保持测试通过 |


### 1.2 TDD示例：实现字符串计算器


```
// 第一轮：Red - 写测试
test('空字符串返回0', () => {
    expect(add('')).toBe(0);
});
// 运行：失败（add函数不存在）

// 第一轮：Green - 最少代码
function add(str) {
    return 0;
}
// 运行：通过

// 第二轮：Red - 添加新测试
test('单个数字返回该数字', () => {
    expect(add('5')).toBe(5);
});
// 运行：失败

// 第二轮：Green - 让测试通过
function add(str) {
    if (!str) return 0;
    return parseInt(str);
}

// 第三轮：Red - 添加逗号分隔测试
test('逗号分隔两个数字', () => {
    expect(add('1,2')).toBe(3);
});

// 第三轮：Green
function add(str) {
    if (!str) return 0;
    const nums = str.split(',').map(Number);
    return nums.reduce((sum, n) => sum + n, 0);
}

// Refactor：代码已清晰，无需重构
```


### 1.3 TDD的三条法则


1. 不允许写任何产品代码，除非是为了让失败的测试通过
2. 不允许写更多的测试代码，只写刚好会失败的测试（编译失败也算）
3. 不允许写更多的产品代码，只写刚好让测试通过的代码


### 1.4 TDD vs 传统开发


| 对比 | TDD | 传统开发（Test-Last） |
| --- | --- | --- |
| 测试编写时机 | 写代码之前 | 写代码之后 |
| 代码可测试性 | 天然高（为测试而写） | 可能难以测试 |
| 设计质量 | 强迫考虑接口设计 | 可能需要后期修改 |
| 文档 | 测试即文档 | 需要额外文档 |
| 缺陷修复成本 | 低（发现早） | 高（发现晚） |


## 二、持续集成（CI）


持续集成是一种软件开发实践，团队成员频繁地集成代码（通常每天多次），每次集成都通过自动构建和测试验证。


### 2.1 CI的核心实践


- **频繁提交**
   ：每天多次提交代码到主干
- **自动构建**
   ：每次提交触发自动编译和构建
- **自动测试**
   ：构建后自动运行单元测试、集成测试
- **快速反馈**
   ：构建/测试失败立即通知开发者
- **保持主干健康**
   ：主干分支始终处于可发布状态


### 2.2 CI/CD流水线


$$
代码提交 → 自动构建 → 单元测试 → 代码质量检查 → 集成测试 → 构建产物 → 部署到测试环境 → 验收测试 → 部署到生产环境
$$


| 阶段 | 说明 | 工具 |
| --- | --- | --- |
| 代码管理 | 版本控制和代码审查 | Git, GitHub, GitLab |
| 自动构建 | 编译、打包 | Maven, Gradle, npm |
| 自动测试 | 运行各级测试 | Jest, JUnit, pytest |
| 代码质量 | 静态分析、覆盖率 | SonarQube, ESLint |
| CI服务器 | 编排和执行流水线 | Jenkins, GitHub Actions, GitLab CI |
| 容器化 | 环境一致性 | Docker |
| 部署 | 自动化部署 | Kubernetes, Ansible |


### 2.3 GitHub Actions示例


```
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Run linting
        run: npm run lint

      - name: Run tests
        run: npm test -- --coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```


## 三、测试自动化策略


### 3.1 测试金字塔


| 层级 | 数量 | 速度 | 成本 | 说明 |
| --- | --- | --- | --- | --- |
| 单元测试 | 多（70%） | 快 | 低 | 测试函数/类，隔离依赖 |
| 集成测试 | 中（20%） | 中 | 中 | 测试模块间交互 |
| E2E测试 | 少（10%） | 慢 | 高 | 测试完整用户流程 |


> **Important:** **测试金字塔原则：**
>
>
> - 底层测试（单元）多而快，成本低
>
>
> - 顶层测试（E2E）少而慢，成本高
>
>
> - 不要倒金字塔：太多E2E测试会导致反馈慢、维护成本高


### 3.2 测试覆盖率


- **行覆盖率**
   ：执行的代码行 / 总代码行
- **分支覆盖率**
   ：覆盖的分支 / 总分支数
- **函数覆盖率**
   ：调用的函数 / 总函数数
- 业界一般目标：行覆盖 80%+，分支覆盖 70%+


> **Note:** **注意：**
> 覆盖率高不等于质量高。100%覆盖率不代表没有bug，还需要考虑测试用例的有效性。覆盖率是质量的必要条件而非充分条件。


## 四、知识要点总结


1. TDD三步骤：Red（写失败测试）→ Green（写代码通过）→ Refactor（重构）
2. TDD三条法则：先测试、最小测试、最小代码
3. CI：频繁集成 + 自动构建 + 自动测试 + 快速反馈
4. CI/CD流水线：提交→构建→测试→质量检查→部署
5. 测试金字塔：单元70%、集成20%、E2E 10%
6. 覆盖率是质量的必要条件，非充分条件


<!-- Converted from: 02_TDD与持续集成.html -->
