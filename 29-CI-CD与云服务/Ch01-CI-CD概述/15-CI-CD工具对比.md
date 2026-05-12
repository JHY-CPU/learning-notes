# CI-CD工具对比

## 一、概念说明

主流CI/CD工具包括GitHub Actions、Jenkins、GitLab CI等，各有优缺点。

## 二、工具对比

| 特性 | GitHub Actions | Jenkins | GitLab CI |
|------|----------------|---------|-----------|
| 托管方式 | SaaS | 自托管 | SaaS/自托管 |
| 配置方式 | YAML | Jenkinsfile | YAML |
| 学习曲线 | 低 | 中 | 低 |
| 插件生态 | 丰富 | 最丰富 | 中等 |
| 免费额度 | 2000分钟/月 | 无限 | 400分钟/月 |
| 适合场景 | GitHub项目 | 企业级 | GitLab项目 |

## 三、选择建议

```bash
# GitHub项目 → GitHub Actions
# 无缝集成，开箱即用

# 企业级需求 → Jenkins
# 高度可定制，插件丰富

# GitLab项目 → GitLab CI
# 深度集成，一站式

# 云原生 → ArgoCD/FluxCD
# GitOps风格，Kubernetes原生
```

## 四、新兴工具

```bash
# ArgoCD - GitOps持续部署
# FluxCD - GitOps工具
# Tekton - Kubernetes原生CI/CD
# Dagger - 容器化CI/CD
# CircleCI - 云原生CI/CD
```

## 五、注意事项

1. **选择依据**：基于团队技术栈和需求
2. **迁移成本**：考虑工具迁移的成本
3. **社区支持**：选择活跃的社区和工具
4. **安全考虑**：评估工具的安全特性

## 六、详细特性对比

| 特性 | GitHub Actions | Jenkins | GitLab CI | CircleCI |
|------|----------------|---------|-----------|----------|
| 配置语言 | YAML | Groovy | YAML | YAML |
| 并行构建 | 支持 | 支持 | 支持 | 支持 |
| 矩阵构建 | 原生支持 | 需插件 | 原生支持 | 原生支持 |
| Docker支持 | 原生 | 需插件 | 原生 | 原生 |
| 缓存 | 原生 | 需插件 | 原生 | 原生 |
| 制品管理 | 内置 | 需插件 | 内置 | 内置 |
| 秘密管理 | Secrets | Credentials | Variables | Contexts |
| 环境保护 | Environments | 需插件 | Environments | Contexts |
| 审计日志 | 内置 | 需插件 | 内置 | 内置 |
| API | REST/GraphQL | REST | GraphQL/REST | REST/GraphQL |

## 七、迁移指南

### 从Jenkins迁移到GitHub Actions

```groovy
// Jenkinsfile (原始)
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'npm install'
                sh 'npm run build'
            }
        }
        stage('Test') {
            steps {
                sh 'npm test'
            }
        }
    }
}
```

```yaml
# .github/workflows/ci.yml (迁移后)
name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm install
      - run: npm run build
  test:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm install
      - run: npm test
```

### Jenkins概念到GitHub Actions映射

| Jenkins概念 | GitHub Actions对应 |
|------------|-------------------|
| Jenkinsfile | .github/workflows/*.yml |
| agent | runs-on |
| stage | job |
| steps | steps |
| sh | run |
| credentials | secrets |
| environment | env |
| post { failure } | if: failure() |
| parallel | strategy.matrix / jobs并行 |

## 八、工具选型决策树

```
你的代码托管在哪里?
├── GitHub → GitHub Actions（首选）
├── GitLab → GitLab CI（首选）
├── Bitbucket → Bitbucket Pipelines
└── 自建/其他 → 考虑Jenkins

是否有特殊需求?
├── 需要高度定制化 → Jenkins
├── 需要Kubernetes原生 → Tekton / ArgoCD
├── 需要GitOps → ArgoCD / FluxCD
└── 需要快速上手 → GitHub Actions / GitLab CI

团队规模?
├── 小团队（<10人）→ SaaS方案（GitHub Actions/GitLab CI）
├── 中型团队（10-50人）→ SaaS + 自托管混合
└── 大型团队（>50人）→ 自托管 Jenkins / GitLab
```
