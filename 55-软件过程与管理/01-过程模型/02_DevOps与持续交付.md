# DevOps 与持续交付


## 1. DevOps 核心理念

**DevOps**
= Development + Operations。不是工具或职位，而是文化和实践，目标是缩短交付周期、提高部署频率、降低变更失败率。

#### CALMS 框架


- **C**
   ulture — 文化协作
- **A**
   utomation — 自动化一切
- **L**
   ean — 精益思维
- **M**
   easurement — 度量反馈
- **S**
   haring — 知识共享


#### DORA 四个关键指标


- **部署频率**
   (Deployment Frequency)
- **变更前置时间**
   (Lead Time for Changes)
- **变更失败率**
   (Change Failure Rate)
- **恢复时间**
   (Time to Restore)


### DevOps 生命周期

Plan
→
Code
→
Build
→
Test
→
Release
→
Deploy
→
Operate
→
Monitor
↺
Plan

## 2. CI/CD 流水线 (Pipeline)


### 2.1 持续集成 (CI)


开发人员频繁地将代码合并到主分支，每次合并自动触发构建和测试。


```
# GitHub Actions CI 配置 (.github/workflows/ci.yml)
name: CI Pipeline
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: 设置 JDK
        uses: actions/setup-java@v4
        with:
          java-version: '21'
          distribution: 'temurin'

      - name: 构建并测试
        run: |
          ./gradlew build
          ./gradlew test

      - name: 代码质量检查
        run: ./gradlew checkstyleMain

      - name: 上传构建产物
        uses: actions/upload-artifact@v4
        with:
          name: app-jar
          path: build/libs/*.jar
```


### 2.2 持续交付 (CD)


将通过 CI 的构建产物自动部署到各个环境，确保随时可发布。


```
# CD 部署阶段
  deploy-staging:
    needs: build-and-test
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: 部署到 Staging
        run: |
          kubectl set image deployment/app \
            app=${{ secrets.REGISTRY }}/app:${{ github.sha }}
          kubectl rollout status deployment/app -n staging

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'
    steps:
      - name: 部署到生产
        run: |
          kubectl set image deployment/app \
            app=${{ secrets.REGISTRY }}/app:${{ github.sha }}
          kubectl rollout status deployment/app -n production
```


## 3. 部署策略 (Deployment Strategies)


| 策略 | 原理 | 回滚速度 | 风险 | 资源消耗 |
| --- | --- | --- | --- | --- |
| 蓝绿部署 | 两套环境并行，切流量 | 秒级 | 低 | 高（双倍资源） |
| 金丝雀发布 | 小比例流量导向新版本 | 秒级 | 低 | 低 |
| 滚动更新 | 逐个替换旧实例 | 分钟级 | 中 | 低 |
| A/B 测试 | 按条件分流到不同版本 | 秒级 | 低 | 中 |
| 灰度发布 | 按地区/用户组逐步放量 | 分钟级 | 低 | 中 |


### 蓝绿部署示例


```
# 蓝绿部署 — Nginx 切流
upstream app {
    server blue.internal:8080;   # 蓝色环境（当前生产）
    # server green.internal:8080;  # 绿色环境（待切）
}

# 部署新版本到绿色环境后，修改配置切换：
upstream app {
    # server blue.internal:8080;   # 下线蓝色
    server green.internal:8080;   # 切换到绿色
}
# nginx -s reload  （秒级切换）
```


### 金丝雀发布 (Kubernetes)


```
# 金丝雀发布 — Istio 流量分配
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: app-canary
spec:
  hosts: ["app.example.com"]
  http:
    - match:
        - headers:
            x-canary:
              exact: "true"
      route:
        - destination:
            host: app-v2          # 金丝雀版本
    - route:
        - destination:
            host: app-v1
          weight: 95              # 95% 流量到稳定版
        - destination:
            host: app-v2
          weight: 5               # 5% 流量到新版本
```


## 4. 基础设施即代码 (IaC)


#### IaC 工具


- **Terraform**
   — 多云基础设施
- **Pulumi**
   — 通用语言定义基础设施
- **Ansible**
   — 配置管理
- **Kubernetes YAML**
   — 容器编排
- **Docker Compose**
   — 本地开发环境


#### IaC 原则


- 不可变基础设施 (Immutable)
- 声明式而非命令式
- 版本控制一切配置
- 环境一致性 (Dev = Prod)
- 幂等性（重复执行结果一致）


```
# Terraform — 声明式基础设施
resource "aws_ecs_cluster" "main" {
  name = "production-cluster"
}

resource "aws_ecs_service" "app" {
  name            = "web-app"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
}

# 运行: terraform init → terraform plan → terraform apply
```


## 5. 可观测性 (Observability)


#### 三大支柱


- **日志 (Logs)**
   ：发生了什么
- **指标 (Metrics)**
   ：系统状态如何
- **链路追踪 (Traces)**
   ：请求经过了哪里


#### 工具生态


- 日志：ELK Stack, Loki
- 指标：Prometheus + Grafana
- 追踪：Jaeger, Zipkin, OpenTelemetry
- 告警：PagerDuty, AlertManager

**DevOps 成熟度路径：**
手动部署 → 脚本化 → CI/CD 流水线 → 自动化测试 → 基础设施即代码 → GitOps → 可观测性驱动 → 全自动运维


<!-- Converted from: 02_DevOps与持续交付.html -->
