# DevOps与CI-CD

## 一、概念说明

DevOps是一种文化和实践，强调开发（Dev）和运维（Ops）的协作。CI/CD是DevOps的核心技术实践。

## 二、DevOps生命周期

```
计划 → 编码 → 构建 → 测试 → 发布 → 部署 → 运营 → 监控
  └────────────── 反馈循环 ──────────────┘
```

## 三、CI/CD在DevOps中的位置

```bash
# DevOps实践
# 1. 版本控制 - Git
# 2. CI/CD - 自动化构建和部署
# 3. 基础设施即代码 - Terraform/Ansible
# 4. 监控和日志 - Prometheus/ELK
# 5. 沟通协作 - Slack/Jira

# CI/CD是DevOps的自动化支柱
```

## 四、实践原则

```bash
# 1. 一切皆代码
# 应用代码、配置、基础设施都版本控制

# 2. 自动化一切
# 构建、测试、部署、监控都自动化

# 3. 快速反馈
# 尽早发现问题，快速修复

# 4. 持续改进
# 定期回顾，持续优化流程
```

## 五、工具链

```bash
# 版本控制: Git (GitHub/GitLab/Bitbucket)
# CI/CD: GitHub Actions/Jenkins/GitLab CI
# 容器化: Docker/Kubernetes
# 监控: Prometheus/Grafana
# 日志: ELK Stack
# 通信: Slack/Teams
```

## 六、注意事项

1. **文化先行**：工具是手段，文化是根本
2. **渐进改进**：不要一步到位
3. **度量驱动**：用数据衡量改进效果
4. **团队协作**：打破开发和运维的壁垒

## 七、DevOps实践详解

### 基础设施即代码（IaC）

```hcl
# Terraform示例 - AWS EC2实例
provider "aws" {
  region = "ap-southeast-1"
}

resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.micro"

  tags = {
    Name        = "web-server"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

resource "aws_security_group" "web" {
  name        = "web-sg"
  description = "Allow HTTP and HTTPS"

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

### 监控与告警配置

```yaml
# Prometheus告警规则
groups:
  - name: application
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }}"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 10m
        labels:
          severity: warning
```

### Grafana仪表盘JSON模型

```json
{
  "dashboard": {
    "title": "CI/CD Pipeline Metrics",
    "panels": [
      {
        "title": "Build Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(ci_build_total{status=\"success\"}[24h])) / sum(rate(ci_build_total[24h])) * 100"
          }
        ]
      },
      {
        "title": "Deployment Frequency",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(deployment_total[1d]))"
          }
        ]
      }
    ]
  }
}
```

## 八、DevOps团队结构

```
传统模式:
  开发团队 → 交付 → 运维团队 → 监控
  (各自独立，交接频繁)

DevOps模式:
  [开发 + 运维 + QA + 安全] → 全生命周期负责
  (跨职能团队，端到端负责)

SRE模式:
  产品团队 ←→ SRE团队
  (开发负责功能，SRE负责可靠性)
```
