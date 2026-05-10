# 基础设施即代码(IaC) — 面试指南 - 即答侠

URL: https://interviewasssistant.com/zh/interview-questions/devops-infrastructure-as-code

[DevOps 运维面试题](https://interviewasssistant.com/zh/interview-questions/category/devops)

# 基础设施即代码(IaC) — 面试指南

"什么是基础设施即代码？你如何使用它？"

## 为什么面试官会问这道题

IaC对可重现、可审计的基础设施至关重要。测试你能否以与应用代码同样的严谨性管理基础设施。

## 如何回答

1. 1
定义IaC并解释优势：版本控制、可重现性、自动化。

2. 2
讨论你用过的工具。

3. 3
覆盖最佳实践：状态管理、模块化、测试。

4. 4
分享实际例子。


## 参考回答示例

基础设施即代码是通过声明式配置文件而非手动操作来定义基础设施。这带来版本控制、同行评审、可重现性和自动化。我主要使用Terraform，远程状态存在S3，DynamoDB做锁。代码组织成可复用模块——VPC模块、ECS服务模块、数据库模块——各有自己的测试。所有变更通过PR评审和plan输出后才apply。我还用Terragrunt管理多环境减少重复。我将整个AWS基础设施从手动操作迁移到Terraform——跨3个环境150+资源。环境搭建时间从2天降到15分钟，消除了配置漂移。

## 实用技巧

- 提到状态管理——这是Terraform最棘手的部分。

- 讨论漂移检测和修复。

- 展示你用与应用代码同等的质量标准对待基础设施代码。


## 常见问题

### Terraform还是CloudFormation？

Terraform云无关且社区更大。CloudFormation是AWS原生集成更好。100%AWS用CloudFormation；多云或要灵活性用Terraform。

### 如何测试IaC？

静态分析（tflint、checkov）、plan评审、部署到临时环境的集成测试、策略即代码（OPA/Sentinel）做合规检查。

### 如何在IaC中处理密钥？

永远不要在IaC文件中存储密钥。使用AWS Secrets Manager、HashiCorp Vault或SOPS。通过ARN或路径引用密钥，而非值。

## 更多DevOps 运维面试题

[CI/CD流水线设计 — 面试指南](https://interviewasssistant.com/zh/interview-questions/devops-cicd-pipeline-design) [Docker vs Kubernetes — 面试指南](https://interviewasssistant.com/zh/interview-questions/devops-docker-vs-kubernetes) [监控与告警策略 — 面试指南](https://interviewasssistant.com/zh/interview-questions/devops-monitoring-alerting) [故障响应流程 — 面试指南](https://interviewasssistant.com/zh/interview-questions/devops-incident-response) [Kubernetes Pod生命周期与探针](https://interviewasssistant.com/zh/interview-questions/devops-k8s-pod-lifecycle) [蓝绿、金丝雀、滚动发版对比](https://interviewasssistant.com/zh/interview-questions/devops-blue-green-canary)

[查看全部DevOps 运维面试题](https://interviewasssistant.com/zh/interview-questions/category/devops)

## 面试时担心忘词？即答侠实时助你

即答侠 AI 实时监听面试对话，自动识别问题并即时生成回答建议——无感辅助，让你从容应对每一道题。

[免费试用即答侠](https://interviewasssistant.com/dashboard)