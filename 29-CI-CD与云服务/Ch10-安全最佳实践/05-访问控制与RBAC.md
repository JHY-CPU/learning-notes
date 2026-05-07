# 访问控制与RBAC

## 一、概念说明

访问控制确保只有授权的用户和服务可以访问特定资源。RBAC（Role-Based Access Control）是基于角色的访问控制模型，是Kubernetes和大多数云平台的标准权限模型。

| 概念 | 说明 |
|------|------|
| Subject | 用户、组、ServiceAccount |
| Role | 一组权限的集合 |
| RoleBinding | 将Role绑定到Subject |
| ClusterRole | 集群级别的Role |
| ClusterRoleBinding | 集群级别的绑定 |

## 二、具体用法

### Kubernetes RBAC

```yaml
# 开发者角色 - 仅限dev namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: developer
  namespace: dev
rules:
- apiGroups: ["", "apps", "batch"]
  resources: ["pods", "deployments", "services", "jobs"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods/log", "pods/exec"]
  verbs: ["get", "create"]
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list"]  # 仅读取Secrets

---
# 开发者绑定
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: developer-binding
  namespace: dev
subjects:
- kind: User
  name: dev@example.com
  apiGroup: rbac.authorization.k8s.io
- kind: Group
  name: developers
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: developer
  apiGroup: rbac.authorization.k8s.io

---
# 运维人员角色 - 集群级别
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: sre-admin
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list", "create", "update", "delete"]

---
# 只读角色
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: readonly
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["secrets"]
  verbs: []  # 不能访问Secrets
```

### ServiceAccount权限

```yaml
# CI/CD ServiceAccount
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ci-deployer
  namespace: production
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/ci-deployer

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ci-deployer-role
  namespace: production
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "update", "patch"]
- apiGroups: [""]
  resources: ["services", "configmaps"]
  verbs: ["get", "list", "create", "update"]
- apiGroups: [""]
  resources: ["secrets"]
  verbs: []  # CI不需要访问Secrets
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ci-deployer-binding
  namespace: production
subjects:
- kind: ServiceAccount
  name: ci-deployer
  namespace: production
roleRef:
  kind: Role
  name: ci-deployer-role
  apiGroup: rbac.authorization.k8s.io
```

### OPA/Rego访问控制

```rego
# policy.rego
package kubernetes.admission

# 禁止在production namespace创建特权Pod
deny[msg] {
    input.request.kind.kind == "Pod"
    input.request.namespace == "production"
    container := input.request.object.spec.containers[_]
    container.securityContext.privileged == true
    msg := sprintf("Privileged containers not allowed in production: %v", [container.name])
}

# 要求所有Pod有资源限制
deny[msg] {
    input.request.kind.kind == "Pod"
    container := input.request.object.spec.containers[_]
    not container.resources.limits.cpu
    msg := sprintf("Container %v must have CPU limits", [container.name])
}

deny[msg] {
    input.request.kind.kind == "Pod"
    container := input.request.object.spec.containers[_]
    not container.resources.limits.memory
    msg := sprintf("Container %v must have memory limits", [container.name])
}

# 禁止latest标签
deny[msg] {
    input.request.kind.kind in ["Deployment", "StatefulSet", "DaemonSet"]
    container := input.request.object.spec.template.spec.containers[_]
    endswith(container.image, ":latest")
    msg := sprintf("Container %v must not use latest tag", [container.name])
}

# 只允许从受信仓库拉取镜像
deny[msg] {
    input.request.kind.kind == "Pod"
    container := input.request.object.spec.containers[_]
    not startswith(container.image, "registry.example.com/")
    not startswith(container.image, "docker.io/library/")
    msg := sprintf("Image %v not from trusted registry", [container.image])
}
```

### GitHub Actions权限

```yaml
# 最小权限原则
name: Deploy
on: push
permissions:
  contents: read
  packages: write
  id-token: write  # OIDC

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          # 使用最小的检出范围
          sparse-checkout: |
            app
            k8s

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-actions
          aws-region: us-east-1
```

## 三、注意事项与常见陷阱

1. **最小权限**：每个主体只授予完成任务所需的最小权限
2. **角色分离**：开发、测试、生产使用不同角色
3. **ServiceAccount**：每个应用使用独立的ServiceAccount
4. **权限审计**：定期审查权限分配
5. **禁止ClusterAdmin**：避免给普通用户cluster-admin权限
6. **命名空间隔离**：使用Namespace隔离不同团队/环境
7. **权限继承**：注意ClusterRole和Role的作用域区别
