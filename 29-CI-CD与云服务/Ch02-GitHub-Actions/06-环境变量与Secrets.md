# 环境变量与Secrets

## 一、环境变量

```yaml
# 全局环境变量
env:
  GLOBAL_VAR: global_value

jobs:
  build:
    env:
      JOB_VAR: job_value
    steps:
      - name: Use env
        run: echo $GLOBAL_VAR $JOB_VAR
        env:
          STEP_VAR: step_value
```

## 二、Secrets

```yaml
# 在Settings > Secrets中配置
steps:
  - run: echo ${{ secrets.API_KEY }}
  - run: echo ${{ secrets.DATABASE_URL }}

# 环境Secrets
environment: production
steps:
  - run: echo ${{ secrets.PROD_API_KEY }}
```

## 三、动态环境变量

```yaml
steps:
  - name: Set dynamic var
    run: echo "VERSION=1.0.0" >> $GITHUB_ENV
  
  - name: Use dynamic var
    run: echo $VERSION
```

## 四、最佳实践

1. **最小权限原则**：只给Job分配必要的Secrets，避免全局暴露
2. **环境隔离**：生产Secrets绑定到production环境，需要审批才能使用
3. **命名规范**：使用大写下划线命名（`DATABASE_URL`），与环境变量惯例一致
4. **定期轮换**：定期更换API密钥等敏感信息，防止泄露扩大
5. **使用Dependabot**：监控Actions版本，及时更新安全补丁

## 五、常见陷阱

1. **意外泄露**：将Secrets赋值给普通环境变量后被echo输出，Secrets保护仅对`${{ secrets.X }}`直接引用生效
2. **Fork仓库泄露**：PR事件中Fork仓库的贡献者可能通过恶意脚本窃取Secrets，建议限制PR事件的Secrets访问
3. **日志打印**：`printenv`或`env`命令会暴露所有非Secret环境变量，间接泄露敏感配置
4. **缓存污染**：缓存中不要包含Secrets内容，缓存可能被其他工作流访问

## 六、组织级Secrets与Environments

```yaml
# 组织级Secrets适用于所有仓库
# Settings > Secrets and variables > Organization

# Environment Secrets限定特定环境
jobs:
  deploy:
    environment: production
    steps:
      - run: echo "使用 ${{ secrets.PROD_API_KEY }}"
      # PROD_API_KEY仅在production environment可用

# 环境保护规则
# Settings > Environments > production
# - Required reviewers: [admin1, admin2]
# - Wait timer: 5 minutes
# - Deployment branches: main only
```

## 七、OIDC认证（无需存储长期凭据）

```yaml
# AWS OIDC认证
jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789:role/github-actions
          aws-region: us-east-1
      - run: aws s3 sync ./dist s3://my-bucket

# Azure OIDC认证
      - uses: azure/login@v1
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}

# GCP Workload Identity
      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.WIF_SERVICE_ACCOUNT }}
```

## 八、变量与Secrets的生命周期

```
变量作用域（从低到高）:
  1. Step级 env:
  2. Job级 env:
  3. Workflow级 env:
  4. Repository变量 (Settings > Variables)
  5. Environment变量 (Settings > Environments)
  6. Organization变量 (Settings > Org > Variables)

Secrets作用域:
  1. Repository Secrets (Settings > Secrets)
  2. Environment Secrets (Settings > Environments)
  3. Organization Secrets (Settings > Org > Secrets)

优先级: 高优先级覆盖低优先级
```

## 九、配置文件自动处理

```yaml
# 使用envsubst替换配置文件中的变量
- name: Generate config
  env:
    DB_HOST: ${{ secrets.DB_HOST }}
    DB_PORT: ${{ secrets.DB_PORT }}
  run: |
    envsubst < config.template.yml > config.yml

# 使用SOPS加密配置文件
- name: Decrypt config
  uses: mozilla/sops@v1
  with:
    args: --decrypt --in-place config.enc.yml
```
