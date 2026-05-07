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

## 四、注意事项

1. **Secrets安全**：Secrets不会出现在日志中
2. **分支限制**：可以限制Secrets只能在特定分支使用
3. **环境Secrets**：不同环境可以有不同的Secrets
4. **不要泄露**：确保Secrets不被echo或打印
