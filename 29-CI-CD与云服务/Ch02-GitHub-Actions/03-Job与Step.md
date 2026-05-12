# Job与Step

## 一、概念说明

Job是工作流的基本执行单元，包含多个Step。Job之间可以并行或串行执行。

## 二、Job定义

```yaml
jobs:
  build:
    name: Build Application
    runs-on: ubuntu-latest
    timeout-minutes: 30
    
  test:
    needs: build  # 依赖build job
    runs-on: ubuntu-latest
    
  deploy:
    needs: [build, test]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
```

## 三、Step定义

```yaml
steps:
  # 使用Action
  - uses: actions/checkout@v4
  
  # 运行命令
  - name: Install dependencies
    run: npm install
    
  # 带条件的Step
  - name: Deploy
    if: success()
    run: ./deploy.sh
    
  # 环境变量
  - name: Build
    env:
      NODE_ENV: production
    run: npm run build
```

## 四、Job输出

```yaml
jobs:
  build:
    outputs:
      version: ${{ steps.version.outputs.version }}
    steps:
      - id: version
        run: echo "version=1.0.0" >> $GITHUB_OUTPUT
        
  deploy:
    needs: build
    steps:
      - run: echo "Deploying version ${{ needs.build.outputs.version }}"
```

## 五、注意事项

1. **Job独立**：每个Job在独立的Runner上运行
2. **Step顺序**：Step按顺序执行
3. **Artifact共享**：Job之间通过Artifact共享数据
4. **超时设置**：设置合理的超时时间

## 六、Job容器和服务

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    container:
      image: node:20-alpine
      env:
        NODE_ENV: test
      volumes:
        - /tmp:/tmp
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
    steps:
      - run: node -v
      - run: |
          PGPASSWORD=postgres psql -h postgres -U postgres -c "SELECT 1"
```

## 七、Step高级用法

```yaml
steps:
  # continue-on-error允许失败继续
  - name: Optional step
    continue-on-error: true
    run: exit 1

  # timeout-minutes单步超时
  - name: Long running
    timeout-minutes: 30
    run: ./long-task.sh

  # working-directory指定工作目录
  - name: Build in subdir
    working-directory: ./packages/app
    run: npm run build

  # shell指定shell类型
  - name: PowerShell
    shell: pwsh
    run: Get-Process

  # 多行脚本
  - name: Multi-line
    run: |
      echo "Line 1"
      echo "Line 2"
      if [ "$STATUS" = "success" ]; then
        echo "Done"
      fi
```

## 八、GITHUB_OUTPUT和GITHUB_STATE

```yaml
steps:
  - id: version
    name: Get version
    run: |
      VERSION=$(cat package.json | jq -r .version)
      echo "version=$VERSION" >> $GITHUB_OUTPUT
      echo "major=$(echo $VERSION | cut -d. -f1)" >> $GITHUB_OUTPUT

  - name: Use version
    run: |
      echo "Version: ${{ steps.version.outputs.version }}"
      echo "Major: ${{ steps.version.outputs.major }}"
```

## 九、Job策略与重试

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      max-parallel: 3
      matrix:
        node: [18, 20]
        os: [ubuntu-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4

  # 重试机制
  flaky-test:
    runs-on: ubuntu-latest
    steps:
      - uses: nick-fields/retry@v2
        with:
          max_attempts: 3
          timeout_minutes: 10
          command: npm test
```
