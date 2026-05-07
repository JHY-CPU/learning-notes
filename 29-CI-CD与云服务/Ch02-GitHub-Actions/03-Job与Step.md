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
