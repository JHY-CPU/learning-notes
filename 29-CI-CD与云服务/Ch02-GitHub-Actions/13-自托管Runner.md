# 自托管Runner

## 一、概念说明

自托管Runner是在自己服务器上运行的GitHub Actions运行器，用于私有网络或特殊环境。

## 二、安装Runner

```bash
# 下载Runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# 配置
./config.sh --url https://github.com/your-org --token YOUR_TOKEN

# 运行
./run.sh

# 或作为服务安装
sudo ./svc.sh install
sudo ./svc.sh start
```

## 三、使用自托管Runner

```yaml
jobs:
  build:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - run: npm install
      - run: npm test
```

## 四、Runner标签

```yaml
jobs:
  build:
    runs-on: [self-hosted, linux, x64]
```

## 五、注意事项

1. **安全风险**：自托管Runner可能有安全风险
2. **自动更新**：开启Runner自动更新
3. **网络访问**：确保Runner能访问GitHub
4. **资源管理**：监控Runner资源使用
