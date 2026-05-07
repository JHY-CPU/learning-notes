# CI-CD安全概览

## 一、概念说明

CI/CD安全是将安全实践集成到持续集成和持续部署流程中，确保从代码提交到生产部署的每个环节都经过安全检查。也称为DevSecOps或Shift Left Security。

| 阶段 | 安全实践 | 工具 |
|------|----------|------|
| 代码编写 | 依赖安全、密钥扫描 | Snyk、GitLeaks |
| 提交前 | SAST静态分析 | SonarQube、Semgrep |
| 构建阶段 | 镜像扫描、SCA | Trivy、Snyk |
| 部署前 | DAST动态扫描 | OWASP ZAP |
| 运行时 | 运行时保护 | Falco、OPA |

## 二、具体用法

### DevSecOps流水线

```yaml
# GitHub Actions安全流水线
name: Secure CI/CD Pipeline
on: [push, pull_request]

jobs:
  # 1. 代码安全扫描
  sast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: SonarQube Scan
        uses: sonarqube-quality-gate-action@master
      - name: Semgrep SAST
        uses: returntocorp/semgrep-action@v1
        with:
          config: p/ci

  # 2. 依赖漏洞扫描
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Snyk dependency scan
        uses: snyk/actions/node@master
        with:
          args: --severity-threshold=high

  # 3. 密钥泄露检测
  secret-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: GitLeaks scan
        uses: gitleaks/gitleaks-action@v2

  # 4. 镜像安全扫描
  image-scan:
    needs: [sast, dependency-scan]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .
      - name: Trivy scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          severity: CRITICAL,HIGH
          exit-code: 1

  # 5. 部署（仅在所有安全检查通过后）
  deploy:
    needs: [sast, dependency-scan, secret-scan, image-scan]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: echo "Deploying..."
```

### 安全门禁

```yaml
# 质量门禁配置
Quality_Gates:
  代码质量:
    - 代码覆盖率 > 80%
    - 无Blocker级别问题
    - 无Critical级别问题
  安全扫描:
    - 无Critical漏洞
    - High漏洞数量 < 5
    - 无已知高危CVE
  依赖安全:
    - 无已知高危依赖漏洞
    - 许可证合规
  镜像安全:
    - 基础镜像为最新安全版本
    - 无敏感文件
    - 最小化镜像
```

### 安全策略配置

```yaml
# .github/security.yml
security:
  # 依赖扫描
  dependabot:
    enabled: true
    schedule: weekly
    severity_threshold: high

  # 代码扫描
  code_scanning:
    enabled: true
    tools:
      - semgrep
      - codeql

  # 密钥扫描
  secret_scanning:
    enabled: true
    push_protection: true
```

## 三、注意事项与常见陷阱

1. **Shift Left**：尽早发现安全问题，修复成本最低
2. **自动化**：安全检查必须自动化，不能依赖人工
3. **门禁严格**：Critical级别问题必须阻断部署
4. **快速反馈**：安全扫描要快速，避免影响开发效率
5. **持续更新**：安全规则和漏洞库需要定期更新
6. **团队培训**：定期进行安全开发培训
7. **应急响应**：建立安全漏洞应急响应流程
