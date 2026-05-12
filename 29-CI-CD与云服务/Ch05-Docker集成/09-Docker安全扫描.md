# Docker安全扫描

## 一、Trivy扫描

```bash
# 安装Trivy
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh

# 扫描镜像
trivy image myapp:latest

# CI集成
- name: Run Trivy
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'myapp:latest'
    format: 'sarif'
    output: 'trivy-results.sarif'
    severity: 'CRITICAL,HIGH'
```

## 二、Snyk扫描

```yaml
- name: Snyk Scan
  uses: snyk/actions/docker@master
  with:
    image: 'myapp:latest'
    args: --severity-threshold=high
  env:
    SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
```

## 三、Docker Scout

```bash
# Docker Scout
docker scout cves myapp:latest
docker scout quickview myapp:latest
```

## 四、扫描门禁

```yaml
scan:
  script:
    - trivy image --severity CRITICAL --exit-code 1 myapp:latest
  allow_failure: false
```

## 五、注意事项

1. **及时更新**：定期更新扫描工具
2. **门禁配置**：严重漏洞阻止部署
3. **误报管理**：标记和管理误报
4. **基线镜像**：选择安全的基础镜像

## 六、Grype安全扫描

```yaml
- name: Grype scan
  uses: anchore/scan-action@v3
  id: grype
  with:
    image: myapp:latest
    severity-cutoff: high
    fail-build: true
    output-format: sarif

- name: Upload Grype results
  uses: github/codeql-action/upload-sarif@v3
  if: always()
  with:
    sarif_file: ${{ steps.grype.outputs.sarif }}
```

## 七、Docker Scout分析

```yaml
- name: Docker Scout
  uses: docker/scout-action@v1
  with:
    command: cves
    image: myapp:latest
    sarif-file: scout-results.sarif
    only-severities: critical,high
    exit-code: true
```

## 八、多工具对比

```
┌────────────────┬──────────────┬──────────────┬──────────────┐
│ 工具            │ 速度         │ 准确度       │ 集成度       │
├────────────────┼──────────────┼──────────────┼──────────────┤
│ Trivy          │ 快           │ 高           │ 高 (GitLab)  │
│ Grype          │ 快           │ 高           │ 中           │
│ Snyk           │ 中           │ 最高         │ 高           │
│ Docker Scout   │ 快           │ 高           │ 最高 (Docker)│
│ Clair          │ 中           │ 高           │ 中           │
│ Checkov        │ 快           │ 高 (IaC)     │ 中           │
└────────────────┴──────────────┴──────────────┴──────────────┘
```

## 九、漏洞管理流程

```yaml
# 完整的安全扫描流水线
stages:
  - build
  - scan
  - gate
  - deploy

build:
  stage: build
  script:
    - docker build -t $IMAGE .

scan-trivy:
  stage: scan
  script:
    - trivy image --format json --output trivy.json $IMAGE
    - trivy image --format table $IMAGE
  artifacts:
    reports:
      container_scanning: trivy.json

scan-snyk:
  stage: scan
  script:
    - snyk container test $IMAGE --json > snyk.json
  artifacts:
    reports:
      container_scanning: snyk.json
  allow_failure: true

security-gate:
  stage: gate
  script:
    - |
      CRITICAL=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity=="CRITICAL")] | length' trivy.json)
      HIGH=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity=="HIGH")] | length' trivy.json)
      echo "Critical: $CRITICAL, High: $HIGH"
      if [ "$CRITICAL" -gt 0 ]; then
        echo "ERROR: Found $CRITICAL critical vulnerabilities"
        exit 1
      fi
      if [ "$HIGH" -gt 5 ]; then
        echo "ERROR: Found $HIGH high vulnerabilities (max 5)"
        exit 1
      fi
  needs:
    - scan-trivy
```

## 十、漏洞抑制配置

```yaml
# Trivy忽略文件 - .trivyignore
# 忽略已评估的CVE
CVE-2023-12345  # 已评估，不影响生产环境
CVE-2023-67890  # 不使用该功能

# Trivy配置文件 - trivy.yaml
severity:
  - CRITICAL
  - HIGH

vulnerability:
  ignore-unfixed: true  # 忽略无修复的漏洞

scan:
  scanners:
    - vuln
    - secret
```
