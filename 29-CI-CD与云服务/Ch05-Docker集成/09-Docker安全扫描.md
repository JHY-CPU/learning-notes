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
