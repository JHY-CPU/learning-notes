# 14-安全开发（SDL）与DevSecOps

## 一、安全开发生命周期（SDL）

### 1.1 微软SDL框架

```
SDL七个阶段：
┌──────────────────────────────────────────────┐
│ 1. 培训（Training）                           │
│    安全培训、安全意识教育                       │
├──────────────────────────────────────────────┤
│ 2. 需求（Requirements）                       │
│    安全需求分析、质量标准/bug栏                 │
├──────────────────────────────────────────────┤
│ 3. 设计（Design）                             │
│    威胁建模、攻击面分析、设计评审               │
├──────────────────────────────────────────────┤
│ 4. 实施（Implementation）                     │
│    安全编码、静态分析、代码审查                 │
├──────────────────────────────────────────────┤
│ 5. 验证（Verification）                       │
│    动态分析、模糊测试、渗透测试                 │
├──────────────────────────────────────────────┤
│ 6. 发布（Release）                            │
│    安全审查、事件响应计划、发布存档             │
├──────────────────────────────────────────────┤
│ 7. 响应（Response）                           │
│    漏洞响应、安全更新、事后分析                 │
└──────────────────────────────────────────────┘
```

### 1.2 威胁建模（STRIDE）

```
STRIDE威胁分类：
┌──────────────┬────────────────────────────┐
│ 威胁          │ 违反的安全属性              │
├──────────────┼────────────────────────────┤
│ Spoofing     │ 认证性（仿冒）              │
│ Tampering    │ 完整性（篡改）              │
│ Repudiation  │ 不可否认性（抵赖）          │
│ Info Disclosure │ 机密性（信息泄露）        │
│ Denial of Service │ 可用性（拒绝服务）      │
│ Elevation of Privilege │ 授权（权限提升） │
└──────────────┴────────────────────────────┘

威胁建模步骤：
1. 绘制数据流图（DFD）
2. 识别信任边界
3. 对每个元素应用STRIDE
4. 评估风险等级
5. 确定缓解措施
```

### 1.3 攻击面分析

```
攻击面识别要素：
├── 入口点（Entry Points）
│   ├── 用户输入（表单、API参数）
│   ├── 文件上传
│   ├── 网络端口
│   └── 外部接口
├── 可信边界（Trust Boundaries）
│   ├── 网络边界
│   ├── 进程边界
│   └── 用户/系统边界
├── 数据流（Data Flows）
│   ├── 内部数据流
│   ├── 外部数据流
│   └── 跨信任边界数据流
└── 资产（Assets）
    ├── 敏感数据
    ├── 系统资源
    └── 凭据
```

---

## 二、DevSecOps

### 2.1 DevSecOps理念

```
DevSecOps = Development + Security + Operations

核心理念：
"安全左移"（Shift Left）- 将安全集成到开发流程的早期

传统模式：开发 → 测试 → 安全审查 → 部署（安全在最后）
DevSecOps：开发+安全 → 测试+安全 → 部署+安全（安全贯穿全程）

目标：
├── 自动化安全检查
├── 快速反馈安全问题
├── 安全即代码
├── 持续安全监控
└── 共同安全责任
```

### 2.2 DevSecOps流水线

```
CI/CD安全集成：

代码提交阶段：
├── Pre-commit Hooks
│   ├── 密钥检测（git-secrets, truffleHog）
│   ├── 代码格式检查
│   └── 敏感信息扫描
└── IDE安全插件
    ├── SonarLint
    └── Snyk IDE插件

构建阶段（CI）：
├── SAST（静态应用安全测试）
│   ├── SonarQube
│   ├── Checkmarx
│   ├── Fortify
│   └── Semgrep
├── SCA（软件成分分析）
│   ├── OWASP Dependency-Check
│   ├── Snyk
│   ├── Black Duck
│   └── Trivy
├── 容器镜像扫描
│   ├── Trivy
│   ├── Grype
│   ├── Clair
│   └── Anchore
└── IaC扫描
    ├── Checkov
    ├── tfsec
    └── Terrascan

测试阶段：
├── DAST（动态应用安全测试）
│   ├── OWASP ZAP
│   ├── Burp Suite Enterprise
│   └── Acunetix
├── IAST（交互式安全测试）
│   ├── Contrast Security
│   └── Seeker
├── 模糊测试
│   └── AFL, libFuzzer
└── API安全测试
    └── Postman + 安全测试用例

部署阶段：
├── 配置审计
├── 运行时保护（RASP）
├── WAF部署
└── 密钥和凭据管理

运行阶段：
├── 运行时监控
├── RASP（运行时应用自我保护）
├── SIEM集成
├── 漏洞持续监控
└── 威胁情报集成
```

### 2.3 CI/CD安全配置示例

```yaml
# GitLab CI/CD 安全扫描示例
stages:
  - build
  - test
  - security
  - deploy

# SAST - 静态代码分析
sast:
  stage: security
  script:
    - semgrep --config auto --json -o sast-report.json .
  artifacts:
    reports:
      sast: sast-report.json

# 依赖漏洞扫描
dependency-check:
  stage: security
  script:
    - dependency-check --project myapp --scan . --format JSON --out dc-report.json
  artifacts:
    reports:
      dependency_scanning: dc-report.json

# 容器镜像扫描
container-scan:
  stage: security
  script:
    - trivy image --format json -o trivy-report.json myapp:latest
  artifacts:
    reports:
      container_scanning: trivy-report.json

# DAST - 动态扫描
dast:
  stage: security
  script:
    - zap-baseline.py -t http://staging.myapp.com -r dast-report.html
  artifacts:
    reports:
      dast: dast-report.html

# 密钥检测
secret-detection:
  stage: security
  script:
    - trufflehog filesystem --directory . --json > secrets-report.json
  artifacts:
    reports:
      secret_detection: secrets-report.json
```

---

## 三、安全编码实践

### 3.1 输入验证

```
输入验证原则：
├── 所有外部输入必须验证
├── 白名单优于黑名单
├── 验证在信任边界进行
├── 服务端验证为主，客户端验证为辅
├── 验证数据类型、长度、范围、格式
└── 拒绝不合规输入，不尝试修正

验证类型：
├── 类型检查 - 确保数据类型正确
├── 长度检查 - 不超过允许长度
├── 范围检查 - 数值在合理范围内
├── 格式检查 - 邮箱、电话等格式
├── 枚举检查 - 仅允许预定义值
└── 业务规则检查 - 符合业务逻辑
```

### 3.2 安全编码检查清单

```python
# Python安全编码示例

# ❌ SQL注入
query = f"SELECT * FROM users WHERE id = {user_id}"

# ✅ 参数化查询
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))

# ❌ 命令注入
os.system(f"ping {host}")

# ✅ 安全执行
subprocess.run(["ping", "-c", "4", host], check=True)

# ❌ 路径遍历
file_path = os.path.join(base_dir, user_input)

# ✅ 路径验证
file_path = os.path.realpath(os.path.join(base_dir, user_input))
if not file_path.startswith(os.path.realpath(base_dir)):
    raise ValueError("Invalid path")

# ❌ 硬编码密钥
API_KEY = "sk-1234567890abcdef"

# ✅ 环境变量
API_KEY = os.environ.get("API_KEY")

# ❌ 不安全的反序列化
data = pickle.loads(user_input)

# ✅ 安全格式
data = json.loads(user_input)

# ❌ 敏感信息日志
logger.info(f"User login: {username}, password: {password}")

# ✅ 安全日志
logger.info(f"User login attempt: {username}")
```

### 3.3 安全API设计

```
REST API安全设计：
├── 认证
│   ├── OAuth 2.0 / OpenID Connect
│   ├── JWT令牌
│   ├── API密钥（用于服务间通信）
│   └── mTLS（双向TLS认证）
├── 授权
│   ├── 基于角色的访问控制
│   ├── 基于属性的访问控制
│   └── 资源级别的权限检查
├── 输入验证
│   ├── 请求体验证（JSON Schema）
│   ├── 参数验证
│   └── 内容类型检查
├── 速率限制
│   ├── 基于IP的限流
│   ├── 基于用户的限流
│   └── 分级限流
├── 数据保护
│   ├── HTTPS强制
│   ├── 敏感字段加密
│   └── 响应数据脱敏
├── 错误处理
│   ├── 统一错误响应格式
│   ├── 不泄露内部信息
│   └── 适当的HTTP状态码
└── 日志审计
    ├── 记录所有API调用
    ├── 记录认证失败
    └── 异常行为告警
```

---

## 四、软件成分分析（SCA）

### 4.1 SBOM（软件物料清单）

```
SBOM格式：
├── CycloneDX - OWASP项目，JSON/XML格式
├── SPDX - Linux基金会标准
└── SWID - ISO/IEC 19770-2

SBOM内容：
├── 组件名称和版本
├── 供应商/来源
├── 许可证信息
├── 依赖关系
├── 哈希值
└── 已知漏洞
```

### 4.2 依赖漏洞管理

```bash
# npm审计
npm audit
npm audit fix

# Python安全检查
pip-audit
safety check

# Java依赖检查
mvn dependency-check:check
gradle dependencyCheckAnalyze

# Go依赖检查
govulncheck ./...

# 综合工具
trivy fs .
grype dir:.
snyk test
```

---

## 五、安全测试自动化

### 5.1 安全测试金字塔

```
           ╱╲
          ╱  ╲        渗透测试（手动/低频）
         ╱────╲
        ╱      ╲      DAST（动态扫描）
       ╱────────╲
      ╱          ╲    IAST（交互式测试）
     ╱────────────╲
    ╱              ╲  SAST + SCA（静态扫描）
   ╱────────────────╲
  ╱                  ╲ 单元测试（安全测试用例）
 ╱────────────────────╲

建议比例：
- 安全单元测试：70%（快速、低成本）
- SAST/SCA：20%（中等成本）
- DAST/渗透测试：10%（高成本、高价值）
```

### 5.2 安全测试用例示例

```python
# 安全单元测试
import pytest

def test_sql_injection_prevention():
    """验证SQL注入防护"""
    malicious_input = "'; DROP TABLE users; --"
    result = search_users(malicious_input)
    assert result is not None  # 不应崩溃
    assert verify_no_sql_injection(malicious_input)

def test_xss_prevention():
    """验证XSS防护"""
    xss_payload = '<script>alert("xss")</script>'
    sanitized = sanitize_html(xss_payload)
    assert '<script>' not in sanitized

def test_path_traversal_prevention():
    """验证路径遍历防护"""
    traversal_path = "../../../etc/passwd"
    with pytest.raises(ValueError):
        read_file(traversal_path)

def test_authorization_check():
    """验证授权检查"""
    # 普通用户不应访问管理员功能
    with pytest.raises(PermissionDenied):
        delete_user(admin_token=regular_user_token)

def test_rate_limiting():
    """验证速率限制"""
    for _ in range(101):
        response = api_call()
    assert response.status_code == 429  # Too Many Requests
```

---

## 六、安全度量指标

### 6.1 安全开发指标

| 指标 | 计算方式 | 目标 |
|------|----------|------|
| 漏洞密度 | 漏洞数/千行代码 | <1 |
| 安全缺陷修复时间 | 从发现到修复的平均时间 | 严重<7天 |
| 安全测试覆盖率 | 安全测试覆盖的功能比例 | >80% |
| 依赖漏洞数 | 存在已知漏洞的依赖数 | 0高危 |
| 密钥泄露数 | 代码中发现的硬编码密钥 | 0 |
| 安全门禁通过率 | 通过安全检查的构建比例 | >95% |

### 6.2 安全运营指标

| 指标 | 说明 |
|------|------|
| MTTD | 平均检测时间 |
| MTTR | 平均响应时间 |
| 漏洞修复率 | 已修复/发现的漏洞总数 |
| 安全事件数量 | 按类型和严重程度统计 |
| 安全培训完成率 | 员工培训覆盖率 |
| 钓鱼模拟点击率 | 模拟钓鱼的点击比例 |

---

## 七、供应链安全

### 7.1 软件供应链威胁

```
供应链攻击类型：
├── 依赖混淆 - 利用包管理器优先级
├── 恶意包注入 - 在公共仓库投毒
├── 劫持维护者 - 获取包维护权限
├── 构建系统攻击 - 篡改CI/CD
├── 源代码攻击 - 注入后门到源码
└── 更新劫持 - 篡改合法更新

知名事件：
- SolarWinds（2020）- 构建系统被入侵
- Log4j（2021）- 广泛使用的开源库漏洞
- ua-parser-js（2021）- npm包被劫持
- codecov（2021）- CI工具被篡改
```

### 7.2 供应链安全措施

```
防御措施：
├── SBOM管理 - 清晰了解所有组件
├── 依赖锁定 - 使用lock文件
├── 私有仓库 - 使用Artifactory/Nexus
├── 签名验证 - 验证包的签名
├── 漏洞监控 - 持续监控依赖漏洞
├── 最小依赖 - 减少不必要的依赖
├── 版本固定 - 避免自动升级到最新版
├── 代码审查 - 审查依赖变更
└── SLSA框架 - 供应链安全等级
```

### 7.3 SLSA框架

```
SLSA（Supply chain Levels for Software Artifacts）：

Level 0 - 无保证
Level 1 - 构建过程文档化
Level 2 - 构建系统有版本控制
Level 3 - 构建系统有安全保障
Level 4 - 最高级别，所有更改需两人审查

关键要求：
├── 源代码完整性
├── 构建过程不可伪造
├── 证明可验证
└── 依赖可追溯
```
