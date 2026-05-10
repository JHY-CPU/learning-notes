# 证书体系与PKI


## 证书体系与 PKI


密码学X.509PKI


公钥基础设施（PKI）通过数字证书将公钥与身份绑定，是互联网信任体系的基石。


## X.509 数字证书


```
X.509 v3 证书结构（ASN.1 DER 编码）：
┌─────────────────────────────────────────────────────────┐
│  Certificate ::= SEQUENCE {                             │
│    tbsCertificate       TBSCertificate,  -- 待签证书体   │
│    signatureAlgorithm   AlgorithmIdentifier,            │
│    signatureValue       BIT STRING       -- CA 签名值   │
│  }                                                      │
│                                                         │
│  TBSCertificate ::= SEQUENCE {                          │
│    version         [0] INTEGER,     -- v3 = 2           │
│    serialNumber         INTEGER,     -- CA 唯一序列号    │
│    signature            AlgorithmIdentifier,            │
│    issuer               Name,          -- 颁发者 DN      │
│    validity             SEQUENCE {                      │
│      notBefore  Time,                -- 生效时间         │
│      notAfter   Time                 -- 过期时间         │
│    },                                                   │
│    subject              Name,          -- 主体 DN        │
│    subjectPublicKeyInfo SubjectPublicKeyInfo,            │
│    issuerUniqueID   [1] IMPLICIT UniqueIdentifier,      │
│    subjectUniqueID  [2] IMPLICIT UniqueIdentifier,      │
│    extensions       [3] Extensions   -- v3 扩展字段     │
│  }                                                      │
└─────────────────────────────────────────────────────────┘

关键扩展字段 (Extensions)：
┌─────────────────────────────────────────────────────────┐
│  Subject Alternative Name (SAN)                         │
│  - 证书主体的替代名称（域名列表）                         │
│  - DNS:*.example.com, DNS:example.com                   │
│  - 现代浏览器优先使用 SAN 而非 CN                        │
│                                                         │
│  Key Usage                                              │
│  - digitalSignature: 数字签名                            │
│  - keyEncipherment: 密钥加密                             │
│  - keyCertSign: 签发子证书                               │
│  - cRLSign: 签发 CRL                                    │
│                                                         │
│  Basic Constraints                                      │
│  - CA: TRUE/FALSE（是否为 CA 证书）                      │
│  - pathLenConstraint: 允许的中间 CA 层数                 │
│                                                         │
│  Authority Information Access (AIA)                     │
│  - caIssuers: CA 证书下载 URL                            │
│  - OCSP: OCSP 响应器 URL                                │
│                                                         │
│  CRL Distribution Points                                │
│  - CRL 文件下载地址                                      │
│                                                         │
│  Authority Key Identifier                               │
│  - 签发 CA 的公钥指纹（定位使用哪个 CA 证书）            │
│                                                         │
│  Subject Key Identifier                                 │
│  - 本证书的公钥指纹                                      │
└─────────────────────────────────────────────────────────┘

DN (Distinguished Name) 示例：
CN = www.example.com          -- 通用名
O  = Example Inc              -- 组织名
OU = IT Department            -- 部门
L  = Beijing                  -- 城市
ST = Beijing                  -- 省份
C  = CN                       -- 国家
```


## CA 层次结构与信任链


```
PKI 信任层次：
┌─────────────────────────────────────────────────────────┐
│                     Root CA                              │
│                  (根证书颁发机构)                         │
│  ┌─────────────────────────────────────────┐            │
│  │  - 自签名证书（Issuer = Subject）        │            │
│  │  - 密钥离线保存（HSM / 离线设备）        │            │
│  │  - 有效期 15-25 年                       │            │
│  │  - 预置在操作系统/浏览器信任存储中        │            │
│  │  - 例：DigiCert Root CA                  │            │
│  └───────────────┬─────────────────────────┘            │
│                  │ 签发                                  │
│                  ▼                                       │
│            Intermediate CA                               │
│             (中间证书颁发机构)                             │
│  ┌─────────────────────────────────────────┐            │
│  │  - 由 Root CA 签发                       │            │
│  │  - 在线运行，签发终端证书                 │            │
│  │  - 有效期 1-10 年                        │            │
│  │  - 可以有多个层级（通常 1-2 级）          │            │
│  │  - 例：DigiCert SHA2 Extended Validation │            │
│  └───────────────┬─────────────────────────┘            │
│                  │ 签发                                  │
│                  ▼                                       │
│           End Entity Certificate                         │
│            (终端实体证书)                                 │
│  ┌─────────────────────────────────────────┐            │
│  │  - 由 Intermediate CA 签发               │            │
│  │  - 部署在 Web 服务器 / 邮件服务器等       │            │
│  │  - 有效期 ≤ 398 天（CAB Forum 规定）     │            │
│  │  - 包含域名、公钥、SAN                    │            │
│  │  - 例：*.example.com                     │            │
│  └─────────────────────────────────────────┘            │
│                                                         │
│  为什么使用中间 CA？                                      │
│  - 根 CA 密钥离线保护，减少暴露风险                       │
│  - 中间 CA 被攻击只需吊销中间证书，不影响整个 PKI          │
│  - 灵活的策略管理和地域划分                               │
└─────────────────────────────────────────────────────────┘

证书链验证过程：
┌─────────────────────────────────────────────────────────┐
│  输入：服务器证书 + 中间 CA 证书链                         │
│                                                         │
│  1. 读取服务器证书的 Issuer 字段                          │
│     → 查找对应的中间 CA 证书                              │
│                                                         │
│  2. 验证中间 CA 证书：                                    │
│     a. 检查 Subject 是否匹配服务器证书的 Issuer           │
│     b. 验证签名（用中间 CA 的公钥验签服务器证书）          │
│     c. 检查 BasicConstraints: CA=TRUE                    │
│     d. 检查 KeyUsage: keyCertSign                        │
│     e. 检查有效期                                        │
│     f. 检查是否被吊销                                    │
│                                                         │
│  3. 递归验证到根 CA：                                     │
│     a. 根 CA 证书必须在本地信任存储中                     │
│     b. 根 CA 自签名，用自身公钥验证                       │
│                                                         │
│  4. 路径约束检查：                                        │
│     pathLenConstraint ≥ 实际中间 CA 层数                  │
│                                                         │
│  任何一步失败 → 证书链验证失败 → 连接被拒绝               │
└─────────────────────────────────────────────────────────┘
```


## 证书吊销机制 (CRL / OCSP)


```
为什么需要证书吊销？
- 私钥泄露 → 需要立即吊销证书
- 域名所有权变更
- CA 签发错误证书
- 组织信息变更

方法 1：CRL (Certificate Revocation List)
┌─────────────────────────────────────────────────────────┐
│  CA 定期发布已吊销证书的列表                               │
│                                                         │
│  CRL 结构：                                              │
│  - 签发者 DN                                             │
│  - 本次更新时间 / 下次更新时间                            │
│  - 吊销条目列表：                                        │
│    [serialNumber, revocationDate, reason]               │
│  - CA 签名                                               │
│                                                         │
│  验证流程：                                              │
│  1. 下载 CRL（从 CRL Distribution Point URL）            │
│  2. 验证 CRL 签名                                        │
│  3. 在吊销列表中查找证书序列号                            │
│  4. 找到 = 已吊销 / 未找到 = 有效                        │
│                                                         │
│  缺点：                                                  │
│  - CRL 文件可能很大（数百 KB 到数 MB）                   │
│  - 更新频率有限（数小时到数天）                           │
│  - 在两次更新之间存在窗口期                               │
│  - 客户端需要下载完整 CRL                                │
└─────────────────────────────────────────────────────────┘

方法 2：OCSP (Online Certificate Status Protocol)
┌─────────────────────────────────────────────────────────┐
│  实时查询证书状态（HTTP 请求/响应）                        │
│                                                         │
│  请求：                                                  │
│  POST / HTTP/1.1                                        │
│  Content-Type: application/ocsp-request                 │
│  Body: SEQUENCE {                                       │
│    tbsRequest: {                                        │
│      requestList: [{                                    │
│        reqCert: {                                       │
│          hashAlgorithm: SHA-256                         │
│          issuerNameHash: SHA256(issuerDN)               │
│          issuerKeyHash: SHA256(issuerPublicKey)         │
│          serialNumber: 0x12345678                       │
│        }                                                │
│      }]                                                 │
│    }                                                    │
│  }                                                      │
│                                                         │
│  响应：                                                  │
│  - good:    证书有效                                     │
│  - revoked: 证书已吊销（含吊销时间）                      │
│  - unknown: 未知状态                                     │
│                                                         │
│  优点：实时性好，每次验证都是最新状态                      │
│  缺点：                                                   │
│  - OCSP 响应器可能成为性能瓶颈和单点故障                  │
│  - 暴露用户访问的域名（隐私问题）                         │
│  - 响应有时效性（thisUpdate / nextUpdate）                │
└─────────────────────────────────────────────────────────┘

OCSP Stapling (OCSP 装订)：
┌─────────────────────────────────────────────────────────┐
│  解决 OCSP 的性能和隐私问题                               │
│                                                         │
│  原理：服务器主动获取 OCSP 响应，附加在 TLS 握手中        │
│                                                         │
│  传统 OCSP：                                              │
│  浏览器 → OCSP 响应器（额外请求，暴露访问记录）           │
│                                                         │
│  OCSP Stapling：                                          │
│  服务器 → OCSP 响应器（定期获取 OCSP 响应缓存）           │
│  浏览器 ← 服务器（OCSP 响应随证书一起发送）               │
│                                                         │
│  优势：                                                   │
│  - 客户端不需要单独查询 OCSP                             │
│  - 保护用户隐私（不暴露给 OCSP 响应器）                   │
│  - 减少 OCSP 响应器负载                                  │
│  - 握手不增加额外延迟                                    │
│                                                         │
│  TLS 扩展：status_request (类型 5)                       │
│  ServerHello 中包含 CertificateStatus 消息               │
└─────────────────────────────────────────────────────────┘
```


## Let's Encrypt 与 ACME 协议


```
Let's Encrypt：
┌─────────────────────────────────────────────────────────┐
│  运营方：ISRG (Internet Security Research Group)          │
│  类型：免费、自动化的域名验证证书颁发机构                  │
│  证书类型：DV (Domain Validation)                         │
│  有效期：90 天（自动化续期）                               │
│  市场份额：约 3 亿活跃证书                                │
│  根证书：ISRG Root X1（已内置在主流浏览器和 OS 中）       │
└─────────────────────────────────────────────────────────┘

ACME 协议 (Automatic Certificate Management Environment)：
┌─────────────────────────────────────────────────────────┐
│  ACME 实现了证书申请、验证、签发的全自动化                 │
│                                                         │
│  注册账户：                                               │
│  Client → Server: POST /newAccount                       │
│            { contact: ["mailto:admin@example.com"] }     │
│  Server → Client: accountId, termsOfServiceAgreed        │
│                                                         │
│  申请证书：                                               │
│  Client → Server: POST /newOrder                         │
│            { identifiers: [{type:"dns",                 │
│                              value:"example.com"}] }    │
│  Server → Client: order URL, authorization URLs          │
│                                                         │
│  域名验证（Challenge）：                                   │
│  ┌─────────────────────────────────────────────┐        │
│  │  http-01 验证：                               │        │
│  │  - 在 Web 根目录放置验证文件                   │        │
│  │  - CA 通过 HTTP 访问验证文件                   │        │
│  │  - http://example.com/.well-known/...         │        │
│  │                                               │        │
│  │  dns-01 验证：                                │        │
│  │  - 在 DNS 添加 TXT 记录                       │        │
│  │  - _acme-challenge.example.com TXT "xxx"      │        │
│  │  - 适合通配符证书 (*.example.com)             │        │
│  │                                               │        │
│  │  tls-alpn-01 验证：                           │        │
│  │  - 通过 TLS 握手中的 ALPN 扩展验证            │        │
│  │  - 不需要 HTTP 端口开放                       │        │
│  └─────────────────────────────────────────────┘        │
│                                                         │
│  下载证书：                                               │
│  Client → Server: POST /certificate                      │
│  Server → Client: PEM encoded certificate chain          │
└─────────────────────────────────────────────────────────┘

Certbot（最流行的 ACME 客户端）：
# 安装
sudo apt install certbot

# 自动配置 Nginx
sudo certbot --nginx -d example.com -d www.example.com

# 仅获取证书（手动配置）
sudo certbot certonly --standalone -d example.com

# DNS 验证（通配符证书）
sudo certbot certonly --manual --preferred-challenges dns \
    -d "*.example.com"

# 证书续期（自动）
sudo certbot renew --dry-run
# crontab: 0 3 * * * certbot renew --quiet

其他 ACME 客户端：
- acme.sh：纯 Shell 实现，轻量级
- Caddy：内置 ACME，自动 HTTPS
- Traefik：反向代理，自动证书管理
```


## 证书固定 (Certificate Pinning)


```
证书固定：限制应用只接受特定证书或 CA
┌─────────────────────────────────────────────────────────┐
│  动机：                                                   │
│  - 全球有 100+ 个受信任的根 CA                            │
│  - 任何一个 CA 被攻破或恶意签发都可导致 MITM 攻击          │
│  - 2011 年 DigiNotar 被攻击事件                          │
│  - 证书固定减少信任面                                     │
└─────────────────────────────────────────────────────────┘

方法 1：HTTP Public Key Pinning (HPKP) - 已废弃
┌─────────────────────────────────────────────────────────┐
│  Public-Key-Pins:                                       │
│    pin-sha256="base64==";                               │
│    pin-sha256="base64==";                               │
│    max-age=5184000;                                     │
│    includeSubDomains;                                   │
│    report-uri="https://example.com/hpkp-report"         │
│                                                         │
│  废弃原因：                                               │
│  - 配置错误导致网站不可访问                               │
│  - 被用于 DoS 攻击（恶意固定不存在的密钥）                │
│  - 证书轮换困难                                          │
│  - Chrome 67 已移除 HPKP 支持                            │
└─────────────────────────────────────────────────────────┘

方法 2：Certificate Transparency (CT)
┌─────────────────────────────────────────────────────────┐
│  CA 必须将签发的证书记录到公开的 CT 日志中                 │
│                                                         │
│  工作流程：                                              │
│  1. CA 签发证书 → 提交到 CT 日志                         │
│  2. CT 日志返回 Signed Certificate Timestamp (SCT)       │
│  3. 证书中包含 SCT（或通过 TLS 扩展传输）                │
│  4. 浏览器验证 SCT 的存在和有效性                        │
│                                                         │
│  优势：                                                   │
│  - 所有证书可被公开审计                                   │
│  - 检测 CA 错误/恶意签发                                  │
│  - 域名所有者可以监控自己的证书                           │
│  - Google 的 Merkle Tree 实现保证日志不可篡改            │
│                                                         │
│  现状：Chrome 强制要求 SCT（2018 年起）                   │
└─────────────────────────────────────────────────────────┘

方法 3：应用层证书固定
┌─────────────────────────────────────────────────────────┐
│  Android (Network Security Config)：                      │
│                                 │
│         │
│      example.com│
│                         │
│        base64hash1==        │
│        base64hash2==        │
│                                                │
│                                          │
│                                │
│                                                         │
│  iOS (Info.plist / App Transport Security)：              │
│  固定公钥的 SHA-256 哈希值                               │
│                                                         │
│  注意：                                                   │
│  - 固定 CA 公钥而非终端证书公钥（证书更新更灵活）         │
│  - 始终设置备用 pin（防止单点故障）                       │
│  - 设置合理的过期时间                                     │
└─────────────────────────────────────────────────────────┘
```


> **Note:** HPKP 已被废弃，现代应用推荐使用 Certificate Transparency 进行证书审计，结合应用层的公钥固定策略保护关键域名。


<!-- Converted from: 02_证书体系与PKI.html -->
