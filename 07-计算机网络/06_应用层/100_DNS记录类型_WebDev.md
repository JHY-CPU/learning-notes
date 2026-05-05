# DNS记录类型

## 📋 DNS 记录类型

A/AAAA/CNAME/MX/TXT/NS/SOA/PTR/SRV 记录详解、使用场景、配置示例。

## 核心 DNS 记录类型
```
// ========== A 记录 (Address) ==========
// 域名 → IPv4 地址
// 最基础的 DNS 记录
//
// example.com.  300  IN  A  93.184.216.34
// ───────┬────── ─┬─ ─┬─ ───────┬───────
//  域名   TTL     类  型    IP 地址
//
// 配置示例:
//   example.com     A     93.184.216.34
//   www.example.com A    93.184.216.34
//   api.example.com A    93.184.216.35
//
// 负载均衡: 多个 A 记录同域名
//   example.com  A  203.0.113.1
//   example.com  A  203.0.113.2
//   example.com  A  203.0.113.3
//   客户端随机选择一个 IP

// ========== AAAA 记录 ==========
// 域名 → IPv6 地址
// 格式同 A 记录,但值不同
//
// example.com.  300  IN  AAAA  2001:db8:85a3::8a2e:370:7334

// ========== CNAME 记录 (Canonical Name) ==========
// 域名 → 另一个域名 (别名)
// CNAME 不能与其他记录类型共存
// 根域名 (@) 不能使用 CNAME
//
// www.example.com.  300  IN  CNAME  example.com.
// 访问 www.example.com → DNS 解析到 example.com
// 再通过 A/AAAA 记录获取 IP
//
// 常见用途:
//   www → 根域名
//   blog → GH pages
//   cdn → CDN 提供商
```
## MX 与 TXT 记录
```
// ========== MX 记录 (Mail Exchange) ==========
// 指定邮件服务器
// 优先级: 数字越小优先级越高
//
// example.com.  3600  IN  MX  10  mail.example.com.
// example.com.  3600  IN  MX  20  backup-mail.example.com.
//                              └── 优先级 (10 > 20)
//
// 配置示例:
//   example.com  MX  10  mail.example.com
//   example.com  MX  20  mail2.example.com
//   mail         A       203.0.113.10

// ========== TXT 记录 (Text) ==========
// 存储任意文本信息
// 主要用于验证和反垃圾邮件
//
// 常见用途:
//
// SPF (发件人策略框架):
//   声明哪些服务器可以发此域名的邮件
//   example.com  TXT  "v=spf1 include:_spf.google.com ~all"
//
// DKIM (域名密钥识别邮件):
//   邮件签名公钥
//   google._domainkey  TXT  "v=DKIM1; k=rsa; p=MIGfMA0..."
//
// DMARC (基于域名的消息认证):
//   未通过 SPF/DKIM 的邮件处理策略
//   _dmarc  TXT  "v=DMARC1; p=quarantine; rua=mailto:dmarc@example.com"

// ========== 常见 TXT 验证 ==========
// 域名所有权验证 (Google/Cloudflare/AWS):
//   TXT  "google-site-verification=abc123"
//
// 邮件验证:
//   TXT  "v=spf1 include:spf.mail.com ~all"

// ========== NS 记录 (Name Server) ==========
// 指定域名的权威 DNS 服务器
// example.com.  86400  IN  NS  ns1.example.com.
// example.com.  86400  IN  NS  ns2.example.com.
```
## SOA 与其他记录
```
// ========== SOA 记录 (Start of Authority) ==========
// 区域的权威信息,每个域名必须有一个 SOA
//
// example.com.  3600  IN  SOA  ns1.example.com. admin.example.com. (
//                 2024010101  ; Serial (序号,递增用于同步)
//                 7200        ; Refresh (从服务器刷新间隔)
//                 3600        ; Retry (失败重试间隔)
//                 1209600     ; Expire (从服务器过期时间)
//                 3600        ; Negative TTL (NXDOMAIN 缓存时间)
//                 )

// ========== PTR 记录 (Pointer) ==========
// IP 地址 → 域名 (反向解析)
// 用于反向 DNS 查询
//
// 34.216.184.93.in-addr.arpa.  IN  PTR  example.com.
//
// 用途:
//   邮件服务器验证 (防止垃圾邮件)
//   日志反向查询
//   安全审计

// ========== SRV 记录 (Service) ==========
// 指定特定服务的位置 (主机+端口)
// 格式: _service._proto.name.  IN  SRV priority weight port target
//
// _sip._tcp.example.com.  IN  SRV  10  60  5060  sipserver.example.com.
//                                     │   │    │          │
//                                    优先级 权重  端口  目标服务器
// 用途: SIP, XMPP, LDAP 等服务发现

// ========== CAA 记录 (Certification Authority Authorization) ==========
// 指定哪些 CA 可以为此域名签发证书
// 防止证书被错误签发
//
// example.com.  IN  CAA  0  issue  "letsencrypt.org"
//                          └── 只允许 Let's Encrypt 签发
```
## 记录配置示例
```
// ========== 典型网站 DNS 配置 ==========
//
// 主域名 (example.com):
//   @         A       93.184.216.34       (根域名)
//   @         AAAA    2001:db8::1         (IPv6)
//   @         MX 10   mail.example.com    (邮件)
//   @         TXT     "v=spf1 include:_spf.google.com ~all"
//   @         NS      ns1.cloudflare.com  (DNS)
//   @         NS      ns2.cloudflare.com
//   @         SOA     ns1.cloudflare.com. dns.cloudflare.com. (...)
//
// 子域名:
//   www       CNAME   example.com         (www 别名)
//   api       A       93.184.216.35       (API 服务器)
//   blog      A       185.199.108.153     (GitHub Pages)
//   mail      A       203.0.113.10        (邮件服务器)
//   cdn       CNAME   example.cloudfront.net (CDN)
//   _dmarc    TXT     "v=DMARC1; p=quarantine"

// ========== 配置注意事项 ==========
// 1. CNAME 不能与其他记录共存
//    如 www.example.com 有 CNAME,就不能有 MX
//    但可以使用 ALIAS/ANAME 记录 (部分 DNS 提供商)
//
// 2. TTL 设置
//    稳定记录: 3600-86400
//    可能变更: 60-300
//    变更前降低 TTL
//
// 3. 检查配置
//    $ dig example.com ANY +short
//    $ dig example.com MX +short
//    $ dig _dmarc.example.com TXT +short
```
