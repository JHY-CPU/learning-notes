# HTTPS配置与实践

## 🔒 HTTPS 配置与实践

HTTPS 配置步骤、Nginx 配置、HSTS、证书自动续期、TLS 性能优化。

## HTTPS 配置步骤
```
// ========== 使用 Let's Encrypt + Certbot ==========
//
// 1. 安装 Certbot:
//   $ sudo apt install certbot python3-certbot-nginx
//
// 2. 获取证书:
//   $ sudo certbot --nginx -d example.com -d www.example.com
//
// 3. 自动续期 (Certbot 自动添加 systemd timer):
//   $ sudo certbot renew --dry-run  # 测试续期
//
// 4. 验证:
//   $ curl -I https://example.com
//   $ openssl s_client -connect example.com:443

// ========== 使用 acme.sh (更轻量) ==========
// 1. 安装:
//   $ curl https://get.acme.sh | sh
//
// 2. 获取证书 (DNS 验证,支持通配符):
//   $ acme.sh --issue --dns dns_cf \
//     -d example.com -d '*.example.com'
//
// 3. 安装证书:
//   $ acme.sh --install-cert -d example.com \
//     --key-file /etc/nginx/ssl/key.pem \
//     --fullchain-file /etc/nginx/ssl/cert.pem

// ========== 证书路径 ==========
// Let's Encrypt:
//   /etc/letsencrypt/live/example.com/fullchain.pem
//   /etc/letsencrypt/live/example.com/privkey.pem
//
// acme.sh:
//   ~/.acme.sh/example.com/fullchain.cer
//   ~/.acme.sh/example.com/example.com.key
```
## Nginx HTTPS 配置
```
// ========== 完整的 Nginx HTTPS 配置 ==========
//
// server {
//     listen 443 ssl http2;
//     server_name example.com;
//
//     # 证书路径
//     ssl_certificate     /etc/letsencrypt/live/example.com/fullchain.pem;
//     ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;
//
//     # 现代 TLS 协议
//     ssl_protocols TLSv1.2 TLSv1.3;
//     ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
//     ssl_prefer_server_ciphers on;
//
//     # 会话缓存
//     ssl_session_cache shared:SSL:10m;
//     ssl_session_timeout 1d;
//     ssl_session_tickets off;
//
//     # OCSP Stapling
//     ssl_stapling on;
//     ssl_stapling_verify on;
//     resolver 8.8.8.8 1.1.1.1 valid=300s;
//
//     # HSTS (强制 HTTPS)
//     add_header Strict-Transport-Security
//       "max-age=63072000; includeSubDomains; preload" always;
//
//     # 其他安全头
//     add_header X-Frame-Options DENY;
//     add_header X-Content-Type-Options nosniff;
//
//     location / {
//         proxy_pass http://localhost:3000;
//     }
// }
//
// # HTTP → HTTPS 重定向
// server {
//     listen 80;
//     server_name example.com;
//     return 301 https://$host$request_uri;
// }
```
## HSTS (HTTP Strict Transport Security)
```
// ========== HSTS 原理 ==========
// 告诉浏览器: 以后永远用 HTTPS 访问本网站
// 即使用户输入 HTTP,浏览器也会自动升级
//
// 响应头:
//   Strict-Transport-Security: max-age=63072000; includeSubDomains; preload
//
// 参数:
//   max-age:        缓存时间 (秒)
//   includeSubDomains: 所有子域名也适用
//   preload:        提交到浏览器预加载列表
//
// ========== HSTS Preload ==========
// 提交到 https://hstspreload.org
// 浏览器内置列表,完全禁止 HTTP 连接
// 一旦加入,很难移除!
// 确保所有子域名都支持 HTTPS 再加入

// ========== TLS 性能优化 ==========
// 1. 会话复用 (Session Resumption):
//    缓存 TLS 会话参数,减少握手
//    配置: ssl_session_cache shared:SSL:10m
//    10MB ≈ 40000 个会话
//
// 2. OCSP Stapling:
//    服务器定期获取 OCSP 响应
//    握手中附带,客户端无需单独查询
//    减少一次 HTTP 请求
//
// 3. 证书优化:
//    ECDSA 证书比 RSA 更高效
//    相同安全级别, ECDSA P-256 ≈ RSA 3072
//    但 ECDSA 密钥生成和签名快得多
//
// 4. TLS 1.3:
//    1-RTT 握手比 1.2 的 2-RTT 更快
//    建议仅启用 TLS 1.2 + 1.3

// ========== SSL/TLS 测试 ==========
// $ curl https://example.com -vI
// $ openssl s_client -connect example.com:443
//   - 查看证书链, 密码套件, TLS 版本
//
// 在线工具: https://www.ssllabs.com/ssltest/
//   全面分析 HTTPS 配置安全等级 (A+ 为目标)
```
> **Note**: 🏆 使用 SSL Labs 测试你的 HTTPS 配置,目标分数 A+。关键点: TLS 1.2+、前向安全的密码套件、HSTS preload、OCSP Stapling、安全头配置完整。

## 常见问题排查
```
// ========== 常见问题 ==========
//
// 证书链不完整:
//   症状: 移动端访问显示"证书无效"
//   检查: 配置了 fullchain.pem 还是只有 cert.pem?
//   解决: 使用包含完整链的证书文件
//
// 混合内容 (Mixed Content):
//   症状: 页面锁图标变为"不安全"
//   原因: HTTPS 页面加载了 HTTP 资源
//   解决: 所有资源都使用 HTTPS
//
// 证书过期:
//   症状: 浏览器显示 SEC_ERROR_EXPIRED_CERTIFICATE
//   解决: 检查 certbot renew 定时任务是否正常运行
//   $ sudo certbot renew --dry-run
//
// 协议不匹配:
//   症状: 老设备/浏览器无法访问
//   解决: 确认是否禁用了 TLS 1.0/1.1 (有些老设备需要)

// ========== 验证 HTTPS 配置 ==========
// 检查证书信息:
//   $ echo | openssl s_client -connect example.com:443 \
//     2>/dev/null | openssl x509 -text | grep -E "Subject:|Not Before|Not After"
//
// 检查支持的 TLS 版本:
//   $ nmap --script ssl-enum-ciphers -p 443 example.com
//
// 检查 HSTS:
//   $ curl -sI https://example.com | grep -i strict

// ========== HTTPS 性能开销 ==========
// TLS 握手: 1-3 RTT 额外延迟
// CPU 开销: 现代硬件上 < 1% CPU
// 数据加密: AES-GCM 硬件加速,几乎无开销
//
// 结论: HTTPS 的性能开销可以忽略不计
// 没有理由不使用 HTTPS
```
