# 云解析与CDN

## 一、概念说明

阿里云云解析DNS提供域名解析服务，CDN提供内容分发加速服务。两者配合实现全球加速访问。

| 解析类型 | 说明 | 用途 |
|----------|------|------|
| A记录 | 域名到IPv4 | 基本解析 |
| CNAME | 域名到域名 | CDN/负载均衡 |
| MX | 邮件服务器 | 邮箱服务 |
| TXT | 文本记录 | 验证/SPF |
| AAAA | 域名到IPv6 | IPv6支持 |

## 二、具体用法

### 云解析配置

```bash
# 添加域名
aliyun alidns AddDomain --DomainName example.com

# 添加A记录
aliyun alidns AddDomainRecord \
    --DomainName example.com \
    --RR www \
    --Type A \
    --Value 47.96.xxx.xxx \
    --TTL 600

# 添加CNAME记录
aliyun alidns AddDomainRecord \
    --DomainName example.com \
    --RR cdn \
    --Type CNAME \
    --Value example.com.w.cdngslb.com \
    --TTL 600

# 查询记录列表
aliyun alidns DescribeDomainRecords \
    --DomainName example.com \
    --PageSize 50
```

### CDN加速配置

```bash
# 添加CDN加速域名
aliyun cdn AddCdnDomain \
    --CdnType web \
    --DomainName static.example.com \
    --OriginType oss \
    --Sources '[{"content":"my-bucket.oss-cn-hangzhou.aliyuncs.com","type":"oss","priority":"20","port":80}]' \
    --Scope overseas

# CDN刷新缓存
aliyun cdn RefreshObjectCaches \
    --ObjectPath "https://static.example.com/css/*" \
    --ObjectType Directory

# 预热资源
aliyun cdn PushObjectCache \
    --ObjectPath "https://static.example.com/index.html"

# 查询CDN流量
aliyun cdn DescribeDomainTrafficData \
    --DomainName static.example.com \
    --StartTime "2024-01-15T00:00:00Z" \
    --EndTime "2024-01-16T00:00:00Z" \
    --Interval 3600
```

### HTTPS配置

```bash
# 上传SSL证书
aliyun cdn SetDomainServerCertificate \
    --DomainName static.example.com \
    --ServerCertificateStatus on \
    --CertName my-cert-2024 \
    --CertType upload \
    --ServerCertificate "-----BEGIN CERTIFICATE-----..." \
    --PrivateKey "-----BEGIN RSA PRIVATE KEY-----..."

# 强制HTTPS跳转
aliyun cdn SetHttpHeaderConfig \
    --DomainName static.example.com \
    --HeaderKey Strict-Transport-Security \
    --HeaderValue "max-age=31536000; includeSubDomains"
```

### 全站加速（DCDN）

```bash
# 添加全站加速域名
aliyun dcdn AddDcdnDomain \
    --DomainName www.example.com \
    --ServiceType whole \
    --Sources '[{"content":"1.2.3.4","type":"ipaddr","priority":"20","port":80}]'

# 开启WebSocket支持
aliyun dcdn SetDcdnDomainConfig \
    --DomainName www.example.com \
    --FunctionName websocket \
    --FunctionArgs '[{"argName":"enabled","argValue":"on"}]'
```

## 三、注意事项与常见陷阱

1. **DNS TTL**：变更频繁的记录设置较短TTL（60-300秒）
2. **CDN缓存**：合理设置缓存规则，静态资源长缓存
3. **HTTPS强制**：全站启用HTTPS并强制跳转
4. **回源带宽**：CDN未命中会增加源站带宽压力
5. **防盗链**：配置Referer防盗链防止资源被盗用
6. **缓存刷新**：更新内容后及时刷新CDN缓存
7. **全球加速**：海外用户使用全球加速（DCDN）提升体验
