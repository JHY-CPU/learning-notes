# CloudFront CDN

## 一、概念说明

CloudFront是AWS的CDN（内容分发网络）服务，通过全球边缘节点缓存内容，降低延迟并提高传输速度。

| 概念 | 说明 |
|------|------|
| Distribution | CDN分发配置 |
| Origin | 源站（S3/EC2/自定义） |
| Edge Location | 边缘缓存节点 |
| Cache Policy | 缓存策略 |
| Behavior | 路径行为规则 |

## 二、具体用法

### 创建分发

```bash
# 创建CloudFront分发
aws cloudfront create-distribution --distribution-config '{
    "CallerReference": "my-distribution-001",
    "Origins": {
        "Quantity": 1,
        "Items": [{
            "Id": "myS3Origin",
            "DomainName": "my-bucket.s3.amazonaws.com",
            "S3OriginConfig": {
                "OriginAccessIdentity": ""
            }
        }]
    },
    "DefaultCacheBehavior": {
        "TargetOriginId": "myS3Origin",
        "ViewerProtocolPolicy": "redirect-to-https",
        "AllowedMethods": {
            "Quantity": 2,
            "Items": ["GET", "HEAD"],
            "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}
        },
        "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6"
    },
    "Enabled": true,
    "Comment": "My Distribution"
}'
```

### 与S3配合

```bash
# 创建OAC（Origin Access Control）
aws cloudfront create-origin-access-control \
    --origin-access-control-config '{
        "Name": "my-oac",
        "OriginAccessControlOriginType": "s3",
        "SigningBehavior": "always",
        "SigningProtocol": "sigv4"
    }'

# S3存储桶策略仅允许CloudFront访问
aws s3api put-bucket-policy --bucket my-bucket --policy '{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "cloudfront.amazonaws.com"},
        "Action": "s3:GetObject",
        "Resource": "arn:aws:s3:::my-bucket/*",
        "Condition": {
            "StringEquals": {
                "AWS:SourceArn": "arn:aws:cloudfront::123456789012:distribution/E12345678"
            }
        }
    }]
}'
```

### 缓存失效

```bash
# 创建缓存失效
aws cloudfront create-invalidation \
    --distribution-id E12345678 \
    --paths "/*"

# 特定路径失效
aws cloudfront create-invalidation \
    --distribution-id E12345678 \
    --paths "/images/*" "/css/styles.css"
```

### 自定义域名和HTTPS

```bash
# 请求SSL证书（ACM）
aws acm request-certificate \
    --domain-name example.com \
    --subject-alternative-names "*.example.com" \
    --validation-method DNS

# 更新分发使用自定义域名
aws cloudfront update-distribution \
    --id E12345678 \
    --if-match E1ABC123 \
    --distribution-config file://config.json
```

## 三、注意事项与常见陷阱

1. **缓存策略**：合理设置TTL，静态资源长TTL，动态内容短TTL
2. **价格分层**：不同区域价格不同，注意数据传输费用
3. **HTTPS配置**：必须使用ACM证书，且证书区域必须为us-east-1
4. **缓存失效延迟**：失效操作最多需要15分钟完成
5. **源站保护**：使用OAC防止绕过CDN直接访问S3
6. **压缩启用**：启用Gzip/Brotli压缩减少传输数据量
7. **实时日志**：配置实时日志分析用户访问模式
