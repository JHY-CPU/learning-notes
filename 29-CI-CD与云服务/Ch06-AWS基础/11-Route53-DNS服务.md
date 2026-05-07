# Route 53 DNS服务

## 一、概念说明

Route 53是AWS的DNS（域名系统）Web服务，提供域名注册、DNS路由和健康检查功能。支持多种路由策略。

| 路由策略 | 说明 | 适用场景 |
|----------|------|----------|
| Simple | 单一记录 | 基本DNS解析 |
| Weighted | 权重分配 | 流量分流测试 |
| Latency | 延迟最低 | 全球多区域部署 |
| Failover | 故障切换 | 主备架构 |
| Geolocation | 地理位置 | 按区域路由 |
| Multi-Value | 多值应答 | 负载均衡 |

## 二、具体用法

### 创建托管区域

```bash
# 创建托管区域
aws route53 create-hosted-zone \
    --name example.com \
    --caller-reference $(date +%s)
```

### 添加DNS记录

```bash
# A记录
aws route53 change-resource-record-sets \
    --hosted-zone-id Z1234567890 \
    --change-batch '{
        "Changes": [{
            "Action": "CREATE",
            "ResourceRecordSet": {
                "Name": "www.example.com",
                "Type": "A",
                "TTL": 300,
                "ResourceRecords": [{"Value": "192.0.2.1"}]
            }
        }]
    }'

# CNAME记录
aws route53 change-resource-record-sets \
    --hosted-zone-id Z1234567890 \
    --change-batch '{
        "Changes": [{
            "Action": "CREATE",
            "ResourceRecordSet": {
                "Name": "api.example.com",
                "Type": "CNAME",
                "TTL": 300,
                "ResourceRecords": [{"Value": "elb.amazonaws.com"}]
            }
        }]
    }'

# Alias记录（指向AWS资源）
aws route53 change-resource-record-sets \
    --hosted-zone-id Z1234567890 \
    --change-batch '{
        "Changes": [{
            "Action": "CREATE",
            "ResourceRecordSet": {
                "Name": "example.com",
                "Type": "A",
                "AliasTarget": {
                    "HostedZoneId": "Z35SXDOTRQ7X7K",
                    "DNSName": "my-cloudfront.cloudfront.net",
                    "EvaluateTargetHealth": true
                }
            }
        }]
    }'
```

### 权重路由

```bash
# 70%流量到v1，30%到v2
aws route53 change-resource-record-sets \
    --hosted-zone-id Z1234567890 \
    --change-batch '{
        "Changes": [
            {
                "Action": "CREATE",
                "ResourceRecordSet": {
                    "Name": "api.example.com",
                    "Type": "A",
                    "SetIdentifier": "v1",
                    "Weight": 70,
                    "TTL": 60,
                    "ResourceRecords": [{"Value": "1.1.1.1"}]
                }
            },
            {
                "Action": "CREATE",
                "ResourceRecordSet": {
                    "Name": "api.example.com",
                    "Type": "A",
                    "SetIdentifier": "v2",
                    "Weight": 30,
                    "TTL": 60,
                    "ResourceRecords": [{"Value": "2.2.2.2"}]
                }
            }
        ]
    }'
```

### 健康检查

```bash
# 创建HTTP健康检查
aws route53 create-health-check \
    --caller-reference $(date +%s) \
    --health-check-config '{
        "Type": "HTTP",
        "ResourcePath": "/health",
        "FullyQualifiedDomainName": "www.example.com",
        "Port": 80,
        "RequestInterval": 30,
        "FailureThreshold": 3
    }'
```

## 三、注意事项与常见陷阱

1. **TTL设置**：变更频繁的记录设置较短TTL
2. **Alias vs CNAME**：Alias免费且支持根域名，推荐用于AWS资源
3. **传播延迟**：DNS变更需要时间传播，可能需要48小时
4. **健康检查成本**：每个健康检查每月收费
5. **域名注册**：Route 53域名注册价格可能不是最低
6. **私有托管区域**：VPC内部DNS使用私有托管区域
7. **DNSSEC**：启用DNSSEC增强DNS安全性
