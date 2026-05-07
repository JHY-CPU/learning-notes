# SLB负载均衡

## 一、概念说明

SLB（Server Load Balancer）是阿里云的负载均衡服务，将流量分发到多台后端服务器，提高应用可用性和扩展性。

| 类型 | 协议 | 适用场景 |
|------|------|----------|
| CLB（传统型） | TCP/UDP/HTTP/HTTPS | 简单负载 |
| ALB（应用型） | HTTP/HTTPS | 七层负载 |
| NLB（网络型） | TCP/UDP/SSL | 四层高性能 |

## 二、具体用法

### 创建负载均衡

```bash
# 创建ALB实例
aliyun alb CreateLoadBalancer \
    --LoadBalancerName production-alb \
    --LoadBalancerCategory Standard \
    --VpcId vpc-bp1xxxxxxxx \
    --AddressType Internet \
    --ZoneMappings.1.VSwitchId vsw-bp1xxxxxxxx \
    --ZoneMappings.1.ZoneId cn-hangzhou-b \
    --ZoneMappings.2.VSwitchId vsw-bp1yyyyyyyy \
    --ZoneMappings.2.ZoneId cn-hangzhou-c
```

### 创建监听器和转发规则

```bash
# 创建HTTP监听器
aliyun alb CreateListener \
    --LoadBalancerId alb-bp1xxxxxxxx \
    --ListenerProtocol HTTP \
    --ListenerPort 80 \
    --DefaultActions.1.Type ForwardGroup \
    --DefaultActions.1.ForwardGroupConfig.ServerGroupTuples.1.ServerGroupId sg-bp1xxxxxxxx \
    --DefaultActions.1.ForwardGroupConfig.ServerGroupTuples.1.Weight 100

# 创建HTTPS监听器
aliyun alb CreateListener \
    --LoadBalancerId alb-bp1xxxxxxxx \
    --ListenerProtocol HTTPS \
    --ListenerPort 443 \
    --Certificates.1.CertificateId 12345678 \
    --DefaultActions.1.Type ForwardGroup \
    --DefaultActions.1.ForwardGroupConfig.ServerGroupTuples.1.ServerGroupId sg-bp1xxxxxxxx
```

### 后端服务器组

```bash
# 创建服务器组
aliyun alb CreateServerGroup \
    --ServerGroupName web-servers \
    --VpcId vpc-bp1xxxxxxxx \
    --Protocol HTTP \
    --HealthCheckConfig.HealthCheckEnabled true \
    --HealthCheckConfig.HealthCheckProtocol HTTP \
    --HealthCheckConfig.HealthCheckPath /health \
    --HealthCheckConfig.HealthCheckInterval 5 \
    --HealthCheckConfig.HealthyThreshold 3 \
    --HealthCheckConfig.UnhealthyThreshold 3

# 添加后端服务器
aliyun alb AddServersToServerGroup \
    --ServerGroupId sg-bp1xxxxxxxx \
    --Servers.1.ServerId i-bp1xxxxxxxx \
    --Servers.1.ServerIp 10.0.1.10 \
    --Servers.1.ServerType ECS \
    --Servers.1.Weight 100 \
    --Servers.1.Port 8080 \
    --Servers.2.ServerId i-bp1yyyyyyyy \
    --Servers.2.ServerIp 10.0.1.11 \
    --Servers.2.ServerType ECS \
    --Servers.2.Weight 100 \
    --Servers.2.Port 8080
```

### 转发规则

```bash
# 基于路径的转发
aliyun alb CreateRule \
    --ListenerId lsn-bp1xxxxxxxx \
    --Priority 1 \
    --RuleConditions.1.PathConfig.Values '["/api/*"]' \
    --RuleActions.1.Type ForwardGroup \
    --RuleActions.1.ForwardGroupConfig.ServerGroupTuples.1.ServerGroupId sg-api-server

# 基于域名的转发
aliyun alb CreateRule \
    --ListenerId lsn-bp1xxxxxxxx \
    --Priority 2 \
    --RuleConditions.1.HostConfig.Values '["api.example.com"]' \
    --RuleActions.1.Type ForwardGroup \
    --RuleActions.1.ForwardGroupConfig.ServerGroupTuples.1.ServerGroupId sg-api-server
```

### 查看状态和监控

```bash
# 查看负载均衡状态
aliyun alb DescribeLoadBalancers --LoadBalancerName production-alb

# 查看后端健康状态
aliyun alb ListServerGroupServers \
    --ServerGroupId sg-bp1xxxxxxxx
```

## 三、注意事项与常见陷阱

1. **类型选择**：HTTP/HTTPS应用选ALB，TCP/UDP高性能选NLB
2. **跨可用区**：部署到多个可用区保证高可用
3. **健康检查**：配置合理的健康检查参数
4. **会话保持**：需要状态保持的场景开启会话保持
5. **带宽限制**：注意SLB的带宽和连接数限制
6. **WAF集成**：ALB可集成Web应用防火墙
7. **访问日志**：开启访问日志便于排查问题
