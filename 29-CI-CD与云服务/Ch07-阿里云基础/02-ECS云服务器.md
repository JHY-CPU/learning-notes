# ECS云服务器

## 一、概念说明

ECS（Elastic Compute Service）是阿里云的云服务器服务，提供弹性可扩展的计算能力。支持多种实例规格族满足不同场景需求。

| 规格族 | 适用场景 | 特点 |
|--------|----------|------|
| 通用型g系列 | Web应用 | CPU内存均衡 |
| 计算型c系列 | 批处理 | 高CPU |
| 内存型r系列 | 数据库 | 高内存 |
| 突发型t系列 | 开发测试 | 突发性能 |
| GPU型 | AI/ML | GPU加速 |

## 二、具体用法

### 创建实例

```bash
# 创建ECS实例
aliyun ecs RunInstances \
    --RegionId cn-hangzhou \
    --ImageId aliyun_3_x64_20G_alibase_20231221.vhd \
    --InstanceType ecs.g7.xlarge \
    --SecurityGroupId sg-bp1xxxxxxxx \
    --VSwitchId vsw-bp1xxxxxxxx \
    --InstanceName production-web \
    --HostName web-server \
    --InternetMaxBandwidthOut 10 \
    --SystemDisk.Category cloud_essd \
    --SystemDisk.Size 40 \
    --KeyPairName my-keypair \
    --Tag.1.Key Environment \
    --Tag.1.Value production
```

### 实例操作

```bash
# 启动/停止/重启
aliyun ecs StartInstance --InstanceId i-bp1xxxxxxxx
aliyun ecs StopInstance --InstanceId i-bp1xxxxxxxx
aliyun ecs RebootInstance --InstanceId i-bp1xxxxxxxx

# 重置密码
aliyun ecs ModifyInstanceAttribute \
    --InstanceId i-bp1xxxxxxxx \
    --Password "NewPassword123!"

# 创建快照
aliyun ecs CreateSnapshot \
    --DiskId d-bp1xxxxxxxx \
    --SnapshotName "backup-20240115"

# 创建自定义镜像
aliyun ecs CreateImage \
    --InstanceId i-bp1xxxxxxxx \
    --ImageName "my-custom-image" \
    --Description "生产环境镜像v1.0"
```

### 用户数据脚本

```bash
#!/bin/bash
# 用户数据脚本 - 初始化服务器
yum update -y
yum install -y nginx
systemctl enable nginx
systemctl start nginx

# 安装监控agent
wget https://aliyun-client.oss-cn-hangzhou.aliyuncs.com/aliyun_assist_latest.rpm
rpm -ivh aliyun_assist_latest.rpm
```

### 弹性伸缩

```bash
# 创建伸缩组
aliyun ess CreateScalingGroup \
    --RegionId cn-hangzhou \
    --ScalingGroupName web-scaling-group \
    --MinSize 2 \
    --MaxSize 10 \
    --DefaultCooldown 300 \
    --VSwitchIds.1 vsw-bp1xxxxxxxx \
    --LoadBalancerIds.1 lb-bp1xxxxxxxx

# 创建伸缩规则
aliyun ess CreateScalingRule \
    --ScalingGroupId sg-bp1xxxxxxxx \
    --ScalingRuleName scale-out \
    --AdjustmentType QuantityChangeInCapacity \
    --AdjustmentValue 2

# 创建报警任务
aliyun ess CreateAlarm \
    --ScalingGroupId sg-bp1xxxxxxxx \
    --Name high-cpu-alarm \
    --MetricType system \
    --MetricName CpuUtilization \
    --ComparisonOperator GreaterThanThreshold \
    --Threshold 80 \
    --EvaluationCount 3
```

## 三、注意事项与常见陷阱

1. **实例规格选择**：根据工作负载选择合适规格，避免资源浪费
2. **系统盘配置**：ESSD性能最好，ESSD PL0性价比高
3. **快照策略**：定期自动快照，重要数据跨区域备份
4. **安全组限制**：最小开放原则，避免0.0.0.0/0开放端口
5. **带宽计费**：按固定带宽还是按流量计费要按需选择
6. **实例释放**：按时释放不用的实例，设置自动释放
7. **镜像管理**：定期清理不用的自定义镜像和快照节省费用
