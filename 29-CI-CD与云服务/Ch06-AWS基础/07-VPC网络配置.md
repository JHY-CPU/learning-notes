# VPC网络配置

## 一、概念说明

VPC（Virtual Private Cloud）是在AWS中创建的逻辑隔离网络空间。用户可完全控制网络环境，包括IP地址范围、子网、路由表和网关配置。

| 组件 | 说明 | 用途 |
|------|------|------|
| VPC | 虚拟网络容器 | 网络隔离边界 |
| Subnet | 子网 | 资源部署位置 |
| Route Table | 路由表 | 流量路由规则 |
| Internet Gateway | 互联网网关 | 公网访问 |
| NAT Gateway | NAT网关 | 私有子网出站 |
| Security Group | 安全组 | 实例级防火墙 |
| NACL | 网络ACL | 子网级防火墙 |

## 二、具体用法

### 创建VPC

```bash
# 创建VPC
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --tag-specifications \
    'ResourceType=vpc,Tags=[{Key=Name,Value=MyVPC}]'

# 创建公有子网
aws ec2 create-subnet \
    --vpc-id vpc-12345678 \
    --cidr-block 10.0.1.0/24 \
    --availability-zone us-east-1a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=PublicSubnet}]'

# 创建私有子网
aws ec2 create-subnet \
    --vpc-id vpc-12345678 \
    --cidr-block 10.0.2.0/24 \
    --availability-zone us-east-1a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=PrivateSubnet}]'
```

### 配置网关和路由

```bash
# 创建互联网网关
aws ec2 create-internet-gateway
aws ec2 attach-internet-gateway \
    --internet-gateway-id igw-12345678 \
    --vpc-id vpc-12345678

# 添加路由到公有子网
aws ec2 create-route \
    --route-table-id rtb-12345678 \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id igw-12345678

# 创建NAT网关（供私有子网出站）
aws ec2 allocate-address --domain vpc
aws ec2 create-nat-gateway \
    --subnet-id subnet-public \
    --allocation-id eipalloc-12345678
```

### 安全组配置

```bash
# 创建安全组
aws ec2 create-security-group \
    --group-name WebServerSG \
    --description "Web服务器安全组" \
    --vpc-id vpc-12345678

# 添加入站规则
aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp --port 80 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp --port 443 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp --port 22 --cidr 10.0.0.0/16
```

### VPC Peering

```bash
# 创建VPC对等连接
aws ec2 create-vpc-peering-connection \
    --vpc-id vpc-vpc1 \
    --peer-vpc-id vpc-vpc2

# 接受对等连接
aws ec2 accept-vpc-peering-connection \
    --vpc-peering-connection-id pcx-12345678

# 添加路由
aws ec2 create-route \
    --route-table-id rtb-vpc1 \
    --destination-cidr-block 10.1.0.0/16 \
    --vpc-peering-connection-id pcx-12345678
```

## 三、注意事项与常见陷阱

1. **CIDR规划**：合理规划IP地址范围，避免重叠
2. **子网划分**：公有子网放负载均衡器，私有子网放应用和数据库
3. **安全组vs NACL**：安全组有状态，NACL无状态，两者配合使用
4. **NAT网关费用**：NAT网关按小时和数据处理量计费
5. **可用区分布**：多可用区部署提高可用性
6. **VPC Peering限制**：对等连接不传递，不能重叠CIDR
7. **流日志**：启用VPC流日志用于网络排障和安全审计
