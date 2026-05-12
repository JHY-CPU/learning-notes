# 阿里云与AWS核心服务 - 云服务与DevOps笔记


### 1.1 云服务器（ECS vs EC2）


云服务器是最核心的IaaS产品，提供弹性的虚拟机实例。


| 维度 | 阿里云 ECS | AWS EC2 |
| --- | --- | --- |
| 实例类型 | 通用型/计算型/内存型/GPU型 | t3/m5/c5/r5/p3/g4等系列 |
| 计费模式 | 按量/包年包月/竞价 | On-Demand/Reserved/Spot |
| 镜像 | 自定义镜像/公共镜像/市场镜像 | AMI（Amazon Machine Image） |
| 存储挂载 | 云盘（高效/SSD/ESSD） | EBS（gp2/gp3/io1/io2） |
| 自动伸缩 | ESS（弹性伸缩服务） | Auto Scaling Group (ASG) |
| 密钥管理 | 密钥对 | Key Pair (.pem) |


### 1.2 容器服务


| 类型 | 阿里云 | AWS |
| --- | --- | --- |
| 托管K8s | ACK（容器服务Kubernetes版） | EKS（Elastic Kubernetes Service） |
| Serverless容器 | ASK | Fargate |
| 容器镜像仓库 | ACR | ECR |
| 弹性容器实例 | ECI | Fargate |


### 1.3 函数计算


- **阿里云**
   ：函数计算（Function Compute, FC）— 支持事件驱动的Serverless计算
- **AWS**
   ：Lambda — Serverless函数计算的先驱和标杆
- **共同特点**
   ：按调用次数和执行时长计费、自动扩缩容、支持多种触发器


### 2.1 对象存储（OSS vs S3）


对象存储是云上最常用的存储服务，用于存放海量非结构化数据。


| 维度 | 阿里云 OSS | AWS S3 |
| --- | --- | --- |
| 存储类型 | 标准/低频/归档/深度归档 | Standard/IA/Glacier/Deep Archive |
| 访问方式 | RESTful API / SDK / 控制台 | RESTful API / SDK / 控制台 |
| 权限控制 | Bucket Policy / RAM Policy | Bucket Policy / IAM Policy |
| 生命周期 | 生命周期规则自动转存/删除 | Lifecycle Rules |
| 版本控制 | 支持 | 支持 |
| 静态网站 | 支持托管静态网站 | 支持托管静态网站 |
| 事件通知 | 支持触发函数计算/MNS | 支持触发Lambda/SQS/SNS |
| 数据一致性 | 强一致性 | 强一致性（2020年后） |


> **Example:** **阿里云OSS基本操作示例（CLI）：**
>
> ```
> # 创建Bucket
> ossutil mb oss://my-bucket-cn-hangzhou --region cn-hangzhou
>
> # 上传文件
> ossutil cp ./photo.jpg oss://my-bucket-cn-hangzhou/photos/
>
> # 列出文件
> ossutil ls oss://my-bucket-cn-hangzhou/ -s
>
> # 设置生命周期规则（30天后转低频，90天后删除）
> ossutil lifecycle set lifecycle.xml oss://my-bucket-cn-hangzhou
> ```


> **Example:** **AWS S3基本操作示例（CLI）：**
>
> ```
> # 创建Bucket
> aws s3 mb s3://my-bucket-us-east-1 --region us-east-1
>
> # 上传文件
> aws s3 cp ./photo.jpg s3://my-bucket-us-east-1/photos/
>
> # 列出文件
> aws s3 ls s3://my-bucket-us-east-1/ --recursive
>
> # 设置生命周期规则
> aws s3api put-bucket-lifecycle-configuration \
>   --bucket my-bucket-us-east-1 \
>   --lifecycle-configuration file://lifecycle.json
> ```


### 2.2 块存储与文件存储


| 类型 | 阿里云 | AWS | 特点 |
| --- | --- | --- | --- |
| 块存储 | 云盘（ESSD/SSD/高效） | EBS（gp3/io2/st1） | 挂载到ECS，类似硬盘 |
| 文件存储 | NAS（通用型/极速型） | EFS | NFS协议，多实例共享 |
| 共享块存储 | 共享盘 | Multi-Attach EBS | 多实例同时挂载 |


| 数据库类型 | 阿里云产品 | AWS产品 | 典型场景 |
| --- | --- | --- | --- |
| MySQL | RDS MySQL / PolarDB | RDS for MySQL / Aurora | 通用Web应用 |
| PostgreSQL | RDS PostgreSQL / PolarDB | RDS for PostgreSQL / Aurora | GIS/JSON/分析 |
| Redis | Tair（Redis增强版） | ElastiCache for Redis | 缓存/会话/排行榜 |
| MongoDB | 云MongoDB / Lindorm | DocumentDB | 文档存储/内容管理 |
| OLAP分析 | AnalyticDB / MaxCompute | Redshift / Athena | 数据仓库/BI |
| NoSQL宽表 | Tablestore / Lindorm | DynamoDB | 物联网/大数据 |
| 图数据库 | GDB | Neptune | 社交网络/知识图谱 |


### 3.1 关系型数据库对比细节


- **阿里云RDS**
   ：提供MySQL、PostgreSQL、SQL Server、MariaDB，支持高可用版（主备）、集群版
- **阿里云PolarDB**
   ：云原生数据库，计算存储分离，兼容MySQL/PostgreSQL/Oracle
- **AWS RDS**
   ：支持MySQL、PostgreSQL、MariaDB、Oracle、SQL Server
- **AWS Aurora**
   ：云原生数据库，兼容MySQL/PostgreSQL，性能是标准RDS的5倍


| 服务 | 阿里云 | AWS | 功能 |
| --- | --- | --- | --- |
| 虚拟私有云 | VPC | VPC | 隔离的虚拟网络 |
| 负载均衡 | SLB（ALB/NLB/CLB） | ELB（ALB/NLB/CLB） | 流量分发 |
| 弹性公网IP | EIP | Elastic IP | 固定公网IP |
| NAT网关 | NAT网关 | NAT Gateway | 私网访问外网 |
| VPN | VPN网关 | VPN Gateway | 安全加密连接 |
| 专线 | 高速通道/物理专线 | Direct Connect | 专用网络连接 |
| CDN | CDN / DCDN | CloudFront | 内容分发加速 |
| DNS | 云解析DNS | Route 53 | 域名解析/流量调度 |
| 安全组 | 安全组 | Security Group | 实例级防火墙 |
| 网络ACL | 网络ACL | NACL | 子网级防火墙 |


### 5.1 IAM对比


| 维度 | 阿里云 RAM | AWS IAM |
| --- | --- | --- |
| 用户管理 | RAM用户 | IAM User |
| 用户组 | 用户组 | IAM Group |
| 角色 | RAM角色 | IAM Role |
| 策略 | 自定义/系统策略（JSON） | Managed/Inline Policy（JSON） |
| SSO | RAM SSO / IDaaS | IAM Identity Center (SSO) |
| MFA | 虚拟MFA | Virtual MFA / 硬件MFA |
| 临时凭证 | STS临时凭证 | STS Temporary Credentials |


### 5.2 其他安全服务


| 安全领域 | 阿里云 | AWS |
| --- | --- | --- |
| Web应用防火墙 | WAF | AWS WAF |
| DDoS防护 | DDoS高防IP | AWS Shield |
| 密钥管理 | KMS | KMS |
| 证书管理 | SSL证书服务 | ACM (Certificate Manager) |
| 日志审计 | ActionTrail | CloudTrail |
| 配置审计 | 配置审计 | AWS Config |
| 漏洞扫描 | 安全中心 | Inspector |

**IAM安全最佳实践：**

- 遵循最小权限原则，不要使用根账号/主账号进行日常操作
- 为所有IAM用户启用MFA（多因素认证）
- 使用角色（Role）而非长期密钥进行服务间调用
- 定期轮转访问密钥（Access Key）
- 使用策略版本控制，方便回滚


| 功能 | 阿里云 | AWS |
| --- | --- | --- |
| 云监控 | CloudMonitor | CloudWatch |
| 日志服务 | SLS（日志服务） | CloudWatch Logs |
| 告警通知 | 云监控告警 + MNS | CloudWatch Alarms + SNS |
| 基础设施编排 | ROS（资源编排） | CloudFormation |
| 运维编排 | OOS | Systems Manager |
| 应用监控 | ARMS / Prometheus | X-Ray / CloudWatch Application Insights |
| 消息通知 | MNS / 云原生MNS | SNS / SQS |


### 6.1 全面对照速查表


> **Example:** **阿里云 vs AWS 速查：**
>
> - `ECS`
>    ↔
>    `EC2`
>    （云服务器）
> - `OSS`
>    ↔
>    `S3`
>    （对象存储）
> - `RDS`
>    ↔
>    `RDS`
>    （关系数据库）
> - `VPC`
>    ↔
>    `VPC`
>    （虚拟私有云）
> - `SLB`
>    ↔
>    `ELB`
>    （负载均衡）
> - `RAM`
>    ↔
>    `IAM`
>    （身份管理）
> - `CDN`
>    ↔
>    `CloudFront`
>    （内容分发）
> - `ROS`
>    ↔
>    `CloudFormation`
>    （资源编排）
> - `ACK`
>    ↔
>    `EKS`
>    （Kubernetes服务）
> - `FC`
>    ↔
>    `Lambda`
>    （函数计算）


<!-- Converted from: 02_阿里云AWS核心服务.html -->
