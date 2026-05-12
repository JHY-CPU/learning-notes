# 对象存储与Ceph - 存储系统


## 对象存储与 Ceph

第一部分：对象存储概念

## 一、对象存储基础概念


对象存储（Object Storage）是一种以"对象"为基本单位管理数据的存储架构，每个对象包含数据、元数据和唯一标识符。


### 1.1 存储类型对比


#### 块存储（Block）


- 数据以固定大小的块管理
- 无文件系统概念
- 直接操作磁盘扇区
- 高性能、低延迟
- 适合数据库、虚拟机磁盘


#### 文件存储（File）


- 数据以文件和目录组织
- 有文件系统和路径
- NFS/CIFS 协议
- 适合共享文件系统
- 元数据开销大


#### 对象存储（Object）


- 数据以对象（数据+元数据+ID）组织
- 扁平命名空间（无层级目录）
- HTTP/REST API 访问
- 无限扩展、高可靠
- 适合非结构化数据、备份归档


#### 对比总结


- 块：最快，无元数据
- 文件：最通用，有层次结构
- 对象：最易扩展，丰富的元数据
- 选择取决于应用需求
- 现代系统常组合使用


### 1.2 对象结构


```
对象 = 数据 + 元数据 + 对象 ID

┌──────────────────────────────────────────┐
│               对象（Object）              │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │ 对象 ID (Object ID)              │   │
│  │ 全局唯一标识符，如 UUID           │   │
│  └──────────────────────────────────┘   │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │ 数据 (Data)                      │   │
│  │ 实际存储的文件内容（任意大小）     │   │
│  └──────────────────────────────────┘   │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │ 元数据 (Metadata)                │   │
│  │ 系统元数据：大小、创建时间、校验和 │   │
│  │ 用户元数据：自定义键值对           │   │
│  └──────────────────────────────────┘   │
└──────────────────────────────────────────┘
```

第二部分：S3 API

## 二、Amazon S3 API


S3（Simple Storage Service）是 Amazon 于 2006 年推出的对象存储服务，其 API 已成为对象存储的事实标准。


### 2.1 核心概念


- **Bucket（存储桶）**
   ：对象的容器，全局唯一名称。可设置区域、权限、版本控制。
- **Object（对象）**
   ：存储的基本单元，由 Key（键）、Value（数据）、Metadata 组成。
- **Key（键）**
   ：对象在 Bucket 中的唯一标识，可包含 "/" 模拟目录结构（但实际是扁平的）。
- **Region（区域）**
   ：数据存储的地理区域，影响延迟和合规性。


### 2.2 S3 API 操作


| HTTP 方法 | API 操作 | 功能 |
| --- | --- | --- |
| PUT | PutObject | 上传对象（单次最大 5GB） |
| GET | GetObject | 下载对象 |
| DELETE | DeleteObject | 删除对象 |
| GET | ListObjects | 列出 Bucket 中的对象 |
| POST | InitiateMultipartUpload | 发起分段上传（大文件） |
| PUT | UploadPart | 上传分段 |
| POST | CompleteMultipartUpload | 完成分段上传 |
| HEAD | HeadObject | 获取对象元数据 |
| GET | GetBucketLocation | 获取 Bucket 区域 |


### 2.3 S3 兼容存储


- **MinIO**
   ：高性能 S3 兼容对象存储（Go 语言）
- **Ceph RADOS Gateway**
   ：Ceph 的 S3 兼容网关
- **Swift（OpenStack）**
   ：OpenStack 对象存储
- **阿里云 OSS**
   ：阿里云对象存储
- **腾讯云 COS**
   ：腾讯云对象存储

第三部分：Ceph 架构

## 三、Ceph 架构


Ceph 是一个统一的分布式存储系统，由 Sage Weil 于 2004 年在 UCSC 的博士论文中提出，同时支持对象存储、块存储和文件系统。


### 3.1 Ceph 统一存储


```
Ceph 的三种存储接口：

                    ┌────────────────────────┐
                    │     Ceph 集群          │
                    │                        │
  ┌─────────┐      │  ┌──────────────────┐  │
  │ RADOS   │──────┼─▶│                  │  │
  │ Gateway │      │  │     RADOS        │  │
  │(S3/Swift│      │  │  (Reliable        │  │
  │  API)   │      │  │   Autonomic       │  │
  └─────────┘      │  │   Distributed     │  │
                    │  │   Object Store)   │  │
  ┌─────────┐      │  │                  │  │
  │ RBD     │──────┼─▶│  核心对象存储层   │  │
  │(块存储) │      │  │                  │  │
  └─────────┘      │  └──────────────────┘  │
                    │                        │
  ┌─────────┐      │                        │
  │ CephFS  │──────┼─▶                      │
  │(文件系统│      │                        │
  └─────────┘      └────────────────────────┘
```


### 3.2 核心组件


| 组件 | 全称 | 功能 | 部署建议 |
| --- | --- | --- | --- |
| **OSD** | Object Storage Daemon | 管理单个磁盘，存储实际数据对象 | 每块磁盘一个 OSD |
| **MON** | Monitor | 维护集群映射（CRUSH Map、OSD Map 等） | 3 或 5 个（奇数） |
| **MDS** | Metadata Server | 管理 CephFS 的元数据（文件/目录信息） | CephFS 需要，至少 2 个 |
| **MGR** | Manager | 集群状态监控、性能指标、REST API | 至少 2 个（Active/Standby） |
| **RADOS GW** | RADOS Gateway | S3/Swift 兼容的对象存储网关 | 按需部署 |


### 3.3 RADOS 对象存储层


```
RADOS 数据写入流程：

Client                      Monitor          OSD
  │                            │              │
  │  1. 获取 CRUSH Map         │              │
  │ ─────────────────────────▶ │              │
  │  2. 返回 Map               │              │
  │ ◀───────────────────────── │              │
  │                            │              │
  │  3. 计算 PG（Placement Group）             │
  │     object_name ──hash──▶ PG ID            │
  │     PG ID ──CRUSH──▶ OSD 列表              │
  │                            │              │
  │  4. 写入主 OSD              │              │
  │ ────────────────────────────────────────▶ │
  │                            │              │
  │                            │  5. 复制到副本 OSD
  │                            │  ───────────▶│
  │                            │              │
  │  6. 确认写入完成            │              │
  │ ◀──────────────────────────────────────── │
```

第四部分：CRUSH 算法

## 四、CRUSH 算法


CRUSH（Controlled Replication Under Scalable Hashing）是 Ceph 的核心数据分布算法，一种伪随机数据分布算法，根据权重和故障域将对象映射到 OSD。


### 4.1 CRUSH 的优势


- **无中心节点**
   ：客户端直接计算数据位置，无需查询元数据服务器
- **确定性**
   ：给定相同的输入（对象 ID + Map），总是得到相同的结果
- **故障域感知**
   ：确保副本分布在不同的机架/主机上
- **最小化数据迁移**
   ：OSD 增减时，只迁移受影响的数据
- **权重控制**
   ：按 OSD 容量分配数据


### 4.2 CRUSH 层级结构


```
CRUSH Map 层级：

         ┌─────────┐
         │  Root   │  ← 根节点
         │ (root)  │
         └────┬────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌──────┐ ┌──────┐ ┌──────┐
│Row 0 │ │Row 1 │ │Row 2 │  ← 行级
└──┬───┘ └──┬───┘ └──┬───┘
   │        │        │
┌──┴──┐  ┌──┴──┐  ┌──┴──┐
│Rack │  │Rack │  │Rack │  ← 机架级
│  0  │  │  1  │  │  2  │
└──┬──┘  └──┬──┘  └──┬──┘
   │        │        │
┌──┴──┐  ┌──┴──┐  ┌──┴──┐
│Host │  │Host │  │Host │  ← 主机级
│  0  │  │  1  │  │  2  │
└──┬──┘  └──┬──┘  └──┬──┘
   │        │        │
┌──┴──┐  ┌──┴──┐  ┌──┴──┐
│OSD  │  │OSD  │  │OSD  │  ← OSD 级（叶子节点）
│  0  │  │  1  │  │  2  │
└─────┘  └─────┘  └─────┘

CRUSH 规则确保：3 副本分布在不同的机架！
```


### 4.3 CRUSH 规则（Rule）


```
# CRUSH 规则示例（伪代码风格）

rule replicated_rule {
    id 0
    type replicated
    step take root              # 从根节点开始
    step chooseleaf firstn 0 type rack  # 选择不同机架的叶子节点
    step emit                   # 输出选择的 OSD 列表
}

# chooseleaf 的工作方式：
# 1. take(root) → 开始遍历
# 2. choose firstn 3 type rack → 在 root 下选择 3 个不同的 rack
# 3. 在每个 rack 内 chooseleaf → 选择一个 OSD（叶子节点）
# 4. emit → 输出结果

# 结果：3 个 OSD 分布在 3 个不同的机架
# 即使一个机架整体故障，数据仍然可用
```

第五部分：PG 与数据分布

## 五、Placement Group（PG）与数据分布


### 5.1 PG 概念


PG（Placement Group）是 Ceph 中数据管理的逻辑单元。所有对象被哈希映射到某个 PG，PG 再通过 CRUSH 算法映射到一组 OSD。


```
数据映射过程：

  Object ──hash──▶ Pool/PG ──CRUSH──▶ OSD 列表

  例：
  object_name = "photo.jpg"
  pool_id = 1

  1. hash("photo.jpg") % num_pgs = PG 3.5a
  2. PG 3.5a 的 Acting Set = [OSD.12, OSD.25, OSD.38]
     - OSD.12 是主 OSD（Primary）
     - OSD.25, OSD.38 是副本 OSD

  一个 PG 包含多个对象（通常数千个）
  一个 OSD 包含多个 PG（通常数百个）
```


### 5.2 PG 状态


| 状态 | 含义 | 是否正常 |
| --- | --- | --- |
| **active+clean** | 所有副本完整且可用 | 正常 |
| **active+degraded** | 有副本丢失，数据仍在恢复中 | 可读写（降级） |
| **active+recovering** | 正在恢复丢失的副本 | 可读写 |
| **active+backfilling** | 正在后台回填数据 | 可读写 |
| **peering** | 正在同步 PG 状态 | 临时状态 |
| **stale** | PG 状态过期 | 异常 |
| **incomplete** | 没有足够的副本 | 异常（不可写） |


### 5.3 PG 数量计算


```
# PG 数量估算公式
#
#   Total PGs = (OSD_num × 100) / replica_count
#
# 建议每个 OSD 分配 50-100 个 PG
# PG 数量必须是 2 的幂次（如 256, 512, 1024...）

# 示例：100 个 OSD，3 副本
#   Total PGs = (100 × 100) / 3 ≈ 3333
#   最接近的 2 的幂 = 4096

# Ceph 自动计算（ceph osd pool create）：
#   ceph osd pool create mypool 4096 4096
#   其中 4096 是 PG 数量
```

第六部分：数据再平衡

## 六、数据再平衡（Rebalancing）


### 6.1 触发再平衡的场景


- **OSD 增加**
   ：新节点加入集群
- **OSD 移除**
   ：节点故障或维护
- **权重调整**
   ：调整 OSD 的 CRUSH 权重
- **故障域变化**
   ：添加/移除机架


### 6.2 再平衡过程


```
OSD 故障后的数据恢复：

正常状态：
  PG 3.5a → [OSD.12, OSD.25, OSD.38]  3 副本完整

OSD.25 故障：
  1. Monitor 检测到 OSD.25 超时（默认 5 分钟）
  2. 更新 OSD Map，标记 OSD.25 为 down/out
  3. CRUSH 重新计算：
     PG 3.5a → [OSD.12, OSD.38, OSD.47]  ← 新增 OSD.47
  4. OSD.12（Primary）通知 OSD.47 开始恢复
  5. OSD.47 从 OSD.12/38 复制缺失的对象
  6. 恢复完成后 PG 状态变为 active+clean

关键点：
  - 恢复是渐进式的（逐 PG 进行）
  - 可以限制恢复带宽（避免影响正常 I/O）
  - 支持优先级（紧急恢复先进行）
```


### 6.3 再平衡优化


```
# Ceph 再平衡相关配置

# 限制恢复带宽（避免影响业务 I/O）
ceph config set osd osd_recovery_max_active 3      # 最大活跃恢复数
ceph config set osd osd_recovery_sleep 0.1          # 恢复间隔
ceph config set osd osd_max_backfills 1             # 最大并行回填数
ceph config set osd osd_recovery_max_chunk 8388608  # 恢复块大小（8MB）

# 渐进式权重调整（安全迁移）
ceph osd crush reweight osd.25 0.5   # 逐步降低权重
ceph osd crush reweight osd.25 0.0   # 最终移除

# 暂停恢复（维护模式）
ceph osd set nobackfill
ceph osd set norecover
# 维护完成后
ceph osd unset nobackfill
ceph osd unset norecover
```

总结

## 总结


对象存储以对象为单位管理数据，通过 S3 API 等标准接口提供无限扩展的存储能力。Ceph 是一个统一的分布式存储系统，其核心 RADOS 层通过 CRUSH 算法实现无中心的数据分布，支持故障域感知和自动再平衡。理解 CRUSH 的层级结构、PG 的作用和数据恢复机制，是构建和运维 Ceph 集群的关键。

Object Storage
S3 API
Ceph
RADOS
CRUSH
Placement Group
Rebalancing


<!-- Converted from: 02_对象存储与Ceph.html -->
