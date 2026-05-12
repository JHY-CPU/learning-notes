# Terraform入门 - 云服务与DevOps笔记


Terraform是HashiCorp开发的开源基础设施即代码（IaC）工具，使用声明式配置文件定义和管理云基础设施。它支持多云平台，通过Provider插件与AWS、阿里云、Azure、GCP等对接。


### 1.1 核心特性


- **声明式**
   ：描述期望的最终状态，Terraform自动计算如何达到该状态
- **多云支持**
   ：通过Provider支持几乎所有主流云平台和SaaS服务
- **执行计划**
   ：在应用变更前生成执行计划（plan），预览所有变更
- **资源图**
   ：构建资源依赖图，并行创建无依赖关系的资源
- **状态管理**
   ：跟踪管理的资源状态，支持增量变更
- **可扩展**
   ：通过模块实现配置复用和版本管理


### 1.2 Terraform工作流程


```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  Write   │───→│  Init    │───→│   Plan   │───→│  Apply   │
  │ (编写HCL) │    │(初始化Provider)│  │(预览变更) │    │(执行变更) │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                        │
                                                   ┌────┴────┐
                                                   │ Destroy │
                                                   │(销毁资源)│
                                                   └─────────┘
```


### 1.3 安装与基本命令


```
# 安装 Terraform (macOS)
brew install terraform

# 安装 Terraform (Windows)
choco install terraform

# 验证安装
terraform version

# 核心命令
terraform init      # 初始化工作目录，下载Provider
terraform plan      # 生成并显示执行计划
terraform apply     # 执行变更，创建/更新/删除资源
terraform destroy   # 销毁所有管理的资源
terraform state     # 管理状态文件
terraform fmt       # 格式化配置文件
terraform validate  # 验证配置语法
terraform output    # 显示输出值
```


HCL（HashiCorp Configuration Language）是Terraform使用的配置语言，兼具人类可读性和机器可解析性。


### 2.1 基本结构


```
# 块（Block）的基本结构
block_type "label1" "label2" {
  # 块内的参数
  key = value
  # 嵌套块
  nested_block {
    inner_key = inner_value
  }
}

# 示例：定义一个AWS EC2实例
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.micro"

  tags = {
    Name = "HelloWorld"
    Env  = "dev"
  }
}
```


### 2.2 数据类型


| 类型 | 示例 | 说明 |
| --- | --- | --- |
| string | `"hello"` | 字符串 |
| number | `42` / `3.14` | 数字 |
| bool | `true` / `false` | 布尔值 |
| list | `["a", "b", "c"]` | 有序列表 |
| map | `{key = "value"}` | 键值对集合 |
| object | `{name = "x", age = 1}` | 结构化对象 |
| tuple | `["a", 1, true]` | 混合类型列表 |


### 2.3 变量与输出


```
# 定义变量
variable "instance_type" {
  description = "EC2实例类型"
  type        = string
  default     = "t3.micro"
}

variable "availability_zones" {
  description = "可用区列表"
  type        = list(string)
  default     = ["cn-hangzhou-a", "cn-hangzhou-b"]
}

# 使用变量
resource "aws_instance" "web" {
  instance_type = var.instance_type  # 引用变量
}

# 定义输出
output "instance_ip" {
  description = "EC2实例的公网IP"
  value       = aws_instance.web.public_ip
}

# 条件表达式
resource "aws_eip" "ip" {
  count    = var.create_eip ? 1 : 0
  instance = aws_instance.web.id
}

# for表达式
output "instance_ids" {
  value = [for inst in aws_instance.web : inst.id]
}

# locals 局部值
locals {
  common_tags = {
    Project   = "my-project"
    ManagedBy = "terraform"
  }
}
```


### 3.1 Provider配置


```
# 配置AWS Provider
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.0"
}

provider "aws" {
  region = "cn-hangzhou"
# 可选：指定profile或使用环境变量
  # profile = "default"
}

# 配置阿里云Provider
terraform {
  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = "~> 1.200"
    }
  }
}

provider "alicloud" {
  region = "cn-hangzhou"
# 通过环境变量 ALICLOUD_ACCESS_KEY / ALICLOUD_SECRET_KEY 认证
}
```


### 3.2 Resource资源


Resource是Terraform的核心概念，表示一个基础设施对象。


```
# 语法：resource "资源类型" "本地名称" { 配置 }
# 创建阿里云VPC
resource "alicloud_vpc" "main" {
  vpc_name   = "my-vpc"
  cidr_block = "10.0.0.0/16"
}

# 创建子网
resource "alicloud_vswitch" "main" {
  vpc_id       = alicloud_vpc.main.id
  cidr_block   = "10.0.1.0/24"
  zone_id      = "cn-hangzhou-h"
}

# 创建ECS实例
resource "alicloud_instance" "web" {
  instance_name        = "web-server"
  instance_type        = "ecs.t6-c1m1.large"
  image_id             = "aliyun_3_x64_20G_alibase_2023xxxx.vhd"
  vswitch_id           = alicloud_vswitch.main.id
  security_groups      = [alicloud_security_group.main.id]
  system_disk_category = "cloud_essd"

  internet_max_bandwidth_out = 5
  internet_charge_type       = "PayByTraffic"

  tags = merge(local.common_tags, {
    Name = "web-server"
  })
}

# 引用其他资源的属性
output "vpc_id" {
  value = alicloud_vpc.main.id
}

output "instance_private_ip" {
  value = alicloud_instance.web.private_ip
}
```


### 3.3 Data Source（数据源）


Data Source用于查询已有资源的信息，不创建新资源。


```
# 查询最新的Ubuntu镜像
data "alicloud_images" "ubuntu" {
  name_regex = "^ubuntu_20.*64"
  owners     = "system"
  most_recent = true
}

# 在资源中使用数据源
resource "alicloud_instance" "web" {
  image_id = data.alicloud_images.ubuntu.images[0].id
  # ...
}
```


### 4.1 State的作用


Terraform使用State文件（`terraform.tfstate`）跟踪管理的资源状态。它是Terraform知道"当前基础设施长什么样"的唯一来源。


- **映射配置到真实资源**
   ：将HCL中的资源定义映射到云端真实资源
- **元数据存储**
   ：存储资源依赖关系、资源属性等
- **性能优化**
   ：通过缓存状态减少API调用
- **变更检测**
   ：对比期望状态和当前状态，计算需要的变更


### 4.2 远程State存储（团队协作必备）


```
# 使用阿里云OSS存储State（推荐）
terraform {
  backend "oss" {
    bucket = "my-terraform-state"
    prefix = "prod/network"
    region = "cn-hangzhou"
  }
}

# 使用AWS S3存储State
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/network/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks" # 用于状态锁定
    encrypt        = true
  }
}
```


### 4.3 State管理最佳实践

**重要规则：**

- **永远不要**
   将State文件提交到Git（包含敏感信息）
- **务必**
   使用远程后端存储State（OSS/S3/GCS）
- **务必**
   启用状态锁定，防止多人同时操作导致状态损坏
- 定期备份State文件
- 使用
   `terraform state`
   命令管理状态（mv/rm/import/show）


### 5.1 模块的概念


模块是Terraform中组织和复用配置的基本单元。每个包含`.tf`文件的目录都是一个模块（根模块）。通过模块可以将基础设施拆分为可复用的组件。


### 5.2 模块结构


```
# 目录结构
project/
├── main.tf           # 根模块入口
├── variables.tf      # 输入变量
├── outputs.tf        # 输出值
├── terraform.tfvars  # 变量值文件
└── modules/
    └── vpc/           # 子模块
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```


### 5.3 调用模块


```
# 调用本地模块
module "vpc" {
  source = "./modules/vpc"

  vpc_cidr = "10.0.0.0/16"
  azs      = ["cn-hangzhou-a", "cn-hangzhou-b"]
}

# 调用远程模块（Terraform Registry）
module "vpc" {
  source  = "alibaba/vpc/alicloud"
  version = "~> 1.0"

  vpc_name = "my-vpc"
  vpc_cidr = "10.0.0.0/16"
}

# 使用模块的输出
output "vpc_id" {
  value = module.vpc.vpc_id
}

# 子模块定义
# modules/vpc/variables.tf
variable "vpc_cidr" {
  type = string
}

# modules/vpc/main.tf
resource "alicloud_vpc" "this" {
  cidr_block = var.vpc_cidr
}

# modules/vpc/outputs.tf
output "vpc_id" {
  value = alicloud_vpc.this.id
}
```


### 6.1 Workspace（工作空间）


Workspace用于管理同一套配置的不同环境（dev/staging/prod），每个workspace拥有独立的State文件。


```
# 创建和切换workspace
terraform workspace new dev
terraform workspace new staging
terraform workspace new prod

# 列出所有workspace
terraform workspace list

# 查看当前workspace
terraform workspace show

# 在配置中使用workspace
locals {
  env = terraform.workspace
  instance_type = terraform.workspace == "prod" ? "ecs.c6.xlarge" : "ecs.t6-c1m1.large"
}
```


### 6.2 实战：完整项目结构


```
# terraform.tfvars - 变量值定义
region          = "cn-hangzhou"
project_name    = "my-web-app"
vpc_cidr        = "10.0.0.0/16"
instance_count  = 2
instance_type   = "ecs.t6-c1m1.large"
# main.tf - 主配置
module "network" {
  source       = "./modules/network"
  vpc_cidr     = var.vpc_cidr
  project_name = var.project_name
}

module "compute" {
  source          = "./modules/compute"
  vpc_id          = module.network.vpc_id
  vswitch_id      = module.network.vswitch_ids[0]
  sg_id           = module.network.security_group_id
  instance_type   = var.instance_type
  instance_count  = var.instance_count
  project_name    = var.project_name
}

module "database" {
  source     = "./modules/database"
  vpc_id     = module.network.vpc_id
  vswitch_id = module.network.vswitch_ids[1]
  sg_id      = module.network.security_group_id
}

# outputs.tf
output "web_url" {
  value = "http://${module.compute.elb_address}"
}

output "db_endpoint" {
  value     = module.database.connection_string
  sensitive = true
}
```


### 6.3 Terraform常用命令速查


| 命令 | 用途 |
| --- | --- |
| `terraform init` | 初始化，下载Provider和模块 |
| `terraform plan` | 预览变更（dry-run） |
| `terraform apply` | 执行变更 |
| `terraform apply -auto-approve` | 跳过确认执行变更 |
| `terraform destroy` | 销毁所有资源 |
| `terraform state list` | 列出State中的资源 |
| `terraform state show resource` | 显示资源详情 |
| `terraform import` | 导入已有资源到State |
| `terraform taint resource` | 标记资源强制重建 |
| `terraform output` | 显示输出值 |
| `terraform fmt` | 格式化配置文件 |
| `terraform validate` | 验证配置语法 |


<!-- Converted from: 01_Terraform入门.html -->
