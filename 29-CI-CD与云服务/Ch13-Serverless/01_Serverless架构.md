# Serverless架构 - 云服务与DevOps笔记


Serverless（无服务器架构）并非真的没有服务器，而是开发者无需管理服务器基础设施。云平台负责自动分配和管理计算资源，开发者只需关注业务逻辑代码。


### 1.1 Serverless的两大核心


| 核心组件 | 说明 | 代表产品 |
| --- | --- | --- |
| **FaaS（函数即服务）** | 按需执行函数代码，事件触发，自动扩缩容 | AWS Lambda、阿里云函数计算、Azure Functions |
| **BaaS（后端即服务）** | 使用托管的后端服务替代自建 | 数据库（DynamoDB）、存储（S3）、认证（Cognito） |


### 1.2 演进过程


```
物理服务器 → 虚拟机(EC2) → 容器(Docker/K8s) → Serverless(Lambda)
  ───────────────────────────────────────────────────────→
  管理负担递减     弹性递增     计费粒度递减
```


### 1.3 Serverless的核心特征


- **无需管理服务器**
   ：不关心OS、运行时环境的维护
- **按需执行**
   ：没有请求时不产生计算费用
- **自动弹性伸缩**
   ：从0到数千实例的瞬间扩展
- **按实际使用量计费**
   ：精确到函数调用次数和执行时长（毫秒级）
- **事件驱动**
   ：由HTTP请求、消息队列、文件上传等事件触发


### 2.1 主流FaaS平台对比


| 维度 | AWS Lambda | 阿里云函数计算(FC) | Azure Functions | Google Cloud Functions |
| --- | --- | --- | --- | --- |
| 支持语言 | Node/Python/Java/Go/.NET/Rust | Node/Python/Java/Go/PHP/C++ | C#/JS/Python/Java/Go | Node/Python/Java/Go/.NET |
| 最大执行时间 | 15分钟 | 10分钟(异步24h) | 无限(消耗计划) | 60分钟(2代) |
| 内存范围 | 128MB-10GB | 128MB-32GB | 1.5GB-14GB | 128MB-32GB |
| 计费粒度 | 1ms | 1ms | 1ms | 100ms |
| 并发模型 | 单请求/实例 | 多请求/实例 | 多请求/实例 | 单请求/实例 |
| 容器镜像 | 支持 | 支持 | 支持 | 支持 |


### 2.2 Lambda函数示例


```
# Python Lambda函数处理API请求
import json

def handler(event, context):
    """
    event: 触发事件数据（API Gateway传递的请求信息）
    context: 运行时上下文（内存限制、请求ID等）
    """
# 解析请求
    http_method = event.get('httpMethod', 'GET')
    path = event.get('path', '/')
    body = json.loads(event.get('body', '{}'))

    # 业务逻辑
    result = {
        "message": "Hello from Lambda!",
        "method": http_method,
        "path": path
    }

    # 返回HTTP响应
return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(result)
    }
```


### 2.3 阿里云函数计算示例


```
# index.py - 阿里云函数计算
import json

def handler(environ, start_response):
    # WSGI接口
    status = '200 OK'
    response_headers = [('Content-type', 'application/json')]

    body = json.dumps({
        "message": "Hello from Alibaba Cloud FC!"
    })

    start_response(status, response_headers)
    return [body.encode('utf-8')]

# 或使用HTTP触发器的简化接口
def hello(event, context):
    return {
        "statusCode": 200,
        "body": json.dumps({"msg": "hello"})
    }
```


### 3.1 什么是冷启动


冷启动（Cold Start）是指函数在长时间未被调用后，平台需要重新分配执行环境（包括下载代码、启动运行时、初始化依赖）而导致的延迟。这是Serverless架构最主要的性能挑战。


```
冷启动流程：
  事件触发 → 创建执行环境 → 下载代码 → 启动运行时 → 加载依赖 → 执行初始化 → 处理请求
  ────────────────────────────────────────────────────────────────────────────────
  典型延迟：100ms ~ 数秒（取决于语言、代码大小、依赖量）

  热启动流程：
  事件触发 → 复用已有环境 → 处理请求
  ───────────────────────
  延迟：通常 < 10ms
```


### 3.2 冷启动的影响因素


| 因素 | 影响 | 优化方向 |
| --- | --- | --- |
| 运行时语言 | Java冷启动最慢，Python/Node较快 | 选择轻量运行时 |
| 代码包大小 | 依赖越多，下载解压越慢 | 精简依赖，使用Layer |
| VPC配置 | 配置VPC会增加冷启动时间 | 使用VPC Endpoint |
| 内存配置 | 内存越大，CPU也越多 | 合理分配内存 |
| 初始化逻辑 | 全局变量/外部连接初始化 | 延迟初始化 |


### 3.3 冷启动优化策略


- **预置并发（Provisioned Concurrency）**
   ：预先分配一定数量的热实例，消除冷启动
- **使用轻量运行时**
   ：Node.js/Python优于Java，或使用GraalVM Native Image
- **减少代码包体积**
   ：Tree shaking、按需引入依赖
- **分层架构**
   ：使用Lambda Layer共享公共依赖
- **连接池复用**
   ：在全局作用域建立数据库连接，跨调用复用
- **保持活跃**
   ：定时触发函数保持环境温热（有额外成本）

**Java冷启动优化示例：**

- 使用AWS SnapStart（基于快照恢复）可将冷启动从5-10秒降至200ms以内
- 使用GraalVM编译为Native Image，启动时间降至毫秒级
- 使用Spring Cloud Function + Native编译


### 4.1 常见事件源


| 事件源 | 触发时机 | 应用场景 |
| --- | --- | --- |
| HTTP请求 | API Gateway收到请求 | REST API、Webhook |
| 对象存储 | 文件上传/删除 | 图片处理、数据导入 |
| 消息队列 | 收到消息 | 异步任务处理 |
| 定时触发 | Cron表达式 | 定时任务、数据清理 |
| 数据库变更 | 数据表记录变化 | 数据同步、缓存更新 |
| 日志 | 新日志写入 | 日志分析、告警 |
| IoT设备 | 设备数据上报 | 物联网数据处理 |


### 4.2 API Gateway触发Lambda


```
# serverless.yml - Serverless Framework配置
service: my-api

provider:
  name: aws
  runtime: python3.11
  region: ap-southeast-1

functions:
  hello:
    handler: handler.hello
    events:
      - http:
          path: /hello
          method: get
      - http:
          path: /hello
          method: post

  processImage:
    handler: handler.process_image
    events:
      - s3:
          bucket: my-images
          event: s3:ObjectCreated:*
          existing: true

  scheduledTask:
    handler: handler.cleanup
    events:
      - schedule: rate(1 hour)

  queueProcessor:
    handler: handler.process_queue
    events:
      - sqs:
          arn: arn:aws:sqs:region:account:my-queue
          batchSize: 10
```


### 5.1 解决的问题


当业务逻辑涉及多个步骤（调用多个Lambda、条件分支、错误重试、并行处理）时，直接在单个函数中编排会变得复杂。Step Functions提供了可视化的工作流编排能力。


### 5.2 核心状态类型


| 状态类型 | 说明 | 用途 |
| --- | --- | --- |
| Task | 执行一个工作单元（Lambda/API调用等） | 调用函数 |
| Choice | 根据条件分支 | 条件判断 |
| Parallel | 并行执行多个分支 | 并行处理 |
| Wait | 等待一段时间或到指定时间 | 延迟执行 |
| Map | 对数组每个元素执行 | 批量处理 |
| Pass | 透传数据，不做处理 | 数据转换 |
| Succeed/Fail | 标记成功/失败 | 流程结束 |


### 5.3 阿里云FnF（函数工作流）


阿里云的函数工作流（Function Flow, FnF）是与AWS Step Functions对应的Serverless编排服务。


```
# FnF流程定义示例
version: v1beta1
type: flow
steps:
  - type: parallel
    name: parallelProcess
    branches:
      - steps:
          - type: task
            name: processA
            resourceArn: acs:fc:cn-hangzhou::functions/processA
      - steps:
          - type: task
            name: processB
            resourceArn: acs:fc:cn-hangzhou::functions/processB

  - type: choice
    name: checkResult
    choices:
      - condition: $.result.status == "success"
        next: successStep
    default: retryStep

  - type: task
    name: successStep
    resourceArn: acs:fc:cn-hangzhou::functions/notifySuccess
    end: true
```


### 6.1 优势


- **零运维**
   ：无需管理服务器、操作系统、运行时环境
- **极致弹性**
   ：自动从0扩展到任意规模，无需预留容量
- **成本优化**
   ：空闲时不计费，精确到毫秒级计费
- **开发效率高**
   ：专注业务逻辑，快速上线
- **高可用**
   ：平台自动跨AZ部署，内置容灾
- **快速迭代**
   ：独立部署每个函数，互不影响


### 6.2 劣势


- **冷启动延迟**
   ：首次调用或长时间未调用后的延迟
- **执行时间限制**
   ：通常最长15分钟
- **调试困难**
   ：本地环境与云端环境差异，日志分散
- **厂商锁定**
   ：不同云平台的API和触发器不兼容
- **状态管理**
   ：无状态设计，外部状态需依赖数据库/缓存
- **复杂业务逻辑**
   ：涉及多步骤编排时架构复杂度上升
- **成本不可预测**
   ：突发高流量可能导致意外账单


### 6.3 适用与不适用场景


| 适用场景 | 不适用场景 |
| --- | --- |
| REST API / Webhook | 长时间运行的计算任务（>15分钟） |
| 事件驱动的数据处理 | 需要大量内存/GPU的计算 |
| 定时任务（替代Cron） | 低延迟要求的实时系统 |
| 文件/图片处理流水线 | 状态密集型应用（游戏服务器等） |
| Chatbot / Webhook处理器 | 需要固定IP的出站请求 |
| ETL数据管道 | 高频调用的低延迟API |
| IoT数据采集和处理 | 需要本地硬件访问的应用 |

**Serverless Framework 与 Serverless工具：**

- **Serverless Framework**
   ：最流行的Serverless部署框架，支持AWS/Azure/GCP/阿里云
- **AWS SAM**
   ：AWS官方的Serverless应用模型
- **AWS CDK**
   ：使用编程语言定义云基础设施
- **阿里云 Serverless Devs**
   ：阿里云官方Serverless开发者工具
- **esbuild/Webpack**
   ：代码打包优化，减小部署包体积


<!-- Converted from: 01_Serverless架构.html -->
