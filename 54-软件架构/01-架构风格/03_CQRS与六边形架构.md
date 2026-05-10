# C4模型、六边形架构、整洁架构与洋葱架构


## 1. C4 模型 (Context, Container, Component, Code)

**核心思想：**
用四个层级的抽象来描述软件架构，类似地图的缩放——从全球视图逐步放大到街道视图。

| 层级 | 视角 | 关注点 | 受众 |
| --- | --- | --- | --- |
| L1 Context（系统上下文） | 系统边界 | 系统与外部用户/系统的关系 | 所有人 |
| L2 Container（容器） | 技术选型 | 应用、数据库、消息队列等可部署单元 | 架构师、技术负责人 |
| L3 Component（组件） | 模块划分 | 容器内的主要模块/服务 | 开发团队 |
| L4 Code（代码） | 实现细节 | 类图、关键方法 | 开发人员 |


### C4 图示例（L2 Container）


```
┌──────────────────────────────────────────────────────┐
│                    电商系统 (L1 Context)               │
│                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ Web 前端  │───→│ API Gateway  │───→│ 用户服务    │ │
│  │ (React)   │    │ (Kong/Nginx) │    │ (Spring)   │ │
│  └──────────┘    └──────┬───────┘    └─────┬──────┘ │
│                         │                   │        │
│                  ┌──────▼───────┐    ┌──────▼──────┐ │
│                  │ 订单服务      │    │ PostgreSQL  │ │
│                  │ (Spring)     │    │ (Users DB)  │ │
│                  └──────┬───────┘    └─────────────┘ │
│                         │                            │
│                  ┌──────▼───────┐                    │
│                  │   Kafka      │                    │
│                  │ (消息队列)    │                    │
│                  └──────────────┘                    │
└──────────────────────────────────────────────────────┘
```


```
// 使用 Structurizr (C4 模型 DSL) 定义
workspace {
    model {
        user = person "用户"
        system = softwareSystem "电商系统" {
            web = container "Web 前端" "React"
            api = container "API Gateway" "Kong"
            orderSvc = container "订单服务" "Spring Boot"
            db = container "PostgreSQL" "关系型数据库"
        }
        user -> web "浏览商品"
        web -> api "API 调用"
        api -> orderSvc "订单操作"
        orderSvc -> db "读写数据"
    }
}
```


## 2. 六边形架构 (Hexagonal / Ports & Adapters)

**核心思想（Alistair Cockburn）：**
将业务逻辑放在中心，通过"端口"(Port) 定义接口，通过"适配器"(Adapter) 连接外部世界。使核心逻辑与技术实现完全解耦。

### 架构结构


```
┌─────────────────────────────────┐
        │        外部适配器 (Driving)        │
        │   REST Controller / CLI / GUI     │
        └──────────────┬──────────────────┘
                       │ 调用
        ┌──────────────▼──────────────────┐
        │       端口 (接口定义)              │
        │   Input Port (Use Case 接口)      │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │     核心领域 (Domain/Business)     │
        │   实体、值对象、领域服务            │
        │   ★ 不依赖任何外部技术 ★           │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │       端口 (接口定义)              │
        │   Output Port (Repository 接口)   │
        └──────────────┬──────────────────┘
                       │ 实现
        ┌──────────────▼──────────────────┐
        │        外部适配器 (Driven)         │
        │   JPA Repository / Redis / MQ     │
        └─────────────────────────────────┘
```


```
// 端口 — 纯接口，属于核心层
public interface CreateOrderUseCase { // 输入端口
    OrderId create(CreateOrderCommand cmd);
}
public interface OrderRepository {   // 输出端口
    void save(Order order);
    Order findById(OrderId id);
}

// 适配器 — 驱动侧 (Inbound)
@RestController
public class OrderController {
    private final CreateOrderUseCase useCase;
    @PostMapping("/orders")
    public Response create(@RequestBody CreateOrderRequest req) {
        var id = useCase.create(req.toCommand());
        return Response.ok(id);
    }
}

// 适配器 — 被驱动侧 (Outbound)
@Repository
public class JpaOrderAdapter implements OrderRepository {
    public void save(Order order) {
        jpaRepo.save(toEntity(order)); // 转换为 JPA 实体
    }
}
```


## 3. 整洁架构 (Clean Architecture — Uncle Bob)


### 同心圆层级

实体层 (Entities) — 企业业务规则
用例层 (Use Cases) — 应用业务规则
接口适配器层 (Interface Adapters) — Controller, Presenter, Gateway
框架与驱动层 (Frameworks & Drivers) — Web, DB, External
**依赖规则：**
源码依赖只能向内。外层可以依赖内层，内层绝不依赖外层。内层不知道外层的存在。

```
// 整洁架构的目录结构
src/
├── domain/                    // 实体层（最内层）
│   ├── entities/
│   │   ├── Order.java         // 企业实体
│   │   └── OrderItem.java
│   └── valueobjects/
│       └── Money.java
├── usecases/                  // 用例层
│   ├── CreateOrderUseCase.java
│   ├── impl/
│   │   └── CreateOrderInteractor.java
│   └── ports/
│       ├── OrderRepository.java   // 输出端口
│       └── PaymentGateway.java
├── adapters/                  // 接口适配器层
│   ├── web/
│   │   └── OrderController.java   // 入站适配器
│   └── persistence/
│       └── JpaOrderAdapter.java   // 出站适配器
└── infrastructure/            // 框架层（最外层）
    ├── config/
    └── SpringBootApplication.java
```


## 4. 洋葱架构 (Onion Architecture — Jeffrey Palermo)


#### 与整洁架构的相似


- 核心层不依赖外层
- 领域模型在最内层
- 依赖注入连接各层
- 接口定义在内层


#### 与整洁架构的差异


- 强调"领域服务"作为一等公民
- 基础设施层是外环而非底层
- DTO 在最外层而非跨层
- 强调仓储模式


## 5. 四种方法对比


| 维度 | C4 模型 | 六边形架构 | 整洁架构 | 洋葱架构 |
| --- | --- | --- | --- | --- |
| 本质 | 可视化表示法 | 架构风格 | 架构原则 | 架构风格 |
| 核心理念 | 分层可视化 | 端口与适配器 | 依赖规则 | 领域为中心 |
| 边界机制 | 层级缩放 | Port 接口 | 同心圆层 | 接口 + DI |
| 关注点 | 沟通与文档 | 可测试性 | 可维护性 | 领域纯粹性 |
| 可否组合 | 可与其他配合 | 可与整洁组合 | 可与 C4 配合 | 与整洁高度相似 |

**实践建议：**
C4 用于架构文档和团队沟通；六边形/整洁/洋葱三者本质相同（反转依赖 + 接口隔离），选择团队最熟悉的即可。关键在于：领域逻辑不依赖框架、外部接口通过抽象访问。


<!-- Converted from: 03_CQRS与六边形架构.html -->
