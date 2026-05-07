# 架构模式 (Architectural Patterns)

> 软件系统的高层组织方式，定义骨架与组成要素间的关系。

---

## 1. MVC模式

Model（数据与逻辑）- View（展示）- Controller（协调）。

### 架构描述
```
Controller ──调用──> Model
  │                      │ 通知
  └──控制──> View <──────┘
```
适用于 Web 应用（Spring MVC）、桌面应用。

### 代码示例
```java
public class UserController {
    private UserModel model; private UserView view;
    public void setUsername(String n) { model.setUsername(n); }
    public void updateView() { view.display(model.getUsername()); }
}
```
**优点**：职责分离，支持多视图。**缺点**：Controller 可能臃胀。

---

## 2. MVP模式

Presenter 作为中间人，View 通过接口通信。

### 架构
```
View ──接口──> Presenter ──调用──> Model
View <──被动更新── Presenter
```
适用于 Android 原生开发、高测试性要求场景。

### 代码示例
```java
public interface UserView { String getUsername(); void showUser(String n); }
public class UserPresenter {
    public void loadUser() { view.showUser(view.getUsername()); }
}
```
**优点**：完全解耦。**缺点**：接口多，样板代码多。

---

## 3. MVVM模式

ViewModel 通过数据绑定自动同步 View。

### 架构
```
Model <──> ViewModel <=数据绑定=> View
```
适用于 Vue.js、Angular、WPF、SwiftUI。

### 代码示例（Vue.js）
```vue
<input v-model="username">
<p>邮箱: {{ email }}</p>
```
**优点**：数据驱动，声明式。**缺点**：调试困难。

---

## 4. 分层架构

每层只与相邻层通信，上层依赖下层。

### 架构
```
表现层(Controller) -> 业务层(Service) -> 持久层(Repository) -> 数据库
```
适用于企业级 Web 应用，最为经典的架构风格。

### 代码示例（Spring Boot）
```java
@RestController -> @Service -> @Repository extends JpaRepository
```
**优点**：结构清晰，易于分工。**缺点**：严格分层有性能开销。

---

## 5. 管道-过滤器模式

将处理分解为独立过滤器，数据通过管道流动。

### 代码示例
```java
public class Pipeline<T> {
    private List<Filter<T>> filters = new ArrayList<>();
    public Pipeline<T> add(Filter<T> f) { filters.add(f); return this; }
    public T execute(T input) {
        T result = input;
        for (Filter<T> f : filters) result = f.process(result);
        return result;
    }
}
```
适用于 ETL、编译器、日志处理。**优点**：可重组、可并发。

---

## 6. 事件驱动架构

组件通过事件松耦合通信。

### 代码示例
```java
public class EventBus {
    public <T> void on(Class<T> type, Consumer<T> h) { /* 注册 */ }
    public <T> void emit(T event) { /* 分发 */ }
}
```
适用于实时系统、微服务异步通信。**优点**：高度解耦。

---

## 7. 发布-订阅模式

通过消息代理（topic/channel）解耦发布者与订阅者。

### 与观察者区别
观察者直接关联、同步；发布-订阅通过中间件、异步、支持跨进程。中间件：RabbitMQ、Kafka。

### 优缺点
- **优点**：完全解耦，支持跨进程
- **缺点**：引入中间件复杂度，消息可靠性需保证

---

## 8. CQRS模式

读操作（Query）和写操作（Command）分离到不同模型。

### 架构
```
Command -> 写模型 -> 写库 ──同步──> 读库 <- 读模型 <- Query
```
适用于读多写少系统，常与事件溯源结合。

### 代码示例
```java
// 写端: repo.save(new Order(cmd)); bus.publish(new OrderCreatedEvent());
// 读端: readRepo.findById(id) // 独立优化
```
**优点**：读写独立扩展。**缺点**：最终一致性，维护两套存储。

---

## 9. 六边形架构

核心逻辑在中心，通过端口和适配器与外部交互。

### 架构
```
外部适配器(Web/CLI) ──> API端口 ──> 核心领域 <── SPI端口 <── 外部适配器(DB/MQ)
```
适用于 DDD、多接口系统。

### 代码示例
```java
public interface OrderUseCase { OrderDTO create(CreateOrderRequest r); } // API端口
public interface OrderRepository { void save(Order o); } // SPI端口
public class OrderService implements OrderUseCase { /* 核心逻辑，只依赖接口 */ }
```
**优点**：核心独立于框架。**缺点**：接口多，搭建成本高。

---

## 10. 整洁架构

Robert Martin 提出，依赖由外向内，内层不依赖外层。

### 层次（同心圆外到内）
```
框架与驱动 -> 接口适配器 -> 用例 -> 实体
```

| 层 | 职责 |
|---|------|
| 实体 | 领域模型、核心规则 |
| 用例 | 应用服务、业务流程 |
| 适配器 | Controller、Repository实现 |
| 框架 | Web、DB、MQ 等外部细节 |

适用于大型长期项目。**优点**：可测试、框架无关。**缺点**：间接层多。

---

## 11. 微内核架构

核心提供最小功能集，通过插件扩展。

### 架构
```
Plugin A │ Plugin B │ Plugin C   ← 插件层
──────────────────────────────
通信机制 │ 生命周期管理           ← 微内核
```
适用于 IDE 插件系统（Eclipse、VS Code）、浏览器扩展、CMS。

### 代码示例
```java
public interface Plugin { void init(); void execute(); void shutdown(); }
public class MicroKernel {
    public void register(Plugin p) { p.init(); }
    public void run(String name) { /* 调用插件 */ }
}
```
**优点**：高度模块化。**缺点**：核心设计复杂。

---

## 12. 各架构模式对比

| 模式 | 核心特点 | 最佳场景 | 复杂度 |
|------|---------|---------|:------:|
| MVC | 数据/展示/控制分离 | Web、桌面 | 低 |
| MVP | View-Model 完全解耦 | Android、测试驱动 | 中 |
| MVVM | 数据绑定驱动 | 前端、声明式UI | 中 |
| 分层架构 | 严格依赖方向 | 企业应用、CRUD | 低 |
| 管道-过滤器 | 数据流处理 | ETL、编译器 | 中 |
| 事件驱动 | 异步事件通信 | 实时系统 | 中 |
| 发布-订阅 | 中间件解耦 | 微服务通信 | 中 |
| CQRS | 读写分离 | 读多写少 | 高 |
| 六边形 | 核心与外部隔离 | DDD、多接口 | 高 |
| 整洁架构 | 同心圆依赖 | 大型长期项目 | 高 |
| 微内核 | 插件化扩展 | 平台产品 | 高 |

**选型**：简单 Web 选分层+MVC；前端选 MVVM；微服务选事件驱动+PubSub；高性能查询选 CQRS；长期项目选整洁/六边形；平台产品选微内核。

**演进路径**：分层架构 -> MVC+分层 -> 六边形/整洁架构 -> 微内核+微服务。

---

[返回上级目录](../README.md)
