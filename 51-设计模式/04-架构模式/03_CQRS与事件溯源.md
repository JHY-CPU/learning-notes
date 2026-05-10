# CQRS 与事件溯源 (Event Sourcing)


## 1. CQRS 核心思想


#### 传统 CRUD 模式


读写共享同一数据模型和数据库。简单但存在瓶颈：


- 读写争用同一资源
- 模型既要处理写入又要优化读取
- 难以独立扩展读/写


#### CQRS 模式


将命令(写)与查询(读)分离为不同模型：


- Command 端：处理业务逻辑、验证、持久化
- Query 端：独立的读模型，可异步更新
- 读写可使用不同存储技术


### CQRS 架构流程

Client
→
Command API
→
Command Handler
→
Write DB
Client
→
Query API
→
Query Handler
→
Read DB (物化视图)

```
// Command 端 — 写模型
public class CreateOrderCommand {
    public Guid CustomerId { get; set; }
    public List<OrderItem> Items { get; set; }
}

public class OrderCommandHandler {
    public async Task Handle(CreateOrderCommand cmd) {
        var order = Order.Create(cmd.CustomerId, cmd.Items);
        await _writeRepo.Save(order);
        await _eventBus.Publish(new OrderCreatedEvent(order));
    }
}

// Query 端 — 读模型
public class OrderSummaryQuery {
    public Guid CustomerId { get; set; }
}

public class OrderQueryHandler {
    public async Task<List<OrderSummaryDTO>> Handle(OrderSummaryQuery q) {
        // 直接从优化过的读视图查询
        return await _readDb.OrderSummaries
            .Where(o => o.CustomerId == q.CustomerId).ToListAsync();
    }
}
```


## 2. 事件溯源 (Event Sourcing)

**核心思想：**
不存储"当前状态"，而是存储导致当前状态的所有事件。当前状态通过重放事件得到。

#### 事件存储 (Event Store)


- 只追加 (Append-Only) 的事件日志
- 每个事件不可变 (Immutable)
- 按聚合根 ID + 版本号有序存储
- 代表：EventStoreDB, Marten, DynamoDB


#### 聚合重建


- 加载聚合的所有历史事件
- 按顺序应用到聚合根
- 得到当前状态
- 可用快照 (Snapshot) 优化性能


#### 投射 (Projection)


- 消费事件流，构建读模型
- 同一事件可生成多种视图
- 异步处理，最终一致
- 可随时重建（从事件重放）


```
// 事件定义
public class AccountOpened {
    public Guid AccountId { get; }
    public decimal InitialBalance { get; }
    public DateTime OpenedAt { get; }
}
public class MoneyDeposited {
    public Guid AccountId { get; }
    public decimal Amount { get; }
}
public class MoneyWithdrawn {
    public Guid AccountId { get; }
    public decimal Amount { get; }
}

// 聚合根 — 通过事件重建状态
public class BankAccount {
    public Guid Id { get; private set; }
    public decimal Balance { get; private set; }

    // 应用事件来改变状态
    private void Apply(AccountOpened e) {
        Id = e.AccountId; Balance = e.InitialBalance;
    }
    private void Apply(MoneyDeposited e) { Balance += e.Amount; }
    private void Apply(MoneyWithdrawn e) { Balance -= e.Amount; }

    // 业务命令产生事件
    public MoneyDeposited Deposit(decimal amount) {
        if (amount <= 0) throw new Exception("金额须大于零");
        var evt = new MoneyDeposited { AccountId = Id, Amount = amount };
        Apply(evt); // 立即应用
        return evt;
    }

    // 从事件流重建
    public static BankAccount LoadFrom(List<object> events) {
        var account = new BankAccount();
        foreach (var e in events) {
            ((dynamic)account).Apply((dynamic)e);
        }
        return account;
    }
}
```


## 3. 最终一致性 (Eventual Consistency)


CQRS + ES 中，读模型通过异步消费事件更新，因此读写之间存在短暂延迟。


#### 一致性保障手段


- 事件版本号乐观锁
- 幂等消费者（去重表/事件ID）
- 死信队列 + 告警
- 进度跟踪（Checkpoint / Projection Position）


#### 快照 (Snapshot) 优化


- 每 N 个事件保存一次状态快照
- 重建时从最近快照 + 后续事件
- 大幅减少重放事件数量
- 快照间隔取决于业务变更频率


## 4. 适用场景与权衡


| 维度 | 适合 CQRS+ES | 不适合 |
| --- | --- | --- |
| 业务复杂度 | 复杂领域、多读视图 | 简单 CRUD 应用 |
| 一致性要求 | 可接受最终一致 | 需要强一致性 |
| 审计需求 | 需要完整操作历史 | 只需当前状态 |
| 团队经验 | 熟悉 DDD 和事件驱动 | 初学者团队 |
| 读写比例 | 读多写少、读写差异大 | 读写均衡 |

**设计建议：**
并非所有模块都需要 CQRS。可在同一系统中对核心领域使用 CQRS+ES，其他部分保持传统 CRUD。


<!-- Converted from: 03_CQRS与事件溯源.html -->
