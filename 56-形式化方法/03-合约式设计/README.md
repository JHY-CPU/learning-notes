# 03-合约式设计

## 1. 前置条件与后置条件

合约式设计（Design by Contract，DbC）由Bertrand Meyer在Eiffel语言中提出，核心思想是软件组件之间如同商业合约：双方各有权责。

### 前置条件（Precondition）

调用者在调用方法前必须满足的条件。

```python
def withdraw(amount):
    """从账户取款"""
    # 前置条件：金额必须为正数，且余额充足
    assert amount > 0, "取款金额必须为正数"
    assert self.balance >= amount, "余额不足"
    self.balance -= amount
```

### 后置条件（Postcondition）

方法执行完毕后必须保证的条件。

```python
def withdraw(amount):
    """从账户取款"""
    old_balance = self.balance  # 保存旧值
    
    assert amount > 0  # 前置条件
    assert self.balance >= amount
    
    self.balance -= amount
    
    # 后置条件
    assert self.balance == old_balance - amount
```

### 合约的责任分配

```
调用者（Client）          被调用者（Supplier）
┌──────────────────┐    ┌──────────────────┐
│ 满足前置条件      │ →  │ 保证后置条件      │
│                  │    │ 维护不变量        │
│ 如果违反前置条件   │    │                  │
│ → 调用者的错      │    │ 如果违反后置条件   │
│                  │    │ → 被调用者的错     │
└──────────────────┘    └──────────────────┘
```

## 2. 类不变量

类不变量（Class Invariant）是对象在所有稳定状态下必须满足的条件。

### 定义

- 对象创建后必须满足不变量
- 每个公开方法执行前后必须满足不变量
- 对象销毁前必须满足不变量

### 示例

```python
class Stack:
    def __init__(self, capacity):
        self.items = []
        self.capacity = capacity
        self._check_invariant()
    
    def _check_invariant(self):
        assert 0 <= len(self.items) <= self.capacity
        assert self.capacity > 0
    
    def push(self, item):
        self._check_invariant()
        assert len(self.items) < self.capacity  # 前置条件
        old_size = len(self.items)
        self.items.append(item)
        # 后置条件
        assert len(self.items) == old_size + 1
        assert self.items[-1] == item
        self._check_invariant()
    
    def pop(self):
        self._check_invariant()
        assert len(self.items) > 0  # 前置条件
        old_size = len(self.items)
        result = self.items.pop()
        # 后置条件
        assert len(self.items) == old_size - 1
        self._check_invariant()
        return result
```

### 不变量的继承

子类可以加强不变量（添加额外约束），但不能削弱已有的不变量。

## 3. Eiffel语言中的合约

Eiffel是第一种原生支持合约式设计的编程语言。

### 语法

```eiffel
class ACCOUNT
feature
    balance: INTEGER
    
    withdraw (amount: INTEGER)
        require           -- 前置条件
            positive: amount > 0
            sufficient: balance >= amount
        do
            balance := balance - amount
        ensure            -- 后置条件
            decreased: balance = old balance - amount
        end

invariant                 -- 类不变量
    non_negative: balance >= 0
end
```

### Eiffel的合约特性

- `require`：前置条件
- `ensure`：后置条件
- `invariant`：类不变量
- `old`：引用方法执行前的值
- 合约继承：子类自动继承父类的合约

### 合约继承规则

| 规则 | 说明 |
|------|------|
| 前置条件 | 子类可以**弱化**（使用OR组合） |
| 后置条件 | 子类可以**强化**（使用AND组合） |
| 不变量 | 子类可以**强化**（使用AND组合） |

这与Liskov替换原则一致：子类必须满足父类的契约。

## 4. Java中的合约（assert、JML）

### Java assert机制

Java从1.4开始支持 `assert` 关键字。

```java
public class Calculator {
    /**
     * 计算阶乘
     * @param n 非负整数
     * @return n的阶乘
     */
    public static long factorial(int n) {
        // 前置条件
        assert n >= 0 : "n must be non-negative, got: " + n;
        
        long result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        
        // 后置条件（简单检查）
        assert result > 0 || n == 0 : "result must be positive";
        return result;
    }
}
```

**注意**：assert默认禁用，需使用 `-ea` 参数启用。

### JML（Java Modeling Language）

JML为Java提供完整的合约式设计支持。

```java
public class BankAccount {
    //@ public invariant balance >= 0;
    private int balance;
    
    /*@ requires amount > 0;
      @ requires balance >= amount;
      @ assignable balance;
      @ ensures balance == \old(balance) - amount;
    */
    public void withdraw(int amount) {
        balance -= amount;
    }
    
    /*@ requires amount > 0;
      @ assignable balance;
      @ ensures balance == \old(balance) + amount;
    */
    public void deposit(int amount) {
        balance += amount;
    }
}
```

### JML工具

- **OpenJML**：JML规范的参考实现，支持运行时检查和静态验证
- **KeY**：针对JML的交互式定理证明器
- **ESC/Java2**：增强型静态检查工具

## 5. Design by Contract在API设计中的应用

### API合约的层次

```
1. 接口层：方法签名和参数约束
2. 行为层：前置/后置条件和副作用
3. 错误层：违反合约时的行为
```

### REST API中的合约

```yaml
# OpenAPI 规范中的合约示例
paths:
  /users/{id}:
    get:
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: integer
            minimum: 1          # 前置条件
      responses:
        '200':
          description: 成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  id:
                    type: integer
                  name:
                    type: string
                    minLength: 1  # 后置条件
        '404':
          description: 用户不存在
```

### API版本演进中的合约

| 变更类型 | 对合约的影响 |
|---------|------------|
| 增加可选参数 | 合法——不违反已有合约 |
| 放宽前置条件 | 合法——向后兼容 |
| 收紧前置条件 | 不兼容——可能破坏客户端 |
| 增强后置条件 | 合法——提供更多信息 |
| 减弱后置条件 | 不兼容——客户端可能依赖 |

### 微服务间的合约

```
服务A（调用者）         合约           服务B（提供者）
┌──────────────┐    ┌─────────┐    ┌──────────────┐
│ 保证请求格式   │ ←→ │ 接口定义 │ ←→ │ 保证响应格式  │
│ 处理超时      │    │ SLA     │    │ 保证性能     │
│ 处理错误码    │    │ 错误码  │    │ 返回正确错误码│
└──────────────┘    └─────────┘    └──────────────┘
```

### 实践建议

1. **显式定义合约**：在接口文档中明确前置/后置条件
2. **自动化验证**：使用工具在运行时检查合约
3. **合约优先开发**：先定义合约，再实现功能
4. **错误信息友好**：违反合约时提供有意义的错误信息
5. **渐进式引入**：从关键模块开始引入合约，逐步扩展
