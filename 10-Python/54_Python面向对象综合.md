# Python面向对象综合


## 🏗️ Python 面向对象综合


OOP 四大特性 (封装/继承/多态/抽象)、SOLID 原则、综合实战案例 (银行系统)、设计模式简介 (单例/工厂/策略)、Python OOP 风格总结。


## OOP 四大特性回顾


```
// ========== 封装 ==========
// 隐藏内部实现,对外提供接口
// Python 用 _单下划线 (约定) 和 __双下划线 (名称修饰)

class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner         # 公开
        self._balance = balance    # 保护 (约定内部使用)
        self.__pin = "0000"        # 私有 (名称修饰)

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            return True
        return False

    def get_balance(self):
        return self._balance

    # 属性封装:
    @property
    def balance(self):
        return self._balance

// 名称修饰:
print(dir(BankAccount("test")))
# _BankAccount__pin  ← 实际存储的名称!
# 所以 __pin 不是真正的私有,只是改名了

// ========== 继承 ==========
// 子类复用/扩展父类

class SavingsAccount(BankAccount):
    interest_rate = 0.03

    def apply_interest(self):
        interest = self._balance * self.interest_rate
        self._balance += interest
        return interest

// ========== 多态 ==========
// 同一接口不同实现

class Account(ABC):
    @abstractmethod
    def withdraw(self, amount):
        pass

class CheckingAccount(Account):
    def withdraw(self, amount):
        return f"取款 {amount} (活期)"

class CreditAccount(Account):
    def withdraw(self, amount):
        return f"取款 {amount} (信用卡)"

// ========== 抽象 ==========
// 隐藏复杂细节,暴露简单接口

class PaymentGateway(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

class Alipay(PaymentGateway):
    def pay(self, amount):
        return f"支付宝支付 {amount}元"

class WechatPay(PaymentGateway):
    def pay(self, amount):
        return f"微信支付 {amount}元"
```


## SOLID 原则


```
// ========== S — 单一职责 ==========
// 一个类只负责一件事

// ❌ 不好的: 用户类做了太多事
class User:
    def __init__(self, name): self.name = name
    def save(self): ...        # 数据库操作
    def send_email(self): ...  # 邮件操作
    def validate(self): ...    # 验证操作

// ✅ 好的: 每个类单一职责
class User:
    def __init__(self, name): self.name = name

class UserRepository:
    def save(self, user): ...

class EmailService:
    def send(self, user, msg): ...

// ========== O — 开闭原则 ==========
// 对扩展开放,对修改关闭

# 用策略模式替代 if-else
class Discount:
    def apply(self, price):
        return price

class NoDiscount(Discount):
    def apply(self, price):
        return price

class PercentDiscount(Discount):
    def __init__(self, percent):
        self.percent = percent
    def apply(self, price):
        return price * (1 - self.percent)

// ========== L — 里氏替换 ==========
// 子类必须能替换父类

class Rectangle:
    def __init__(self, w, h):
        self.w, self.h = w, h
    def set_width(self, w): self.w = w
    def set_height(self, h): self.h = h
    def area(self): return self.w * self.h

// ❌ 违反 LSP: Square 改变了行为
class Square(Rectangle):
    def set_width(self, w):
        self.w = w
        self.h = w           # 改高度? 违反预期!

// ========== I — 接口隔离 ==========
// 不要强迫类实现不需要的接口

// ❌ 胖接口
class Worker(ABC):
    @abstractmethod
    def work(self): pass
    @abstractmethod
    def eat(self): pass       # RobotWorker 不需要!

// ✅ 细接口
class Workable(ABC):
    @abstractmethod
    def work(self): pass

class Eatable(ABC):
    @abstractmethod
    def eat(self): pass

// ========== D — 依赖反转 ==========
// 依赖抽象,不依赖具体实现

# ❌ 依赖具体实现
class NotificationService:
    def __init__(self):
        self.email = EmailSender()  # 紧耦合

# ✅ 依赖抽象
class NotificationService:
    def __init__(self, sender: MessageSender):
        self.sender = sender  # 可以注入任何实现
```


## 综合案例: 银行系统


```
// ========== 银行系统 ==========
// 综合运用: 封装/继承/多态/抽象/组合

from abc import ABC, abstractmethod
from datetime import datetime
from functools import total_ordering

# ---------- 异常 ----------
class InsufficientFundsError(Exception):
    pass

class AccountError(Exception):
    pass

# ---------- 交易记录 ----------
@total_ordering
class Transaction:
    def __init__(self, amount, type_, description=""):
        self.amount = amount
        self.type = type_            # "deposit", "withdraw", "transfer"
        self.description = description
        self.timestamp = datetime.now()

    def __eq__(self, other):
        return self.timestamp == other.timestamp

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __repr__(self):
        return f"[{self.timestamp:%H:%M}] {self.type}: {self.amount:+} {self.description}"

# ---------- 账户基类 ----------
class Account(ABC):
    def __init__(self, account_id, owner):
        self.account_id = account_id
        self.owner = owner
        self._balance = 0.0
        self._transactions = []
        self._is_active = True

    @property
    def balance(self):
        return self._balance

    @abstractmethod
    def get_type_name(self):
        pass

    def deposit(self, amount):
        if not self._is_active:
            raise AccountError("账户已关闭")
        if amount <= 0:
            raise ValueError("存款金额必须为正数")
        self._balance += amount
        self._transactions.append(Transaction(amount, "deposit"))
        return self._balance

    def withdraw(self, amount):
        if not self._is_active:
            raise AccountError("账户已关闭")
        if amount <= 0:
            raise ValueError("取款金额必须为正数")
        if amount > self._balance:
            raise InsufficientFundsError(f"余额不足: {self._balance}")
        self._balance -= amount
        self._transactions.append(Transaction(-amount, "withdraw"))
        return self._balance

    def transfer(self, target, amount):
        self.withdraw(amount)
        target.deposit(amount)
        self._transactions.append(Transaction(-amount, "transfer", f"→ {target.account_id}"))

    def close(self):
        self._is_active = False

    def print_statement(self):
        print(f"对账单: {self.owner} ({self.account_id})")
        print("-" * 40)
        for t in sorted(self._transactions):
            print(f"  {t}")
        print(f"余额: {self._balance}")

# ---------- 子类 ----------
class SavingsAccount(Account):
    def __init__(self, account_id, owner, interest_rate=0.03):
        super().__init__(account_id, owner)
        self.interest_rate = interest_rate

    def get_type_name(self):
        return "储蓄账户"

    def apply_interest(self):
        interest = self._balance * self.interest_rate
        self._balance += interest
        return interest

class CheckingAccount(Account):
    def __init__(self, account_id, owner, overdraft_limit=1000):
        super().__init__(account_id, owner)
        self.overdraft_limit = overdraft_limit

    def get_type_name(self):
        return "活期账户"

    def withdraw(self, amount):       # 重写: 支持透支
        if amount > self._balance + self.overdraft_limit:
            raise InsufficientFundsError("超过透支限额")
        self._balance -= amount
        self._transactions.append(Transaction(-amount, "withdraw"))
        return self._balance

# ---------- 使用演示 ----------
savings = SavingsAccount("SA001", "Alice", 0.04)
checking = CheckingAccount("CA001", "Alice")

savings.deposit(10000)
savings.apply_interest()
savings.transfer(checking, 2000)
savings.print_statement()

checking.withdraw(500)
print(f"活期余额: {checking.balance}")
```


## Python OOP 风格总结


```
// ========== Python OOP 特有风格 ==========
// 1. 属性用 @property,不用 getter/setter
// 2. 鸭子类型优于抽象基类 (EAFP)
// 3. 组合优于继承
// 4. Mixin 提供可复用功能
// 5. 魔术方法让自定义类有原生类型的行为

// ========== 类设计检查清单 ==========
// ✓ __init__ 初始化所有实例属性
// ✓ __repr__ 方便调试
// ✓ @property 封装需要验证的属性
// ✓ 不变属性设为只读 (只有 getter)
// ✓ __eq__ + __hash__ 如果做相等比较
// ✓ @classmethod 工厂方法
// ✓ slot 优化性能 (如果需要大量实例)

// ==========  vs Java 对比 ==========
// Java                Python
// private int x;      self._x (约定)
// getX()              @property
// setX()              @x.setter
// implements          ABC + @abstractmethod
// extends             class Child(Parent)
// @Override           直接重写
// interface           ABC (或鸭子类型)
// static              @staticmethod
// generic          ️ 不需要 (鸭子类型)

// ========== 推荐资源 ==========
// 1. Fluent Python — Python OOP 圣经
// 2. Python Cookbook — 实用模式
// 3. Design Patterns in Python
// 4. SOLID Python 原则 — 写出可维护代码
```


> **Note:** 💡 OOP 总结: (1) Python OOP 灵活轻量,善用 @property 和魔术方法; (2) 遵循 SOLID 原则写出可维护代码; (3) 鸭子类型 (有方法就行) 是 Python 的核心哲学; (4) 组合优于继承,Mixin 模式增加灵活性; (5) 设计模式在 Python 中通常更简洁 (函数+闭包替代类)。


## 练习


<!-- Converted from: 54_Python面向对象综合.html -->
