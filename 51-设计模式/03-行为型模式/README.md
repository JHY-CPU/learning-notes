# 行为型模式 (Behavioral Patterns)

> 关注对象之间的通信与职责分配。

---

## 1. 策略模式

### 定义与场景
定义一系列可互换的算法。适用于支付方式切换、排序策略、折扣规则。

### 代码示例
```java
public interface PaymentStrategy { void pay(double amount); }
public class AlipayPayment implements PaymentStrategy {
    public void pay(double a) { System.out.println("支付宝支付 " + a); }
}
public class PaymentContext {
    private PaymentStrategy strategy;
    public PaymentContext(PaymentStrategy s) { this.strategy = s; }
    public void execute(double a) { strategy.pay(a); }
}
// Java 8+ Lambda: new PaymentContext(a -> System.out.println("支付" + a));
```

### 优缺点
- **优点**：消除条件分支；可运行时切换；符合开闭原则
- **缺点**：客户端需了解所有策略；类数量增多

---

## 2. 观察者模式

### 定义与场景
一对多依赖，状态变化时自动通知。适用于 GUI 事件、消息队列、数据绑定。

### 代码示例
```java
public interface Observer { void update(String event, Object data); }
public class EventBus {
    private Map<String, List<Observer>> map = new HashMap<>();
    public void subscribe(String e, Observer o) {
        map.computeIfAbsent(e, k -> new ArrayList<>()).add(o);
    }
    public void publish(String e, Object d) {
        List<Observer> obs = map.get(e);
        if (obs != null) obs.forEach(o -> o.update(e, d));
    }
}
```

### 优缺点
- **优点**：松耦合；支持广播；易扩展
- **缺点**：通知顺序不可控；可能内存泄漏（未取消订阅）

---

## 3. 命令模式

将请求封装为对象，支持参数化、排队和撤销。

### 代码示例
```java
public interface Command { void execute(); void undo(); }
public class InsertCommand implements Command {
    private TextEditor editor; private String text; private int pos;
    public void execute() { editor.insert(text, pos); }
    public void undo() { editor.delete(pos, pos + text.length()); }
}
public class CommandHistory {
    private Stack<Command> history = new Stack<>();
    public void execute(Command c) { c.execute(); history.push(c); }
    public void undo() { if (!history.isEmpty()) history.pop().undo(); }
}
```
适用于任务队列、GUI 按钮、事务处理。

---

## 4. 状态模式

对象内部状态改变时改变其行为。

### 代码示例
```java
public interface OrderState { void pay(Order o); void ship(Order o); }
public class PendingState implements OrderState {
    public void pay(Order o) { o.setState(new PaidState()); }
    public void ship(Order o) { System.out.println("未支付无法发货"); }
}
public class Order {
    private OrderState state = new PendingState();
    public void pay() { state.pay(this); }
    public void setState(OrderState s) { this.state = s; }
}
```

### 与策略模式区别
策略由客户端主动选算法；状态由内部自动驱动行为。

---

## 5. 模板方法模式

定义算法骨架，部分步骤延迟到子类。

### 代码示例
```java
public abstract class DataMiner {
    public final void mine(String path) {
        openFile(path); extractData(); parseData(); report();
    }
    protected abstract void openFile(String path);
    protected abstract void extractData();
    protected abstract void parseData();
    protected void report() { System.out.println("生成报告"); } // 钩子
}
```
适用于构建流程、测试框架（JUnit）、数据解析。

---

## 6. 迭代器模式

提供顺序访问聚合元素的方法，不暴露内部结构。

### 代码示例
```java
public class NameRepository {
    private String[] names = {"Alice", "Bob", "Charlie"};
    public Iterator<String> iterator() {
        return new Iterator<>() {
            int i = 0;
            public boolean hasNext() { return i < names.length; }
            public String next() { return names[i++]; }
        };
    }
}
```

---

## 7. 中介者模式

用中介封装对象交互，降低耦合。适用于聊天室、交通管制、UI 联动。

### 代码示例
```java
public class ChatRoom {
    public static void showMsg(User from, String msg) {
        System.out.println(from.getName() + ": " + msg);
    }
}
public class User {
    private String name;
    public void send(String msg) { ChatRoom.showMsg(this, msg); }
}
```

---

## 8. 备忘录模式

捕获并保存对象状态，以便恢复。适用于撤销/重做、游戏存档。

### 代码示例
```java
public class Memento {
    private final String content;
    public Memento(String c) { this.content = c; }
    public String getContent() { return content; }
}
public class Editor {
    private String content = "";
    public Memento save() { return new Memento(content); }
    public void restore(Memento m) { content = m.getContent(); }
}
```

---

## 9. 责任链模式

请求沿处理者链传递直到被处理。

### 代码示例
```java
public abstract class Approver {
    protected Approver next;
    public Approver link(Approver n) { next = n; return n; }
    public void approve(double amount) {
        if (canApprove(amount)) System.out.println("已审批 " + amount);
        else if (next != null) next.approve(amount);
    }
    protected abstract boolean canApprove(double amount);
}
```
适用于审批流程、Servlet 过滤器、异常处理链。

---

## 10. 访问者模式

在不修改元素类的前提下定义新操作。

### 代码示例
```java
public interface Shape { void accept(ShapeVisitor v); }
public class Circle implements Shape {
    public double radius;
    public void accept(ShapeVisitor v) { v.visit(this); }
}
public interface ShapeVisitor { void visit(Circle c); void visit(Rectangle r); }
```
适用于 AST 遍历、多格式导出。

---

## 11. 解释器模式

定义文法表示和解释器。

### 代码示例
```java
public interface Expr { int interpret(Map<String, Integer> ctx); }
public class Add implements Expr {
    private Expr l, r;
    public int interpret(Map<String, Integer> c) { return l.interpret(c) + r.interpret(c); }
}
// new Add(new Var("x"), new Var("y")).interpret(Map.of("x",3,"y",5)) = 8
```
适用于正则表达式、DSL、表达式求值。

---

## 12. 行为型模式对比

| 模式 | 核心意图 | 典型场景 |
|------|---------|---------|
| 策略 | 算法可互换 | 支付方式、排序 |
| 观察者 | 一对多通知 | 事件系统 |
| 命令 | 封装请求 | 任务队列、撤销 |
| 状态 | 状态驱动行为 | 状态机 |
| 模板方法 | 算法骨架不变 | 构建流程 |
| 迭代器 | 统一遍历 | 集合遍历 |
| 中介者 | 封装交互 | UI协调 |
| 备忘录 | 保存/恢复状态 | 撤销重做 |
| 责任链 | 沿链传递 | 审批流程 |
| 访问者 | 新操作不改元素 | AST遍历 |
| 解释器 | 文法解释 | DSL |

**辨析**：策略由客户端选算法，状态由内部驱动；命令封装含接收者，策略只封装算法；观察者一对多，中介者多对多。

---

[返回上级目录](../README.md)
