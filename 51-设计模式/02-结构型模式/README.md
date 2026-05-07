# 结构型模式 (Structural Patterns)

> 关注类和对象的组合方式，形成更大的结构。

---

## 1. 适配器模式

### 定义与场景
将不兼容的接口转换为客户期望的接口。适用于旧系统迁移、第三方库适配。

### 类适配器（继承）
```java
public class ClassAdapter extends Adaptee implements Target {
    public void request() { specificRequest(); }
}
```

### 对象适配器（组合，推荐）
```java
public class ObjectAdapter implements Target {
    private Adaptee adaptee;
    public ObjectAdapter(Adaptee a) { this.adaptee = a; }
    public void request() { adaptee.specificRequest(); }
}
```

### C++ 实现
```cpp
class Adapter : public Target {
    Adaptee* adaptee;
public:
    Adapter(Adaptee* a) : adaptee(a) {}
    void request() override { adaptee->specificRequest(); }
};
```
对象适配器更灵活，可适配被适配者及其所有子类。

---

## 2. 桥接模式

将抽象与实现分离，使两者可独立变化。

### 代码示例
```java
public interface Color { String fill(); }
public abstract class Shape {
    protected Color color;
    public Shape(Color c) { this.color = c; }
    public abstract void draw();
}
public class Circle extends Shape {
    public void draw() { System.out.println(color.fill() + "圆形"); }
}
```
避免类爆炸（形状 x 颜色 = 2N 而非 N^2）。

---

## 3. 组合模式

将对象组合成树形结构，统一叶子和组合的接口。

### 代码示例
```java
public abstract class FileComponent {
    protected String name;
    public void display(int d) { System.out.println("  ".repeat(d) + name); }
}
public class Directory extends FileComponent {
    private List<FileComponent> children = new ArrayList<>();
    public void add(FileComponent c) { children.add(c); }
    public void display(int d) {
        System.out.println("  ".repeat(d) + "+ " + name);
        children.forEach(c -> c.display(d + 1));
    }
}
```
适用于文件系统、组织架构、UI 组件树。

---

## 4. 装饰器模式

动态地给对象添加职责，比继承更灵活。

### 代码示例
```java
public interface Coffee { double cost(); String desc(); }
public abstract class CoffeeDecorator implements Coffee {
    protected Coffee coffee;
    public CoffeeDecorator(Coffee c) { this.coffee = c; }
}
public class MilkDecorator extends CoffeeDecorator {
    public double cost() { return coffee.cost() + 2.0; }
    public String desc() { return coffee.desc() + " + 牛奶"; }
}
// new MilkDecorator(new SimpleCoffee()) -> "普通咖啡 + 牛奶" 7.0
```
适用于 Java I/O 流、日志增强、中间件叠加。

---

## 5. 外观模式

为子系统提供统一高层接口，简化调用。

### 代码示例
```java
public class ComputerFacade {
    private CPU cpu = new CPU();
    private Memory mem = new Memory();
    public void start() { cpu.start(); mem.load(); }
}
// 客户端只调用 facade.start()
```
**优点**：简化调用，松耦合。**缺点**：可能成为上帝类。

---

## 6. 享元模式

共享细粒度对象以减少内存占用，内部状态共享，外部状态由客户端传入。

### 代码示例
```java
public class ChessPieceFactory {
    private static Map<String, ChessPiece> pool = new HashMap<>();
    public static ChessPiece get(String color) {
        return pool.computeIfAbsent(color, ChessPiece::new);
    }
}
```
适用于大量相似对象（棋子、字符对象、连接池）。

---

## 7. 代理模式

为对象提供代理以控制访问。

### 静态代理
```java
public class ProxyImage implements Image {
    private RealImage real;
    public void display() {
        if (real == null) real = new RealImage(file); // 延迟加载
        real.display();
    }
}
```

### JDK 动态代理
```java
Image proxy = (Image) Proxy.newProxyInstance(
    target.getClass().getClassLoader(),
    target.getClass().getInterfaces(),
    (p, m, a) -> { System.out.println("[日志]"); return m.invoke(target, a); }
);
```

### CGLIB 动态代理
基于继承，可代理无接口类。Spring 默认：有接口用 JDK，无接口用 CGLIB。

### 代理类型
远程代理、虚代理（延迟加载）、保护代理（权限）、日志代理、缓存代理。

---

## 8. 结构型模式对比

| 模式 | 核心意图 | 典型场景 |
|------|---------|---------|
| 适配器 | 接口转换 | 旧系统迁移、第三方集成 |
| 桥接 | 抽象与实现分离 | 多维度独立变化 |
| 组合 | 树形结构统一 | 文件系统、组件树 |
| 装饰器 | 动态增强功能 | I/O 流、中间件 |
| 外观 | 简化调用 | 框架 Facade |
| 享元 | 共享对象 | 大量相似对象 |
| 代理 | 控制访问 | AOP、权限、缓存 |

**辨析**：装饰器增强功能，代理控制访问；适配器解决已有兼容，桥接设计时分离。

---

[返回上级目录](../README.md)
