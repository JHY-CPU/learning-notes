# 创建型模式 (Creational Patterns)

> 关注对象的创建机制，将创建与使用分离。

---

## 1. 单例模式

### 定义与场景
确保一个类只有一个实例，提供全局访问点。适用于连接池、配置管理、日志对象。

### 饿汉式
```java
public class EagerSingleton {
    private static final EagerSingleton INSTANCE = new EagerSingleton();
    private EagerSingleton() {}
    public static EagerSingleton getInstance() { return INSTANCE; }
}
```
类加载时创建，线程安全，无法延迟加载。

### 懒汉式
```java
public static synchronized LazySingleton getInstance() {
    if (instance == null) instance = new LazySingleton();
    return instance;
}
```
延迟加载，但每次加锁，性能差。

### 双重检查锁定
```java
private static volatile DCLSingleton instance;
public static DCLSingleton getInstance() {
    if (instance == null) {
        synchronized (DCLSingleton.class) {
            if (instance == null) instance = new DCLSingleton();
        }
    }
    return instance;
}
```
`volatile` 防止指令重排序，兼顾安全与性能。

### 静态内部类（推荐）
```java
public static InnerClassSingleton getInstance() { return Holder.INSTANCE; }
private static class Holder { static final InnerClassSingleton INSTANCE = new InnerClassSingleton(); }
```

### 枚举实现（Effective Java 推荐）
```java
public enum EnumSingleton { INSTANCE; }
```
天然线程安全，防反射和反序列化攻击。

---

## 2. 简单工厂模式

根据参数创建产品实例。不属于 GoF 23 种模式。

### 代码示例
```java
public interface Button { void render(); }
public class ButtonFactory {
    public static Button create(String type) {
        switch (type) {
            case "windows": return new WindowsButton();
            case "mac": return new MacButton();
            default: throw new IllegalArgumentException("未知类型");
        }
    }
}
```
**缺点**：违反开闭原则，新增产品需修改工厂。

---

## 3. 工厂方法模式

定义创建接口，子类决定实例化哪个类。

### 代码示例
```java
public interface Logger { void log(String msg); }
public abstract class LoggerFactory {
    public abstract Logger createLogger();
    public void log(String msg) { createLogger().log(msg); }
}
public class FileLoggerFactory extends LoggerFactory {
    public Logger createLogger() { return new FileLogger(); }
}
```
**优点**：符合开闭原则。**缺点**：类数量增多。

---

## 4. 抽象工厂模式

创建一系列相关对象（产品族）的接口。

### 代码示例
```java
public interface GUIFactory {
    Button createButton();
    Checkbox createCheckbox();
}
public class WindowsFactory implements GUIFactory {
    public Button createButton() { return new WindowsButton(); }
    public Checkbox createCheckbox() { return new WindowsCheckbox(); }
}
```
与工厂方法区别：工厂方法创建单一产品，抽象工厂创建产品族。

---

## 5. 建造者模式

将复杂对象的构建与表示分离，支持链式调用。

### 代码示例
```java
public class ComputerBuilder {
    private Computer computer = new Computer();
    public ComputerBuilder cpu(String c)    { computer.setCpu(c); return this; }
    public ComputerBuilder ram(String r)    { computer.setRam(r); return this; }
    public ComputerBuilder gpu(String g)    { computer.setGpu(g); return this; }
    public Computer build() { return computer; }
}
// 使用: new ComputerBuilder().cpu("i9").ram("32GB").gpu("RTX4090").build()
```

---

## 6. 原型模式

通过克隆已有实例创建新对象，适用于创建成本高的对象。

### 代码示例
```java
public class Circle extends Shape implements Cloneable {
    private int radius;
    public Circle clone() throws CloneNotSupportedException {
        return (Circle) super.clone(); // 浅拷贝
    }
}
```
注意：默认浅拷贝，引用类型属性需手动深拷贝。

---

## 7. 创建型模式对比

| 模式 | 核心意图 | 典型场景 |
|------|---------|---------|
| 单例 | 全局唯一实例 | 配置管理、连接池 |
| 简单工厂 | 参数化创建 | 产品种类少且固定 |
| 工厂方法 | 子类决定创建 | 需扩展产品种类 |
| 抽象工厂 | 创建产品族 | 跨平台 UI 组件 |
| 建造者 | 分步复杂构建 | 多参数对象 |
| 原型 | 克隆已有对象 | 创建成本高的对象 |

**选型指南**：唯一实例选单例；简单创建选简单工厂；扩展性选工厂方法；产品族选抽象工厂；复杂构建选建造者；高成本对象选原型。

---

[返回上级目录](../README.md)
