# Java 接口


## 📐 Java 接口


interface 声明、implements 实现、接口多继承、default/static/private 方法 (Java 8/9+)、函数式接口 @FunctionalInterface、接口设计原则。


## 接口基础


```
// ========== 接口 (Interface) ==========
// 完全抽象的"契约" — 定义行为规范
// 所有方法默认 public abstract
// 所有字段默认 public static final

// ========== 定义接口 ==========
interface Flyable {
    // 常量 (public static final)
    int MAX_SPEED = 1000;  // 默认 public static final

    // 抽象方法 (public abstract)
    void fly();
    void land();
}

interface Swimmable {
    void swim();
    void dive(int depth);
}

// ========== 实现接口 ==========
class Bird implements Flyable {
    private String name;

    Bird(String name) { this.name = name; }

    @Override
    public void fly() {
        System.out.println(name + " is flying");
    }

    @Override
    public void land() {
        System.out.println(name + " is landing");
    }
}

// ========== 多实现 ==========
class Duck implements Flyable, Swimmable {
    @Override
    public void fly() {
        System.out.println("Duck flies short distance");
    }

    @Override
    public void land() {
        System.out.println("Duck lands on water");
    }

    @Override
    public void swim() {
        System.out.println("Duck is swimming");
    }

    @Override
    public void dive(int depth) {
        System.out.println("Duck dives " + depth + " meters");
    }
}

// ========== 接口多态 ==========
public class InterfaceDemo {
    public static void main(String[] args) {
        Flyable f1 = new Bird("Eagle");
        Flyable f2 = new Duck();

        f1.fly();  // "Eagle is flying"
        f2.fly();  // "Duck flies short distance"

        // 接口引用调用接口方法
        Flyable[] fliers = {f1, f2};
        for (Flyable f : fliers) {
            f.fly();
        }

        // 需要转型才能调用特有接口方法
        if (f2 instanceof Swimmable s) {
            s.swim();  // ✅ 模式匹配自动转型
        }

        // 接口常量
        System.out.println(Flyable.MAX_SPEED);  // 1000
    }
}
```


## 接口进阶 (Java 8+)


```
// ========== Java 8: default 方法 ==========
// 接口可以有默认实现
// 子类可以选择重写或不重写

interface Vehicle {
    void start();

    // default 方法: 带默认实现
    default void honk() {
        System.out.println("Beep beep!");
    }

    // 另一个 default 方法
    default void warning() {
        System.out.println("Vehicle warning!");
    }
}

class Car implements Vehicle {
    @Override
    public void start() {
        System.out.println("Car engine started");
    }
    // honk() 使用默认实现
}

class Bus implements Vehicle {
    @Override
    public void start() {
        System.out.println("Bus engine started");
    }

    @Override
    public void honk() {
        System.out.println("BUUUP BUUUP!");  // 重写默认
    }
}

// ========== default 方法冲突解决 ==========
// 当一个类实现多个接口, 且接口有相同 default 方法时:
interface A {
    default void hello() { System.out.println("A says hello"); }
}

interface B {
    default void hello() { System.out.println("B says hello"); }
}

// 必须重写冲突方法!
class MyClass implements A, B {
    @Override
    public void hello() {
        // 可以选择调用其中一个父接口的默认实现
        A.super.hello();  // 调用 A 的 default 方法
        // B.super.hello();  // 或调用 B 的
        System.out.println("MyClass says hello");
    }
}

// ========== Java 8: static 方法 ==========
interface MathUtils {
    static double sqrt(double x) {
        return Math.sqrt(x);
    }

    static int max(int a, int b) {
        return a > b ? a : b;
    }
}
// 调用: MathUtils.sqrt(25) → 5.0

// ========== Java 9: private 方法 ==========
interface PrivateInterface {
    // 用于共享 default 方法间的公共逻辑
    default void methodA() {
        commonLogic();
        System.out.println("A");
    }

    default void methodB() {
        commonLogic();
        System.out.println("B");
    }

    // private 方法: 接口内部共享, 不暴露给子类
    private void commonLogic() {
        System.out.print("Processing: ");
    }

    // private static 也可以
    private static void helper() {
        System.out.println("Helper");
    }
}
```


## 函数式接口与 Lambda


```
// ========== 函数式接口 @FunctionalInterface ==========
// 只有一个抽象方法的接口
// 可以用 Lambda 表达式实现

// ========== 常见函数式接口 ==========
// Runnable       — void run()                    — 无参无返
// Callable    — T call()                      — 无参有返
// Consumer    — void accept(T)                — 消费
// Supplier    — T get()                        — 供给
// Function  — R apply(T)                    — 转换
// Predicate   — boolean test(T)               — 判断
// Comparator  — int compare(T, T)             — 比较

@FunctionalInterface
interface Greeter {
    String greet(String name);
    // 只能有一个抽象方法!
    // void other();  // ❌ 编译错误
}

// ========== 使用 Lambda ==========
// 传统方式 (匿名内部类)
Greeter traditional = new Greeter() {
    @Override
    public String greet(String name) {
        return "Hello, " + name;
    }
};

// Lambda 方式
Greeter lambda = name -> "Hello, " + name;

// ========== 实际应用 ==========
import java.util.*;
import java.util.function.*;

public class FunctionalDemo {
    public static void main(String[] args) {
        // Predicate: 判断
        Predicate isEmpty = s -> s.isEmpty();
        System.out.println(isEmpty.test(""));  // true

        // Consumer: 消费
        Consumer print = s -> System.out.println(s);
        print.accept("Hello");  // "Hello"

        // Function: 转换
        Function length = s -> s.length();
        System.out.println(length.apply("Java"));  // 4

        // Supplier: 供给
        Supplier random = () -> Math.random();
        System.out.println(random.get());

        // 综合: 方法引用
        List names = Arrays.asList("Alice", "Bob", "Charlie");
        names.forEach(System.out::println);  // 方法引用
    }
}

// ========== @FunctionalInterface 设计指南 ==========
// 1. 只定义一个抽象方法
// 2. 可以用 default/static 方法补充
// 3. 标注 @FunctionalInterface 让编译器检查
// 4. 常用: Comparator, Runnable, Callable
// 5. Lambda 可替代匿名内部类, 更简洁

// 自定义函数式接口:
@FunctionalInterface
interface Transformer {
    R transform(T input);
    // 一个抽象方法: transform
}
```


## 接口设计原则


```
// ========== 接口设计原则 ==========

// 1. 接口隔离原则 (ISP)
//    ❌ 胖接口: 方法太多, 实现类被迫实现不需要的方法
//    ✅ 小接口: 职责单一, 按需实现

// ❌ 胖接口
interface Worker {
    void work();
    void eat();
    void sleep();
    void attendMeeting();
}

// ✅ 接口隔离
interface Workable { void work(); }
interface Eatable { void eat(); }
interface MeetingAttendable { void attendMeeting(); }

// 机器人只需要 Workable
class Robot implements Workable {
    public void work() { System.out.println("Robot working"); }
}

// 人类可以实现多个
class Human implements Workable, Eatable, MeetingAttendable {
    public void work() { System.out.println("Working"); }
    public void eat() { System.out.println("Eating"); }
    public void attendMeeting() { System.out.println("In meeting"); }
}

// ========== 2. 标记接口 (Marker Interface) ==========
// 没有方法的接口, 用于标记能力
// 例如: Serializable, Cloneable, RandomAccess

// 自定义标记接口
interface Secure {}  // 标记为安全

class SecureData implements Secure {
    String data;
}

// 使用标记判断
class SecurityChecker {
    void process(Object obj) {
        if (obj instanceof Secure) {
            System.out.println("Processing secure data");
        } else {
            System.out.println("Warning: not secure!");
        }
    }
}

// ========== 3. 接口 vs 抽象类选择 ==========
// 接口: "can-do" 能力契约
// 抽象类: "is-a" 层次关系

// 鸭子能飞能游 → 接口
// 狗是动物 → 继承

// ========== 常见 JDK 接口 ==========
// Comparable   — compareTo() 自然排序
// Iterable     — iterator() 可迭代
// Collection   — 集合根接口
// List, Set, Map — 集合子接口
// AutoCloseable   — close() 资源自动关闭
// Serializable    — 可序列化 (标记接口)
```


> **Note:** 💡 接口要点: interface 完全抽象契约; implements 多实现; 字段默认 public static final; 方法默认 public abstract; Java 8+ default/static 方法; Java 9+ private 方法; @FunctionalInterface 单抽象方法 → Lambda; default 方法冲突必须重写; 接口隔离原则 (ISP) 小接口优于胖接口; 标记接口无方法。


## 练习


<!-- Converted from: 15_Java 接口.html -->
