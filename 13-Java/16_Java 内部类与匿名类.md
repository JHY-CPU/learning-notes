# Java 内部类与匿名类


## 🏠 Java 内部类与匿名类


4 种内部类 (成员/静态/局部/匿名)、内部类特性 (访问外部类 private 成员)、匿名类、Lambda 替代匿名类、嵌套类设计。


## 成员内部类


```
// ========== 成员内部类 (Member Inner Class) ==========
// 定义在外部类内部, 作为成员
// 可以访问外部类的所有成员 (包括 private)
// 需要外部类实例才能创建

class Outer {
    private String message = "Hello from Outer";
    private static int counter = 0;

    // ========== 成员内部类 ==========
    class Inner {
        private int id;

        Inner() {
            this.id = ++counter;
        }

        void display() {
            // 可以访问外部类的 private 字段
            System.out.println(message);  // "Hello from Outer"
            System.out.println("Inner id: " + id);
        }

        // 获取外部类引用: Outer.this
        Outer getOuter() {
            return Outer.this;
        }
    }

    // 外部类方法创建内部类
    void testInner() {
        Inner inner = new Inner();
        inner.display();
    }

    // 内部类的访问需要外部类实例
    Inner createInner() {
        return new Inner();
    }
}

// ========== 使用成员内部类 ==========
public class InnerClassDemo {
    public static void main(String[] args) {
        // 方式1: 通过外部类实例
        Outer outer = new Outer();
        Outer.Inner inner1 = outer.new Inner();
        inner1.display();

        // 方式2: 通过外部类方法
        Outer.Inner inner2 = outer.createInner();
        inner2.display();

        // 成员内部类可以继承或被继承
        // 成员内部类不能有 static 成员 (除非是常量)
    }
}
```


## 静态内部类


```
// ========== 静态内部类 (Static Nested Class) ==========
// 用 static 修饰的内部类
// 不依赖外部类实例
// 只能访问外部类的静态成员

class MathHelper {
    private static final double PI = 3.14159;
    private String name = "Math";

    // ========== 静态内部类 ==========
    static class Calculator {
        int add(int a, int b) { return a + b; }
        int subtract(int a, int b) { return a - b; }

        double circleArea(double r) {
            // 只能访问外部类的静态成员
            return PI * r * r;
            // System.out.println(name);  // ❌ 不能访问实例成员!
        }
    }

    // ========== 非静态内部类 (示例对比) ==========
    class InstanceInner {
        void show() {
            System.out.println(name);  // ✅ 可访问实例成员
        }
    }
}

// ========== 使用静态内部类 ==========
// 不需要外部类实例!
public class StaticInnerDemo {
    public static void main(String[] args) {
        MathHelper.Calculator calc = new MathHelper.Calculator();
        System.out.println(calc.add(5, 3));       // 8
        System.out.println(calc.circleArea(2));    // 12.56636

        // 常用于构建器模式
        User user = new User.Builder("Alice")
            .age(25)
            .email("alice@example.com")
            .build();
    }
}

// ========== 经典应用: Builder 模式 ==========
class User {
    private String username;
    private int age;
    private String email;

    // private 构造器, 只有 Builder 能调用
    private User(Builder builder) {
        this.username = builder.username;
        this.age = builder.age;
        this.email = builder.email;
    }

    public String getUsername() { return username; }
    public int getAge() { return age; }
    public String getEmail() { return email; }

    // ========== 静态内部类 Builder ==========
    public static class Builder {
        private String username;   // 必需
        private int age = 0;       // 可选 (默认)
        private String email;      // 可选

        public Builder(String username) {
            this.username = username;  // 必需参数在构造器
        }

        public Builder age(int age) {
            this.age = age;
            return this;  // 链式调用
        }

        public Builder email(String email) {
            this.email = email;
            return this;
        }

        public User build() {
            return new User(this);
        }
    }
}
```


## 局部内部类与匿名类


```
// ========== 局部内部类 (Local Inner Class) ==========
// 定义在方法/代码块中
// 作用域仅限于所在方法
// 可以访问外部类的成员和所在方法的 final/effectively final 变量

class LocalInnerDemo {
    private String outerField = "outer";

    void printNumbers(int start, int end) {
        // 局部变量: effectively final (Java 8+)
        String prefix = "Count: ";  // 不能修改!

        // ========== 局部内部类 ==========
        class Counter {
            void display() {
                // 可以访问外部类成员
                System.out.println(outerField);
                // 可以访问所在方法的局部变量 (effectively final)
                System.out.println(prefix);
                // System.out.println(start);  // ✅
            }
        }

        Counter c = new Counter();
        c.display();
        // Counter 不能在此方法外使用
    }
}

// ========== 匿名内部类 (Anonymous Inner Class) ==========
// 没有名字的类, 同时定义和实例化
// 用于简化只需用一次的类

abstract class Greeting {
    abstract String greet(String name);
}

interface ClickHandler {
    void onClick();
}

public class AnonymousDemo {
    public static void main(String[] args) {
        // ========== 继承抽象类的匿名类 ==========
        Greeting greeting = new Greeting() {
            @Override
            String greet(String name) {
                return "Hello, " + name + "!";
            }
        };
        System.out.println(greeting.greet("Alice"));  // "Hello, Alice!"

        // ========== 实现接口的匿名类 ==========
        ClickHandler handler = new ClickHandler() {
            @Override
            public void onClick() {
                System.out.println("Button clicked!");
            }
        };
        handler.onClick();

        // ========== 匿名类作为参数 ==========
        // 常见于事件监听、线程、比较器
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("Thread running");
            }
        });

        // ========== Lambda 替代匿名类 (函数式接口) ==========
        // Runnable 是函数式接口 → 可用 Lambda
        Thread thread2 = new Thread(() ->
            System.out.println("Lambda thread")
        );

        // Comparator 匿名类 → Lambda
        // Collections.sort(list, (a, b) -> a.compareTo(b));
    }

    // ========== 匿名类的注意事项 ==========
    // 1. 不能有显式构造器 (没有类名)
    // 2. 不能是 abstract (立即实例化)
    // 3. 可以定义额外字段和方法
    // 4. 访问外部变量必须是 effectively final
    // 5. Lambda 比匿名类更简洁, 优先使用
}
```


## 内部类对比与选择


```
// ========== 4 种内部类对比 ==========
//
// ┌─────────────┬──────────┬─────────────┬──────────────┐
// │ 类型        │ 依赖外部 │ 可访问成员  │ 应用场景     │
// ├─────────────┼──────────┼─────────────┼──────────────┤
// │ 成员内部类  │ 需要实例 │ 全部        │ 迭代器/事件  │
// │ 静态内部类  │ 不需要   │ 静态成员    │ Builder/分组 │
// │ 局部内部类  │ 方法内   │ 全部+局部   │ 复杂方法     │
// │ 匿名内部类  │ 实例化时 │ 全部+final  │ 回调/监听    │
// └─────────────┴──────────┴─────────────┴──────────────┘

// ========== 选择指南 ==========
// 1. 需要访问外部类实例成员 → 成员内部类
// 2. 不需要外部类实例 → 静态内部类 (性能更好)
// 3. 只在方法内使用 → 局部内部类
// 4. 一次性实现 → 匿名内部类 (或 Lambda)

// ========== 内部类编译 ==========
// 编译后生成独立的 .class 文件:
// Outer.class
// Outer$Inner.class           (成员内部类)
// Outer$StaticInner.class     (静态内部类)
// Outer$1Local.class          (局部/匿名内部类)

// ========== 内部类 & 序列化 ==========
// 内部类实现 Serializable 需谨慎!
// 成员内部类持有外部类引用 → 序列化时会包含外部类
// 静态内部类没有外部类引用 → 更安全
```


> **Note:** 💡 内部类要点: 成员内部类需要外部实例, 可访问所有成员; 静态内部类不需要外部实例, 只能访问静态成员; 局部内部类在方法内定义; 匿名内部类同时定义+实例化, 适合回调; Lambda 替代函数式接口的匿名类; 静态内部类常用于 Builder 模式; 匿名类变量必须 effectively final。


## 练习


<!-- Converted from: 16_Java 内部类与匿名类.html -->
