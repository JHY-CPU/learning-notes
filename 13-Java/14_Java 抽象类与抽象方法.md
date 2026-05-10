# Java 抽象类与抽象方法


## 🔷 Java 抽象类与抽象方法


abstract class、abstract method、抽象类 vs 具体类、模板方法模式、abstract 与 final/static/private 组合规则。


## 抽象类与抽象方法


```
// ========== 抽象类 (Abstract Class) ==========
// 用 abstract 修饰的类
// 不能直接实例化 (不能 new)
// 可以有构造器、字段、具体方法、抽象方法

// ========== 抽象方法 ==========
// 只有声明, 没有方法体
// 必须由子类实现
// 子类必须实现所有抽象方法 (除非子类也是抽象类)

abstract class Shape {
    protected String color;

    // 抽象类可以有构造器
    public Shape(String color) {
        this.color = color;
        System.out.println("Shape constructor");
    }

    // 具体方法
    public String getColor() {
        return color;
    }

    // ========== 抽象方法 (没有方法体) ==========
    public abstract double getArea();
    public abstract double getPerimeter();

    // 具体方法可以使用抽象方法
    public void printInfo() {
        System.out.println("Color: " + color);
        System.out.println("Area: " + getArea());       // 多态调用
        System.out.println("Perimeter: " + getPerimeter());
    }
}

// ========== 具体子类 ==========
class Circle extends Shape {
    private double radius;

    public Circle(String color, double radius) {
        super(color);  // 调用抽象类构造器
        this.radius = radius;
    }

    @Override
    public double getArea() {
        return Math.PI * radius * radius;
    }

    @Override
    public double getPerimeter() {
        return 2 * Math.PI * radius;
    }
}

class Rectangle extends Shape {
    private double width, height;

    public Rectangle(String color, double width, double height) {
        super(color);
        this.width = width;
        this.height = height;
    }

    @Override
    public double getArea() {
        return width * height;
    }

    @Override
    public double getPerimeter() {
        return 2 * (width + height);
    }
}

// ========== 使用 ==========
public class AbstractDemo {
    public static void main(String[] args) {
        // Shape s = new Shape("red");  // ❌ 不能实例化抽象类!

        Shape s1 = new Circle("red", 5);
        Shape s2 = new Rectangle("blue", 3, 4);

        System.out.println(s1.getArea());  // 78.54
        System.out.println(s2.getArea());  // 12

        s1.printInfo();
        // Color: red
        // Area: 78.54
        // Perimeter: 31.42

        // 多态数组
        Shape[] shapes = {
            new Circle("green", 2),
            new Rectangle("yellow", 3, 6),
            new Circle("purple", 1.5)
        };
        for (Shape s : shapes) {
            s.printInfo();
            System.out.println("---");
        }
    }
}
```


## 模板方法模式


```
// ========== 模板方法模式 (Template Method) ==========
// 抽象类定义算法骨架, 子类实现具体步骤
// 经典应用: JdbcTemplate, HttpServlet

abstract class DataProcessor {
    // ========== 模板方法 (final) ==========
    // 定义处理流程, 子类不能修改
    public final void process() {
        loadData();
        if (isValid()) {           // 钩子方法
            processData();
        }
        saveResult();
        cleanup();
    }

    // ========== 抽象方法 (子类必须实现) ==========
    protected abstract void loadData();
    protected abstract void processData();

    // ========== 具体方法 (可选重写) ==========
    protected void saveResult() {
        System.out.println("Saving to database...");
    }

    protected void cleanup() {
        System.out.println("Cleanup resources...");
    }

    // ========== 钩子方法 (Hook) ==========
    // 默认行为, 子类可重写控制流程
    protected boolean isValid() {
        return true;  // 默认验证通过
    }
}

class CSVProcessor extends DataProcessor {
    @Override
    protected void loadData() {
        System.out.println("Loading CSV file...");
    }

    @Override
    protected void processData() {
        System.out.println("Processing CSV rows...");
    }

    @Override
    protected boolean isValid() {
        System.out.println("Validating CSV...");
        return true;
    }
}

class JSONProcessor extends DataProcessor {
    @Override
    protected void loadData() {
        System.out.println("Loading JSON file...");
    }

    @Override
    protected void processData() {
        System.out.println("Parsing JSON objects...");
    }

    // 不重写 saveResult — 使用默认
    // 不重写 isValid — 默认 true
}

// ========== 使用 ==========
// DataProcessor processor = new CSVProcessor();
// processor.process();
// 输出:
// Loading CSV file...
// Validating CSV...
// Processing CSV rows...
// Saving to database...
// Cleanup resources...
```


## 抽象类使用规则


```
// ========== 抽象类规则总结 ==========

public abstract class AbstractRules {

    // ========== 可以有的成员 ==========
    private int x;                   // 字段 (任意类型)
    public static final int MAX = 100;  // 常量
    public AbstractRules() { }       // 构造器
    public void concrete() { }       // 具体方法
    public abstract void absMethod(); // 抽象方法
    public static void staticMethod() { } // 静态方法
    public final void finalMethod() { }   // final 方法

    // ========== 规则 ==========
    // 1. 抽象类用 abstract class 声明
    // 2. 抽象方法没有方法体, 子类必须实现
    // 3. 抽象类可以有 0~n 个抽象方法
    // 4. 抽象类不能 new (不能实例化)
    // 5. 抽象类可以有构造器 (子类 super() 调用)
    // 6. 子类要么实现所有抽象方法, 要么也是抽象类
    // 7. 抽象方法不能是 private (子类看不到)
    // 8. 抽象方法不能是 final (不能被重写)
    // 9. 抽象方法不能是 static (静态属于类)

    // ❌ 不允许:
    // private abstract void m1();    // ❌
    // final abstract void m2();      // ❌
    // static abstract void m3();     // ❌
    // abstract void m4() { }          // ❌ 有方法体
}

// ========== 抽象类 vs 具体类 ==========
// ┌─────────────┬─────────────┬─────────────┐
// │             │ 抽象类      │ 具体类      │
// ├─────────────┼─────────────┼─────────────┤
// │ 实例化      │ ❌          │ ✅          │
// │ 抽象方法    │ ✅          │ ❌          │
// │ 构造器      │ ✅          │ ✅          │
// │ 字段        │ ✅          │ ✅          │
// │ 具体方法    │ ✅          │ ✅          │
// └─────────────┴─────────────┴─────────────┘

// ========== 抽象类 vs 接口 ==========
// ┌─────────────┬─────────────┬─────────────┐
// │             │ 抽象类      │ 接口        │
// ├─────────────┼─────────────┼─────────────┤
// │ 多继承      │ 单继承      │ 多实现      │
// │ 构造器      │ ✅          │ ❌          │
// │ 字段        │ 任意        │ public static final │
// │ 抽象方法    │ abstract    │ default/static/private 也可 │
// │ 语义        │ is-a        │ can-do      │
// └─────────────┴─────────────┴─────────────┘

// 选择指南:
// 有is-a关系 + 共享状态/构造器 → 抽象类
// 有can-do能力 + 多继承需要 → 接口
```


> **Note:** 💡 抽象类要点: abstract class 不能 new; 抽象方法无方法体; 子类必须实现所有抽象方法; 可以有构造器/字段/具体方法/静态方法; 模板方法模式用 final 模板方法定义骨架; 抽象方法不能 private/final/static; 有共享状态选抽象类, 多能力选接口。


## 练习


<!-- Converted from: 14_Java 抽象类与抽象方法.html -->
