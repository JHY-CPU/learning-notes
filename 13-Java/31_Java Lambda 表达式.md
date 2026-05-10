# Java Lambda 表达式


## λ Java Lambda 表达式


Lambda 语法、函数式接口、方法引用 (::)、构造器引用、变量捕获 (effectively final)、Lambda 与匿名类对比、常见 Lambda 模式。


## Lambda 语法


```
// ========== Lambda 表达式 (Java 8+) ==========
// 简洁地实现函数式接口 (只有一个抽象方法的接口)
// 语法: (参数) -> { 方法体 }

import java.util.*;
import java.util.function.*;

public class LambdaSyntax {
    public static void main(String[] args) {
        // ========== 基本语法 ==========
        // (参数列表) -> { 方法体 }
        // (参数列表) -> 表达式          (单行)
        // () -> { 方法体 }              (无参)
        // x -> 表达式                   (单参数可省略括号)

        // ---------- 无参 ----------
        Runnable r1 = () -> System.out.println("Hello");

        // ---------- 单参 (省略括号) ----------
        Consumer c1 = s -> System.out.println(s);
        Consumer c2 = (s) -> { System.out.println(s); };

        // ---------- 多参 ----------
        Comparator cmp = (a, b) -> a - b;
        BinaryOperator sum = (a, b) -> a + b;

        // ---------- 多行 (需要 { }) ----------
        Comparator byLength = (a, b) -> {
            int lenCmp = Integer.compare(a.length(), b.length());
            if (lenCmp != 0) return lenCmp;
            return a.compareTo(b);
        };

        // ========== 类型推断 ==========
        // 编译器从上下文推断参数类型
        Comparator comp = (a, b) -> a.compareTo(b);
        // 等价于:
        Comparator comp2 = (String a, String b) -> a.compareTo(b);

        // ========== 简化规则 ==========
        // 1. 参数类型可省略 (编译器推断)
        // 2. 单参数可省略括号: x -> x * 2
        // 3. 单行方法体可省略 {} 和 return:
        //    (a, b) -> a + b  等价于  (a, b) -> { return a + b; }
        // 4. 无参必须有括号: () -> 42
    }
}

// ========== Lambda 使用场景 ==========
class LambdaUseCases {
    public static void main(String[] args) {
        List list = new ArrayList<>(List.of("Charlie", "Alice", "Bob"));

        // 1. 排序
        list.sort((a, b) -> a.compareTo(b));
        list.sort(Comparator.naturalOrder());

        // 2. 遍历
        list.forEach(s -> System.out.println(s));

        // 3. 过滤 (Stream)
        list.stream()
            .filter(s -> s.startsWith("A"))
            .forEach(System.out::println);

        // 4. 映射
        list.stream()
            .map(s -> s.length())
            .forEach(System.out::println);

        // 5. 事件处理 (Swing/JavaFX)
        // button.addActionListener(e -> System.out.println("Clicked"));

        // 6. 线程
        Thread thread = new Thread(() -> {
            System.out.println("Running in thread");
        });

        // 7. 自定义函数式接口
        Calculator add = (x, y) -> x + y;
        Calculator multiply = (x, y) -> x * y;
        System.out.println(add.calc(3, 4));   // 7
        System.out.println(multiply.calc(3, 4)); // 12
    }

    @FunctionalInterface
    interface Calculator {
        int calc(int a, int b);
    }
}
```


## 方法引用


```
// ========== 方法引用 (::) ==========
// Lambda 的简化写法: 直接引用已有方法
// 4 种类型:

import java.util.*;
import java.util.function.*;

public class MethodRefDemo {

    // ========== 1. 静态方法引用 ==========
    // Class::staticMethod
    static void staticMethodRef() {
        // Lambda:  s -> Integer.parseInt(s)
        // 引用:    Integer::parseInt

        Function parser1 = s -> Integer.parseInt(s);
        Function parser2 = Integer::parseInt;

        // 另一个例子:
        // Comparator cmp = (a, b) -> Integer.compare(a, b);
        Comparator cmp = Integer::compare;
    }

    // ========== 2. 实例方法引用 (特定对象) ==========
    // instance::method
    static void instanceMethodRef() {
        String str = "hello";

        // Lambda:  () -> str.length()
        // 引用:    str::length

        Supplier len1 = () -> str.length();
        Supplier len2 = str::length;

        // 实际例子:
        List list = List.of("a", "b", "c");
        list.forEach(System.out::println);  // System.out::println
    }

    // ========== 3. 实例方法引用 (类型, 任意对象) ==========
    // Class::instanceMethod
    static void arbitraryObjectMethodRef() {
        // Lambda:  (s) -> s.length()
        // 引用:    String::length

        Function len1 = s -> s.length();
        Function len2 = String::length;

        // Lambda:  (a, b) -> a.compareTo(b)
        // 引用:    String::compareTo

        Comparator comp = String::compareTo;

        // Lambda:  (s) -> s.toUpperCase()
        // 引用:    String::toUpperCase

        // 原理: 第一个参数作为调用对象, 其余作为方法参数
    }

    // ========== 4. 构造器引用 ==========
    // Class::new
    static void constructorRef() {
        // Lambda:  () -> new ArrayList<>()
        // 引用:    ArrayList::new

        Supplier> list1 = () -> new ArrayList<>();
        Supplier> list2 = ArrayList::new;

        // 数组构造
        // Lambda:  len -> new int[len]
        // 引用:    int[]::new

        Function arrMaker = int[]::new;
        int[] arr = arrMaker.apply(5);  // new int[5]

        // 带参数的构造器
        // Lambda:  name -> new Person(name)
        // 引用:    Person::new
        // Function personCreator = Person::new;
    }

    // ========== 方法引用速查 ==========
    // ┌──────────────┬────────────────┬───────────────────────┐
    // │ 类型         │ Lambda         │ 方法引用              │
    // ├──────────────┼────────────────┼───────────────────────┤
    // │ 静态方法     │ x -> foo(x)    │ Foo::staticMethod     │
    // │ 特定对象     │ () -> inst.foo │ inst::method          │
    // │ 任意对象     │ x -> x.foo()   │ Class::method         │
    // │ 构造器       │ () -> new Foo()│ Foo::new              │
    // └──────────────┴────────────────┴───────────────────────┘
}
```


## 变量捕获与 effectively final


```
// ========== 变量捕获 ==========
// Lambda 可以访问外部变量
// 被捕获的变量必须是 effectively final (不可修改)

public class VariableCapture {

    public static void main(String[] args) {
        // ========== effectively final ==========
        int factor = 10;  // 没有修改 → effectively final

        Function multiplier = n -> n * factor;
        System.out.println(multiplier.apply(5));  // 50

        // factor = 20;  // ❌ 被 Lambda 捕获后不能修改!

        // ========== 实例变量和静态变量 ==========
        // 实例变量和静态变量没有此限制 (存在堆中)
        VariableCapture vc = new VariableCapture();
        vc.demoInstanceVar();
    }

    private int instanceVar = 100;

    void demoInstanceVar() {
        // 实例变量可以被 Lambda 修改!
        Function func = n -> {
            instanceVar += n;  // ✅ 可修改实例变量
            return instanceVar;
        };
        System.out.println(func.apply(50));  // 150
        instanceVar = 999;  // ✅ 也可以直接修改
        System.out.println(func.apply(1));   // 1000
    }

    // ========== this 关键字 ==========
    // Lambda 中的 this 指向外部类 (而非 Lambda 本身)
    // 匿名类中的 this 指向匿名类实例

    void thisDemo() {
        Runnable r1 = () -> {
            System.out.println(this);  // VariableCapture 实例
        };

        Runnable r2 = new Runnable() {
            @Override
            public void run() {
                System.out.println(this);  // 匿名 Runnable 实例
            }
        };
    }
}

// ========== Lambda vs 匿名内部类 ==========
// ┌──────────────┬────────────────────┬──────────────────┐
// │ 特性         │ Lambda             │ 匿名内部类       │
// ├──────────────┼────────────────────┼──────────────────┤
// │ this         │ 外部类             │ 匿名类自身       │
// │ 编译         │ invokeDynamic      │ 生成 .class 文件 │
// │ 接口限制     │ 函数式接口(1个方法)│ 任何接口/类     │
// │ 简洁性       │ 极简               │ 较啰嗦           │
// │ 变量捕获     │ effectively final  │ effectively final│
// └──────────────┴────────────────────┴──────────────────┘

// Lambda 编译: invokedynamic + LambdaMetafactory
// 不产生匿名 .class 文件, 运行时动态生成

// ========== Lambda 的限制 ==========
// 1. 只能用于函数式接口 (1个抽象方法)
// 2. this 指向外部类 (不是 Lambda)
// 3. 不能序列化 (除非转型)
// 4. 调试时堆栈信息不如匿名类清晰
```


> **Note:** 💡 Lambda 要点: (参数) -> { 表达式 }; 单参可省括号, 单行可省 {} 和 return; 方法引用 :: 简化 Lambda (静态/实例/构造器); 变量必须 effectively final; this 指向外部类; 只能用于函数式接口; 编译用 invokedynamic 更高效; 与 Stream/集合结合使用。


## 练习


<!-- Converted from: 31_Java Lambda 表达式.html -->
