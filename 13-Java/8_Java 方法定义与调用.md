# Java 方法定义与调用


## 📞 Java 方法定义与调用


方法声明 (修饰符/返回类型/方法名/参数)、方法重载 (overloading)、值传递 (pass-by-value)、可变参数、return 语句、命令行参数。


## 方法基础


```
// ========== 方法声明 ==========
// [修饰符] [返回类型] 方法名([参数列表]) [throws 异常] {
//     方法体
//     [return 值;]
// }

public class MethodBasics {

    // 无参无返回值
    void sayHello() {
        System.out.println("Hello!");
    }

    // 有参无返回值
    void greet(String name) {
        System.out.println("Hello, " + name + "!");
    }

    // 有参有返回值
    int add(int a, int b) {
        return a + b;
    }

    // static 方法 (类方法)
    static double circleArea(double radius) {
        return Math.PI * radius * radius;
    }

    public static void main(String[] args) {
        MethodBasics mb = new MethodBasics();
        mb.sayHello();           // "Hello!"
        mb.greet("Alice");       // "Hello, Alice!"
        int sum = mb.add(3, 5);  // 8

        // 调用 static 方法
        double area = MethodBasics.circleArea(5.0);
        System.out.println(area);  // 78.5398...
    }

    // ========== return 关键字 ==========
    // 1. 返回值的 return: return 表达式;
    // 2. void 方法的 return: return;  (可选, 用于提前退出)
    void earlyReturn(int n) {
        if (n < 0) return;  // 提前退出
        System.out.println(n);
    }

    // void 方法可以不写 return
    void noReturn() {
        System.out.println("No return needed");
    }
}
```


## 方法重载 (Overloading)


```
// ========== 方法重载 ==========
// 同类中方法名相同, 参数列表不同
// 编译时多态 (静态绑定)
// 不能仅通过返回类型区分!

public class OverloadDemo {

    // 版本 1: 两个 int
    static int max(int a, int b) {
        System.out.println("max(int, int)");
        return a > b ? a : b;
    }

    // 版本 2: 三个 int
    static int max(int a, int b, int c) {
        System.out.println("max(int, int, int)");
        return max(max(a, b), c);
    }

    // 版本 3: double
    static double max(double a, double b) {
        System.out.println("max(double, double)");
        return a > b ? a : b;
    }

    // 版本 4: 数组
    static int max(int[] arr) {
        System.out.println("max(int[])");
        int m = arr[0];
        for (int n : arr) if (n > m) m = n;
        return m;
    }

    public static void main(String[] args) {
        System.out.println(max(3, 7));        // max(int, int) → 7
        System.out.println(max(3, 7, 5));     // max(int, int, int) → 7
        System.out.println(max(3.5, 2.1));    // max(double, double) → 3.5
        System.out.println(max(new int[]{1, 5, 3})); // max(int[]) → 5

        // ========== 重载与自动提升 ==========
        // 没有精确匹配时, 自动提升类型
        System.out.println(max(3, 7L));  // max(long → double)
        // int + long → 找 max(long, long)? 没有 → 提升到 double
    }

    // ❌ 不能仅靠返回类型区分
    // int get() { return 1; }
    // double get() { return 2.0; }  // 编译错误!

    // ✅ 参数类型自动提升顺序:
    // byte → short → int → long → float → double
}
```


## 值传递 (Pass-by-Value)


```
// ========== Java 只有值传递 ==========
// 基本类型: 传值的副本
// 引用类型: 传引用的副本 (仍指向同一对象)

public class PassByValue {

    public static void main(String[] args) {
        // ========== 基本类型: 值不变 ==========
        int x = 10;
        changePrimitive(x);
        System.out.println(x);  // 10 (不变!)

        // ========== 引用类型: 对象可改, 引用不变 ==========
        int[] arr = {1, 2, 3};
        changeArray(arr);
        System.out.println(arr[0]);  // 99 (对象被修改)

        // 但引用本身不会被改变
        String name = "Alice";
        changeReference(name);
        System.out.println(name);  // "Alice" (不变)

        // ========== 交换失败 ==========
        int a = 5, b = 10;
        swap(a, b);
        System.out.println(a + ", " + b);  // 5, 10 (没交换!)
    }

    static void changePrimitive(int n) {
        n = 999;  // 只修改局部变量
    }

    static void changeArray(int[] arr) {
        arr[0] = 99;  // 通过引用修改对象内容
    }

    static void changeReference(String s) {
        s = "Bob";  // 只修改局部变量 (String 不可变)
    }

    static void swap(int a, int b) {
        int tmp = a;
        a = b;
        b = tmp;  // 只交换了局部变量
    }

    // ========== 实际应用: 通过对象包装实现"交换" ==========
    static class Holder {
        int value;
        Holder(int v) { this.value = v; }
    }

    static void swap(Holder h1, Holder h2) {
        int tmp = h1.value;
        h1.value = h2.value;
        h2.value = tmp;
    }

    // 使用:
    // Holder h1 = new Holder(5);
    // Holder h2 = new Holder(10);
    // swap(h1, h2);  // h1.value=10, h2.value=5 ✅
}
```


## 可变参数与命令行参数


```
// ========== 可变参数 (Varargs) ==========

public class VarargsDemo {

    // 可变参数必须是最后一个参数
    static int sum(int... numbers) {
        int total = 0;
        for (int n : numbers) {
            total += n;
        }
        return total;
    }

    // 多个参数 + 可变参数
    static String format(String prefix, double... values) {
        StringBuilder sb = new StringBuilder(prefix).append(": ");
        for (double v : values) {
            sb.append(String.format("%.2f ", v));
        }
        return sb.toString();
    }

    // 可变参数本质是数组
    static void printAll(String... args) {
        // 可以当作数组处理
        System.out.println("参数个数: " + args.length);
        for (String arg : args) {
            System.out.println("  " + arg);
        }
    }

    public static void main(String[] args) {
        System.out.println(sum(1, 2, 3));         // 6
        System.out.println(sum());                  // 0 (空参数)
        System.out.println(sum(1, 2, 3, 4, 5));    // 15

        System.out.println(format("Scores", 85.5, 92.0, 78.3));
        // "Scores: 85.50 92.00 78.30 "

        printAll("a", "b", "c");
        // 参数个数: 3
        //   a
        //   b
        //   c

        // 也可以直接传数组
        printAll(new String[]{"x", "y"});

        // ========== 命令行参数 ==========
        // java Main arg1 arg2 arg3
        // args[0] = "arg1"
        // args[1] = "arg2"
        // args[2] = "arg3"
    }

    // ========== main 方法详解 ==========
    // public          - JVM 能调用
    // static          - 无需创建实例
    // void            - 不返回值
    // main            - 固定名称
    // String[] args   - 命令行参数数组

    // 其他合法写法:
    // public static void main(String... args)
    // public static void main(String args[])

    // ========== 递归方法 ==========
    static int factorial(int n) {
        if (n <= 1) return 1;          // 基线条件
        return n * factorial(n - 1);   // 递归调用
    }
    // factorial(5) = 5 * 4 * 3 * 2 * 1 = 120
}
```


> **Note:** 💡 方法要点: 修饰符 + 返回类型 + 方法名 + 参数; 重载靠参数列表区分 (不是返回类型); Java 仅值传递 (基本类型副本, 引用副本指向同一对象); 可变参数 int... 相当于数组, 必须最后; main 签名固定; 递归需要基线条件防止栈溢出。


## 练习


<!-- Converted from: 8_Java 方法定义与调用.html -->
