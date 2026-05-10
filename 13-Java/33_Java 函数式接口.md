# Java 函数式接口


## 🔧 Java 函数式接口


Predicate/Function/Consumer/Supplier、原始类型特化 (IntPredicate/ToIntFunction)、Operator (Unary/Binary)、Comparator、自定义函数式接口。


## 核心函数式接口


```
// ========== java.util.function 包 (Java 8+) ==========
// 4 大核心函数式接口:

import java.util.*;
import java.util.function.*;
import java.util.stream.*;

public class FunctionalInterfaces {

    public static void main(String[] args) {
        // ========== 1. Predicate — 判断 ==========
        // boolean test(T t)
        Predicate isEmpty = s -> s.isEmpty();
        Predicate startsWithA = s -> s.startsWith("A");

        System.out.println(isEmpty.test(""));           // true
        System.out.println(startsWithA.test("Apple")); // true

        // 组合
        Predicate notEmpty = isEmpty.negate();       // 非
        Predicate aAndNotEmpty = startsWithA.and(notEmpty); // 与
        Predicate aOrB = startsWithA.or(s -> s.startsWith("B")); // 或

        // 在 Stream 中使用
        List list = List.of("Apple", "Banana", "", "Avocado");
        list.stream()
            .filter(notEmpty)
            .filter(startsWithA)
            .forEach(System.out::println);  // Apple, Avocado

        // ========== 2. Function — 转换 ==========
        // R apply(T t)
        Function length = s -> s.length();
        Function upperCase = s -> s.toUpperCase();

        System.out.println(length.apply("Hello"));  // 5

        // 组合: andThen (先本函数, 再参数)
        Function upperLen = upperCase.andThen(length);
        System.out.println(upperLen.apply("hi"));  // 2

        // 组合: compose (先参数, 再本函数)
        Function addExclaim = s -> s + "!";
        Function yell = addExclaim.compose(upperCase);
        System.out.println(yell.apply("hello"));  // "HELLO!"

        // identity: 返回自身
        Function identity = Function.identity();

        // ========== 3. Consumer — 消费 ==========
        // void accept(T t)
        Consumer print = s -> System.out.println(s);
        Consumer log = s -> System.out.println("[LOG] " + s);

        print.accept("Hello");  // "Hello"

        // 组合: andThen
        Consumer printAndLog = print.andThen(log);
        printAndLog.accept("test");
        // "test"
        // "[LOG] test"

        // ========== 4. Supplier — 供给 ==========
        // T get()
        Supplier random = () -> Math.random();
        Supplier greeting = () -> "Hello, World!";

        System.out.println(random.get());  // 0.12345...
        System.out.println(greeting.get()); // "Hello, World!"

        // 懒加载
        Supplier heavyComp = () -> computeExpensive();
        // computeExpensive 只在 get() 调用时执行
    }

    static Double computeExpensive() {
        System.out.println("Computing...");
        return 42.0;
    }
}
```


## Operator 与原始类型特化


```
// ========== Operator ==========
// 特殊 Function: 输入输出类型相同

public class OperatorsDemo {
    public static void main(String[] args) {
        // ========== UnaryOperator ==========
        // Function 的特化
        UnaryOperator addStar = s -> "*" + s + "*";
        System.out.println(addStar.apply("Hi"));  // "*Hi*"

        UnaryOperator square = n -> n * n;

        // ========== BinaryOperator ==========
        // BiFunction 的特化
        BinaryOperator sum = (a, b) -> a + b;
        BinaryOperator max = BinaryOperator.maxBy(Integer::compare);
        BinaryOperator min = BinaryOperator.minBy(Integer::compare);

        // reduce 中使用
        int total = List.of(1, 2, 3, 4, 5).stream()
            .reduce(0, sum);  // 15

        // ========== Bi 版本 ==========
        // BiPredicate: boolean test(T t, U u)
        BiPredicate lengthCheck = (s, n) -> s.length() == n;
        System.out.println(lengthCheck.test("Java", 4));  // true

        // BiFunction: R apply(T t, U u)
        BiFunction repeat = (s, n) -> s.repeat(n);
        System.out.println(repeat.apply("Ha", 3));  // "HaHaHa"

        // BiConsumer: void accept(T t, U u)
        BiConsumer printKV = (k, v) ->
            System.out.println(k + "=" + v);
        printKV.accept("age", 25);
    }
}

// ========== 原始类型特化 (避免自动装箱) ==========
class PrimitiveFunctional {
    public static void main(String[] args) {
        // 输入为原始类型:
        IntPredicate isEven = n -> n % 2 == 0;
        LongSupplier nanoTime = System::nanoTime;
        DoubleConsumer print = d -> System.out.printf("%.2f%n", d);

        System.out.println(isEven.test(42));  // true (无装箱)

        // 输出为原始类型:
        ToIntFunction strLen = s -> s.length();
        int len = strLen.applyAsInt("Hello");  // 返回 int, 非 Integer

        // 输入输出均为原始:
        IntUnaryOperator doubleIt = n -> n * 2;
        IntBinaryOperator add = (a, b) -> a + b;

        // ========== 常见原始类型接口 ==========
        // ┌──────────────┬────────────────────────────┐
        // │ 输入/输出     │ 接口名                     │
        // ├──────────────┼────────────────────────────┤
        // │ int→boolean  │ IntPredicate               │
        // │ int→int      │ IntUnaryOperator           │
        // │ int,int→int  │ IntBinaryOperator          │
        // │ int→任意     │ IntFunction             │
        // │ 任意→int     │ ToIntFunction           │
        // │ 无→int       │ IntSupplier                │
        // │ int→void     │ IntConsumer                │
        // └──────────────┴────────────────────────────┘

        // Long/Double 也有对应的原始类型接口
    }
}

// ========== Comparator (重要!) ==========
class ComparatorDemo {
    public static void main(String[] args) {
        List list = new ArrayList<>(List.of("banana", "apple", "cherry", "date"));

        // 自然排序
        list.sort(Comparator.naturalOrder());

        // 逆序
        list.sort(Comparator.reverseOrder());

        // 按长度
        list.sort(Comparator.comparingInt(String::length));

        // 多重比较
        list.sort(Comparator
            .comparingInt(String::length)
            .thenComparing(Comparator.naturalOrder())
        );

        // null 处理
        Comparator nullSafe = Comparator.nullsLast(
            Comparator.naturalOrder()
        );

        // 自定义比较器
        Comparator byLastChar =
            Comparator.comparing(s -> s.charAt(s.length() - 1));
    }
}
```


## 实战与组合


```
// ========== 函数式接口实战 ==========
import java.util.function.*;

public class FunctionalPractice {

    // ========== 1. 策略模式 (用 Function 替代) ==========
    // 传统: 需要接口 + 多个实现类
    // 函数式: 直接传 Lambda

    static double processValue(double value, UnaryOperator processor) {
        return processor.apply(value);
    }

    // 使用:
    // double r1 = processValue(5.0, x -> x * 2);           // 10
    // double r2 = processValue(5.0, x -> Math.round(x));    // 5
    // double r3 = processValue(5.0, Math::sqrt);            // 2.236

    // ========== 2. 管道模式 ==========
    static class Pipeline {
        static  T process(T input, List> steps) {
            T result = input;
            for (UnaryOperator step : steps) {
                result = step.apply(result);
            }
            return result;
        }
    }

    // 使用:
    // String result = Pipeline.process(" hello ", List.of(
    //     String::trim,
    //     String::toUpperCase,
    //     s -> s + "!"
    // ));  // "HELLO!"

    // ========== 3. 缓存/记忆化 ==========
    static  Function memoize(Function fn) {
        Map cache = new HashMap<>();
        return t -> cache.computeIfAbsent(t, fn);
    }

    // 使用:
    // Function slowFib = n -> {
    //     if (n <= 1) return n;
    //     return slowFib.apply(n - 1) + slowFib.apply(n - 2);  // 注意: 递归会出问题
    // };
    // Function fastFib = memoize(slowFib);

    // ========== 4. 验证器 ==========
    static class Validator {
        private final T value;
        private final List errors = new ArrayList<>();

        Validator(T value) { this.value = value; }

        Validator validate(Predicate test, String message) {
            if (!test.test(value)) errors.add(message);
            return this;
        }

        List getErrors() { return errors; }
        boolean isValid() { return errors.isEmpty(); }
    }

    // 使用:
    // var v = new Validator("")
    //     .validate(s -> !s.isEmpty(), "不能为空")
    //     .validate(s -> s.length() >= 3, "长度至少3");
    // v.isValid();  // false

    // ========== 函数式接口总结 ==========
    // 常用接口速查:
    // Predicate     T → boolean     filter/match
    // Function    T → R           map
    // Consumer      T → void        forEach
    // Supplier      () → T          lazy/orElseGet
    // UnaryOperator T → T           transform
    // BinaryOperator (T,T) → T      reduce
    // Comparator    (T,T) → int     sort
}
```


> **Note:** 💡 函数式接口要点: 4 大核心 (Predicate/Function/Consumer/Supplier); UnaryOperator/BinaryOperator 特化; Bi 版本 (BiPredicate/BiFunction/BiConsumer); 原始类型 (IntPredicate/ToIntFunction) 避免装箱; andThen/compose/and/or/negate 组合; Comparator.comparingInt/thenComparing 链式; 自定 @FunctionalInterface。


## 练习


<!-- Converted from: 33_Java 函数式接口.html -->
