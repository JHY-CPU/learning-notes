# Java Optional 类


## 🎁 Java Optional 类


Optional 创建 (of/ofNullable/empty)、消费 (ifPresent/ifPresentOrElse)、转换 (map/flatMap/filter)、取值 (orElse/orElseGet/orElseThrow)、Optional 最佳实践。


## Optional 基础


```
// ========== Optional (Java 8+) ==========
// 容器: 可能包含值, 也可能为空
// 目的: 避免 NullPointerException
//      明确表示"可能为空"

import java.util.*;

public class OptionalBasics {

    // ========== 创建 Optional ==========
    public static void main(String[] args) {
        // 1. Optional.of(value)  — 值不能为 null
        Optional opt1 = Optional.of("Hello");
        // Optional optNull = Optional.of(null);  // NullPointerException!

        // 2. Optional.ofNullable(value) — 可以为 null
        Optional opt2 = Optional.ofNullable(null);   // 空
        Optional opt3 = Optional.ofNullable("Hi");   // 有值

        // 3. Optional.empty() — 明确表示空
        Optional empty = Optional.empty();

        // ========== 判断 ==========
        System.out.println(opt1.isPresent());   // true (有值)
        System.out.println(empty.isPresent());  // false

        System.out.println(opt1.isEmpty());     // false (Java 11+)
        System.out.println(empty.isEmpty());    // true
    }
}

// ========== 取值 (谨慎!) ==========
class OptionalGet {
    public static void main(String[] args) {
        Optional opt = Optional.of("Hello");

        // ========== get() — 不推荐! ==========
        // 值存在 → 返回值
        // 值不存在 → NoSuchElementException
        System.out.println(opt.get());  // "Hello"
        // empty.get();  // ❌ NoSuchElementException!

        // ========== 安全取值 ==========
        // orElse: 有值返回值, 无值返回默认
        String r1 = opt.orElse("default");

        // orElseGet: 有值返回值, 无值执行 Supplier (懒加载)
        String r2 = opt.orElseGet(() -> computeDefault());

        // orElseThrow: 有值返回值, 无值抛异常
        String r3 = opt.orElseThrow(() -> new IllegalArgumentException("No value"));

        // orElseThrow(): 无参版本, 抛 NoSuchElementException (Java 10+)
        String r4 = opt.orElseThrow();  // 比 get() 语义更好

        // or: 有值返回自身, 无值执行 Supplier (Java 9+)
        Optional r5 = opt.or(() -> Optional.of("fallback"));
    }

    static String computeDefault() {
        System.out.println("Computing default...");
        return "computed";
    }
}
```


## Optional 操作


```
// ========== 消费 (ifPresent) ==========

public class OptionalOps {

    public static void main(String[] args) {
        Optional opt = Optional.of("Hello");

        // ========== ifPresent ==========
        opt.ifPresent(s -> System.out.println(s.length()));  // 5

        // ========== ifPresentOrElse (Java 9+) ==========
        opt.ifPresentOrElse(
            s -> System.out.println("Value: " + s),
            () -> System.out.println("No value")
        );

        Optional empty = Optional.empty();
        empty.ifPresentOrElse(
            s -> System.out.println("Value: " + s),
            () -> System.out.println("No value")  // 执行这个
        );

        // ========== map: 转换 (类似 Stream.map) ==========
        Optional name = Optional.of("Alice");
        Optional len = name.map(String::length);  // Optional[5]
        Optional upper = name.map(String::toUpperCase);

        // 链式调用 (无 NPE 风险)
        String result = name
            .map(String::trim)
            .filter(s -> s.length() > 3)
            .map(String::toUpperCase)
            .orElse("DEFAULT");

        // ========== flatMap: 展平 (避免嵌套 Optional) ==========
        // map 可能产生 Optional>
        Optional opt1 = Optional.of("hello");
        Optional> nested = opt1.map(s -> Optional.of(s.length()));
        // 需要两次 get

        Optional flat = opt1.flatMap(s -> Optional.of(s.length()));
        // 一次 get

        // 实际应用: 避免嵌套
        class User {
            Optional getEmail() { return Optional.of("a@b.com"); }
        }
        Optional user = Optional.of(new User());

        // map 嵌套:
        Optional> email1 = user.map(u -> u.getEmail());

        // flatMap 展平:
        Optional email2 = user.flatMap(u -> u.getEmail());

        // ========== filter: 过滤 ==========
        Optional filtered = name
            .filter(s -> s.startsWith("A"))   // "Alice" → 保留
            .filter(s -> s.length() > 10);     // false → Optional.empty
        System.out.println(filtered);  // Optional.empty

        // ========== stream() (Java 9+) ==========
        // Optional → Stream (0 或 1 个元素)
        Optional val = Optional.of("Hello");
        val.stream()           // Stream with 1 element
            .map(String::length)
            .forEach(System.out::println);

        // 过滤空值:
        List> list = List.of(
            Optional.of("a"),
            Optional.empty(),
            Optional.of("b")
        );
        List values = list.stream()
            .filter(Optional::isPresent)
            .map(Optional::get)
            .toList();
        System.out.println(values);  // [a, b]

        // Java 9+ 更简单:
        List flatValues = list.stream()
            .flatMap(Optional::stream)
            .toList();  // [a, b]
    }
}
```


## Optional 实战: 避免 null 检查


```
// ========== Optional 实战 ==========
// 传统 null 检查 vs Optional

import java.util.*;

public class OptionalPractice {

    // ========== 传统方式 (容易 NPE) ==========
    static String getCityTraditional(User user) {
        if (user != null) {
            Address addr = user.getAddress();
            if (addr != null) {
                City city = addr.getCity();
                if (city != null) {
                    return city.getName();
                }
            }
        }
        return "Unknown";
    }

    // ========== Optional 方式 ==========
    static String getCityOptional(User user) {
        return Optional.ofNullable(user)
            .map(User::getAddress)
            .map(Address::getCity)
            .map(City::getName)
            .orElse("Unknown");
    }

    // ========== 辅助类 ==========
    static class User {
        private Address address;
        Address getAddress() { return address; }
    }
    static class Address {
        private City city;
        City getCity() { return city; }
    }
    static class City {
        private String name;
        String getName() { return name; }
    }

    // ========== 在方法返回中使用 Optional ==========
    // ✅ 好: 返回 Optional 表示可能无值
    static Optional findUserName(int id) {
        if (id == 1) return Optional.of("Alice");
        return Optional.empty();
    }

    // ❌ 不好: 返回 null
    static String findUserNameBad(int id) {
        if (id == 1) return "Alice";
        return null;  // 调用方必须检查 null!
    }

    // ❌ 不好: 为集合返回 Optional
    // 集合应该返回空集合, 而不是 Optional
    static List getNames() {
        return Collections.emptyList();  // ✅ 空集合, 不是 Optional
    }

    public static void main(String[] args) {
        // 使用 Optional 返回值
        Optional name = findUserName(1);
        name.ifPresent(System.out::println);

        String display = findUserName(999)
            .orElse("Guest");
        System.out.println(display);  // "Guest"
    }
}

// ========== Optional 最佳实践 ==========
// ✅ 适合:
//   - 方法返回值: 可能没有结果
//   - 链式调用: 避免深层 null 检查
//   - 与 Stream 一起使用

// ❌ 不适合:
//   - 字段类型 (序列化问题)
//   - 方法参数 (调用方可能传 null)
//   - 集合 (用空集合替代)
//   - 基本类型包装的性能开销 (用 OptionalInt/OptionalLong/OptionalDouble)

// ========== 原始类型 Optional ==========
// OptionalInt — 避免装箱
// OptionalLong
// OptionalDouble
//
// OptionalInt opt = OptionalInt.of(42);
// int val = opt.orElse(0);
```


> **Note:** 💡 Optional 要点: of(非null)/ofNullable(可为null)/empty() 创建; orElse/orElseGet/orElseThrow 安全取值; map/flatMap/filter 链式操作; ifPresent/ifPresentOrElse 消费; 返回类型用 Optional 表示可能无值; 不要用于字段/参数/集合; flatMap 避免嵌套 Optional; stream() (Java 9+) 转 Stream。


## 练习


<!-- Converted from: 32_Java Optional 类.html -->
