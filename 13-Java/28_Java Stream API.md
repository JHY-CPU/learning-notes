# Java Stream API


## 🌊 Java Stream API


Stream 流水线、创建 Stream、中间操作 (filter/map/flatMap/sorted/peek)、终端操作 (collect/toList/groupingBy/partitioningBy)、原始类型流、并行流。


## Stream 基础


```
// ========== Stream API (Java 8+) ==========
// 函数式方式处理集合数据
// 特点:
// 1. 不存储数据
// 2. 不修改源数据
// 3. 惰性求值 (中间操作延迟)
// 4. 可无限 (limit 截断)
// 5. 可串行/并行

import java.util.*;
import java.util.stream.*;

public class StreamBasics {
    public static void main(String[] args) {
        // ========== 创建 Stream ==========
        // 1. 从集合
        List list = List.of("a", "b", "c");
        Stream s1 = list.stream();
        Stream s2 = list.parallelStream();  // 并行

        // 2. 从数组
        Stream s3 = Arrays.stream(new String[]{"a", "b"});
        IntStream s4 = Arrays.stream(new int[]{1, 2, 3});

        // 3. Stream.of
        Stream s5 = Stream.of("a", "b", "c");
        Stream s6 = Stream.of(1, 2, 3);

        // 4. Stream.iterate / Stream.generate
        Stream s7 = Stream.iterate(0, n -> n + 1);  // 0,1,2,...
        Stream s8 = Stream.generate(Math::random);    // 无限随机

        // 5. 范围
        IntStream s9 = IntStream.range(1, 5);     // 1,2,3,4
        IntStream s10 = IntStream.rangeClosed(1, 5); // 1,2,3,4,5

        // 6. 字符串
        IntStream s11 = "hello".chars();  // 字符的 IntStream

        // ========== Stream 生命周期 ==========
        // 创建 → 0..n 个中间操作 → 1 个终端操作

        List result = list.stream()       // 1. 创建
            .filter(s -> s.length() > 0)          // 2. 中间操作
            .map(String::toUpperCase)             // 3. 中间操作
            .collect(Collectors.toList());        // 4. 终端操作

        // Stream 用完即关闭, 不能重复使用!
        // s1.filter(...);  // ❌ 已关闭!
    }
}
```


## 中间操作


```
// ========== 中间操作 (返回 Stream) ==========
// 惰性执行, 直到终端操作才触发

public class IntermediateOps {
    public static void main(String[] args) {
        List words = List.of("apple", "banana", "avocado", "cherry", "apricot");

        // ========== filter: 过滤 ==========
        words.stream()
            .filter(w -> w.startsWith("a"))
            .forEach(System.out::println);  // apple, avocado, apricot

        // ========== map: 映射/转换 ==========
        words.stream()
            .map(String::toUpperCase)
            .map(w -> w + "!")
            .forEach(System.out::println);  // APPLE!, BANANA!, ...

        // ========== flatMap: 展平 ==========
        List> nested = List.of(
            List.of(1, 2), List.of(3, 4, 5), List.of(6)
        );
        nested.stream()
            .flatMap(Collection::stream)       // 展平为单个流
            .map(n -> n * 10)
            .forEach(System.out::println);     // 10, 20, 30, 40, 50, 60

        // flatMap 字符串拆分
        words.stream()
            .flatMap(w -> Arrays.stream(w.split("")))  // 拆成字符
            .distinct()
            .forEach(System.out::print);  // a p l e b n v o c h r y

        // ========== distinct: 去重 ==========
        List.of(1, 2, 2, 3, 3, 3).stream()
            .distinct()
            .forEach(System.out::print);  // 1 2 3

        // ========== sorted: 排序 ==========
        words.stream()
            .sorted()                               // 自然排序
            .sorted(Comparator.comparingInt(String::length))  // 按长度
            .forEach(System.out::println);

        // ========== peek: 调试/查看 (中间 forEach) ==========
        long count = words.stream()
            .peek(w -> System.out.println("Before: " + w))
            .filter(w -> w.length() > 5)
            .peek(w -> System.out.println("After filter: " + w))
            .count();

        // ========== limit / skip ==========
        IntStream.range(1, 100)
            .skip(10)     // 跳过前 10 个
            .limit(5)     // 取 5 个
            .forEach(System.out::print);  // 11,12,13,14,15

        // ========== takeWhile / dropWhile (Java 9+) ==========
        IntStream.range(1, 20)
            .takeWhile(n -> n < 10)    // 拿直到条件为 false
            .forEach(System.out::print);  // 1,2,3,4,5,6,7,8,9

        IntStream.range(1, 10)
            .dropWhile(n -> n < 5)     // 丢掉直到条件为 false
            .forEach(System.out::print);  // 5,6,7,8,9
    }
}
```


## 终端操作


```
// ========== 终端操作 (触发执行, 关闭 Stream) ==========
import java.util.stream.Collectors;

public class TerminalOps {
    public static void main(String[] args) {
        List nums = List.of(3, 1, 4, 1, 5, 9, 2, 6);

        // ========== 遍历 ==========
        nums.stream().forEach(System.out::println);
        nums.stream().forEachOrdered(System.out::println);  // 保证顺序

        // ========== 聚合 ==========
        long count = nums.stream().count();               // 8
        Optional min = nums.stream().min(Integer::compare);  // 1
        Optional max = nums.stream().max(Integer::compare);  // 9

        // ========== 匹配 ==========
        boolean anyMatch = nums.stream().anyMatch(n -> n > 5);     // true
        boolean allMatch = nums.stream().allMatch(n -> n > 0);     // true
        boolean noneMatch = nums.stream().noneMatch(n -> n < 0);   // true

        // ========== 查找 ==========
        Optional first = nums.stream().findFirst();  // 3
        Optional any = nums.parallelStream().findAny();  // 任意 (并)

        // ========== reduce (归约) ==========
        // 求和
        Optional sum1 = nums.stream().reduce(Integer::sum);
        Integer sum2 = nums.stream().reduce(0, Integer::sum);  // 有初始值

        // 手动归约
        Optional product = nums.stream()
            .reduce((a, b) -> a * b);  // 乘积

        // ========== collect (收集) ==========
        // 收集到 List
        List list = words.stream()
            .filter(w -> w.length() > 5)
            .collect(Collectors.toList());

        // 收集到 Set
        Set set = words.stream()
            .collect(Collectors.toSet());

        // 收集到特定集合
        ArrayList arrayList = words.stream()
            .collect(Collectors.toCollection(ArrayList::new));

        // 收集到 Map
        Map map = words.stream()
            .collect(Collectors.toMap(
                w -> w,                  // 键: 单词本身
                String::length,          // 值: 单词长度
                (v1, v2) -> v1           // 冲突: 保留第一个
            ));

        // ========== joining (字符串拼接) ==========
        String joined = words.stream()
            .collect(Collectors.joining(", ", "[", "]"));
        // "[apple, banana, avocado, cherry, apricot]"
    }
}
```


## Collectors: groupingBy & partitioningBy


```
// ========== 高级收集 ==========

public class GroupingDemo {
    public static void main(String[] args) {
        List words = List.of("apple", "banana", "avocado", "cherry",
            "apricot", "blueberry", "date");

        // ========== groupingBy: 分组 ==========
        // 按首字母分组
        Map> byFirst = words.stream()
            .collect(Collectors.groupingBy(w -> w.charAt(0)));
        System.out.println(byFirst);
        // {a=[apple, avocado, apricot], b=[banana, blueberry], c=[cherry], d=[date]}

        // 分组后计数
        Map countByFirst = words.stream()
            .collect(Collectors.groupingBy(
                w -> w.charAt(0),
                Collectors.counting()
            ));
        // {a=3, b=2, c=1, d=1}

        // 分组后映射
        Map> lengthsByFirst = words.stream()
            .collect(Collectors.groupingBy(
                w -> w.charAt(0),
                Collectors.mapping(String::length, Collectors.toList())
            ));
        // {a=[5, 7, 7], b=[6, 9], c=[6], d=[4]}

        // 分组后求平均
        Map avgByFirst = words.stream()
            .collect(Collectors.groupingBy(
                w -> w.charAt(0),
                Collectors.averagingInt(String::length)
            ));

        // ========== partitioningBy: 分区 (按 boolean) ==========
        Map> longShort = words.stream()
            .collect(Collectors.partitioningBy(w -> w.length() > 5));
        // true=[banana, avocado, cherry, apricot, blueberry]
        // false=[apple, date]

        // 分区后计数
        Map countLongShort = words.stream()
            .collect(Collectors.partitioningBy(
                w -> w.length() > 5,
                Collectors.counting()
            ));
        // {false=2, true=5}

        // ========== summarizing (统计摘要) ==========
        IntSummaryStatistics stats = words.stream()
            .collect(Collectors.summarizingInt(String::length));
        System.out.println(stats);
        // IntSummaryStatistics{count=7, sum=43, min=4, average=6.14, max=9}

        // ========== teeing (Java 12+) ==========
        // 同时执行两个收集器
        record Result(long count, long totalLength) {}

        Result r = words.stream()
            .collect(Collectors.teeing(
                Collectors.counting(),
                Collectors.summingInt(String::length),
                Result::new
            ));
        // Result[count=7, totalLength=43]
    }
}

// ========== 原始类型流 ==========
// IntStream, LongStream, DoubleStream
// 避免自动装箱开销

class PrimitiveStreams {
    public static void main(String[] args) {
        int[] arr = {3, 1, 4, 1, 5, 9, 2};

        int sum = Arrays.stream(arr).sum();           // 25
        double avg = Arrays.stream(arr).average().orElse(0);
        int min = Arrays.stream(arr).min().orElse(0);
        int max = Arrays.stream(arr).max().orElse(0);

        // 基本类型 → 对象
        Stream boxed = IntStream.range(1, 10).boxed();

        // 数值范围
        int rangeSum = IntStream.rangeClosed(1, 100).sum();  // 5050
    }
}
```


## 并行流与性能


```
// ========== 并行流 (Parallel Stream) ==========
// 使用 ForkJoinPool 自动分割任务
// 底层: 集合的 spliterator.trySplit()

public class ParallelStreamDemo {
    public static void main(String[] args) {
        // ========== 创建并行流 ==========
        List list = IntStream.rangeClosed(1, 10_000_000).boxed().toList();

        // 串行
        long start = System.currentTimeMillis();
        list.stream()
            .filter(n -> n % 2 == 0)
            .mapToLong(n -> (long) n * n)
            .sum();
        long seq = System.currentTimeMillis() - start;

        // 并行
        start = System.currentTimeMillis();
        list.parallelStream()
            .filter(n -> n % 2 == 0)
            .mapToLong(n -> (long) n * n)
            .sum();
        long par = System.currentTimeMillis() - start;

        System.out.println("Sequential: " + seq + "ms");
        System.out.println("Parallel: " + par + "ms");

        // ========== 适合并行的情况 ==========
        // ✅ 数据量大 (>10000)
        // ✅ 计算密集型 (CPU 密集)
        // ✅ 无状态操作 (不依赖外部可变状态)
        // ✅ 独立元素 (元素间无依赖)

        // ========== 不适合并行的情况 ==========
        // ❌ 数据量小
        // ❌ I/O 密集 (文件/网络)
        // ❌ 有状态操作 (limit/findFirst 依赖顺序)
        // ❌ 上游操作昂贵但下游合并更贵

        // ========== 并行流的线程池 ==========
        // 默认使用 ForkJoinPool.commonPool()
        // 线程数 = Runtime.getRuntime().availableProcessors() - 1

        // 自定义线程池 (不推荐全局修改):
        // System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", "4");

        // ========== Stream 注意事项 ==========
        // 1. 不要修改 Stream 的数据源
        // 2. Stream 只能使用一次
        // 3. 中间操作惰性求值
        // 4. 并行流不一定比串行快
        // 5. 避免在 forEach 中修改共享变量
    }
}

// ========== Stream 总结 ==========
// 中间操作: filter, map, flatMap, distinct, sorted, peek, limit, skip,
//          takeWhile, dropWhile
// 终端操作: forEach, count, min/max, anyMatch/allMatch/noneMatch,
//          findFirst/findAny, reduce, collect, toArray, sum/average
// 收集器:  toList, toSet, toMap, groupingBy, partitioningBy, joining,
//          mapping, filtering, flatMapping, teeing
```


> **Note:** 💡 Stream 要点: 创建 (集合/数组/of/iterate/range); 中间操作惰性求值; 终端操作触发执行; flatMap 展平嵌套; groupingBy/partitioningBy 分组; reduce 归约; parallelStream 可能加速 CPU 密集任务; Stream 只能用一次; 不修改源数据; Collectors 收集到各种容器。


## 练习


<!-- Converted from: 28_Java Stream API.html -->
