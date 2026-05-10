# Java Arrays 类深入


## 🔧 Java Arrays 类深入


Arrays 工具类深入用法: parallelSort/mismatch/compare/parallelPrefix、对象数组排序 Comparator、Spliterator、数组与 Stream 互转、System.arraycopy 底层。


## 排序与搜索进阶


```
import java.util.Arrays;
import java.util.Comparator;

public class ArraysAdvanced {
    public static void main(String[] args) {
        // ========== 并行排序 ==========
        int[] bigArr = {5, 2, 9, 1, 7, 3, 8, 4, 6};
        Arrays.parallelSort(bigArr);  // 数据量大时用多线程
        System.out.println(Arrays.toString(bigArr));
        // [1, 2, 3, 4, 5, 6, 7, 8, 9]

        // ========== 对象数组排序 ==========
        String[] names = {"Charlie", "Alice", "Bob"};
        Arrays.sort(names);  // 自然顺序 (Comparable)
        System.out.println(Arrays.toString(names));
        // [Alice, Bob, Charlie]

        // 自定义排序 (按长度)
        Arrays.sort(names, Comparator.comparingInt(String::length));
        System.out.println(Arrays.toString(names));
        // [Bob, Alice, Charlie]

        // 自定义排序 (按长度逆序)
        Arrays.sort(names, (a, b) -> b.length() - a.length());
        System.out.println(Arrays.toString(names));
        // [Charlie, Alice, Bob]

        // ========== 二分搜索进阶 ==========
        int[] arr = {1, 3, 5, 7, 9, 11, 13};

        int idx = Arrays.binarySearch(arr, 7);    // 3
        int notFound = Arrays.binarySearch(arr, 6); // -4 (插入点: -3-1=-4)

        // 范围搜索
        int rangeIdx = Arrays.binarySearch(arr, 1, 5, 7);  // 在 [1,5) 中搜索

        // ========== 比较数组 ==========
        int[] a = {1, 2, 3, 4, 5};
        int[] b = {1, 2, 3, 4, 5};
        int[] c = {1, 2, 3, 0, 5};

        System.out.println(Arrays.equals(a, b));  // true
        System.out.println(Arrays.equals(a, c));  // false

        // Java 9+: mismatch() 返回第一个不同索引
        int mm = Arrays.mismatch(a, c);  // 3 (索引3不同)

        // Java 9+: compare() 字典序比较
        int cmp = Arrays.compare(a, b);  // 0 (相等)
        cmp = Arrays.compare(a, c);      // >0 (a[3]=4 > c[3]=0)
    }
}
```


## 数组复制与填充


```
// ========== 数组复制 ==========

public class ArrayCopy {
    public static void main(String[] args) {
        // ========== Arrays.copyOf ==========
        int[] original = {1, 2, 3, 4, 5};

        // 完整复制
        int[] copy = Arrays.copyOf(original, original.length);

        // 截断 (前3个)
        int[] truncated = Arrays.copyOf(original, 3);
        // [1, 2, 3]

        // 扩展 (末尾补默认值 0)
        int[] expanded = Arrays.copyOf(original, 7);
        // [1, 2, 3, 4, 5, 0, 0]

        // ========== Arrays.copyOfRange ==========
        int[] range = Arrays.copyOfRange(original, 1, 4);  // [1,4)
        // [2, 3, 4]

        // ========== System.arraycopy (底层) ==========
        // native 方法, 性能最高
        int[] src = {1, 2, 3, 4, 5};
        int[] dest = new int[5];
        System.arraycopy(src, 0, dest, 0, src.length);
        // dest = [1, 2, 3, 4, 5]

        // 同一数组内移动
        System.arraycopy(src, 0, src, 1, 4);
        // src = [1, 1, 2, 3, 4]

        // ========== Arrays.fill 变体 ==========
        int[] arr = new int[10];

        // 全部填充
        Arrays.fill(arr, 42);
        // [42, 42, 42, 42, 42, 42, 42, 42, 42, 42]

        // 范围填充 [2, 7)
        Arrays.fill(arr, 2, 7, 99);
        // [42, 42, 99, 99, 99, 99, 99, 42, 42, 42]

        // ========== Arrays.setAll / parallelSetAll (Java 8+) ==========
        int[] squares = new int[10];
        Arrays.setAll(squares, i -> i * i);
        // [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

        // 并行版本 (大数组)
        int[] parallel = new int[10];
        Arrays.parallelSetAll(parallel, i -> i * 2);
        // [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

        // ========== Arrays.parallelPrefix (并行前缀) ==========
        int[] prefix = {1, 2, 3, 4, 5};
        Arrays.parallelPrefix(prefix, (x, y) -> x * y);
        // [1, 2, 6, 24, 120]  (累乘)
    }
}
```


## 数组与 Stream


```
// ========== 数组 ↔ Stream ==========
import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class ArrayStream {
    public static void main(String[] args) {
        // ========== 数组 → Stream ==========

        // 基本类型数组 → IntStream/LongStream/DoubleStream
        int[] intArr = {1, 2, 3, 4, 5};
        IntStream intStream = Arrays.stream(intArr);

        // 带范围
        IntStream rangeStream = Arrays.stream(intArr, 1, 4);  // 2,3,4

        // 对象数组 → Stream
        String[] strArr = {"a", "b", "c"};
        Stream strStream = Arrays.stream(strArr);

        // Stream.of 也可
        Stream stream2 = Stream.of(strArr);

        // ========== Stream 操作示例 ==========
        // 过滤偶数并求平方
        int[] result = Arrays.stream(intArr)
            .filter(n -> n % 2 == 0)
            .map(n -> n * n)
            .toArray();
        // [4, 16]

        // 求和
        int sum = Arrays.stream(intArr).sum();        // 15
        double avg = Arrays.stream(intArr).average().orElse(0); // 3.0

        // 统计
        java.util.IntSummaryStatistics stats = Arrays.stream(intArr)
            .summaryStatistics();
        // count=5, sum=15, min=1, max=5, avg=3.0

        // ========== Stream → 数组 ==========
        IntStream is = IntStream.range(0, 10);
        int[] newArr = is.toArray();  // [0..9]

        Stream ss = Stream.of("x", "y", "z");
        String[] newStrArr = ss.toArray(String[]::new);
        // ["x", "y", "z"]
    }
}
```


## 多维数组与 Spliterator


```
// ========== 多维数组操作 ==========

public class MultiArrayOps {
    public static void main(String[] args) {
        // ========== 不规则数组深入 ==========
        int[][] triangle = new int[5][];

        for (int i = 0; i < triangle.length; i++) {
            triangle[i] = new int[i + 1];
            Arrays.fill(triangle[i], i + 1);
        }
        // [1]
        // [2, 2]
        // [3, 3, 3]
        // [4, 4, 4, 4]
        // [5, 5, 5, 5, 5]

        // ========== 深度打印 ==========
        int[][] grid = {{1, 2}, {3, 4}, {5, 6}};
        System.out.println(Arrays.toString(grid));
        // [[I@..., [I@..., [I@...]  (不好!)
        System.out.println(Arrays.deepToString(grid));
        // [[1, 2], [3, 4], [5, 6]]  ✅

        // ========== 深度比较 ==========
        int[][] a = {{1, 2}, {3, 4}};
        int[][] b = {{1, 2}, {3, 4}};
        System.out.println(Arrays.equals(a, b));      // false (浅层)
        System.out.println(Arrays.deepEquals(a, b));   // true ✅

        // ========== 多维复制 ==========
        int[][] original = {{1, 2, 3}, {4, 5, 6}};

        // 浅复制 (只复制外层)
        int[][] shallow = original.clone();
        shallow[0][0] = 99;
        System.out.println(original[0][0]);  // 99 (被修改!)

        // 深复制 (每层都复制)
        int[][] deep = new int[original.length][];
        for (int i = 0; i < original.length; i++) {
            deep[i] = original[i].clone();
        }
        deep[0][0] = 999;
        System.out.println(original[0][0]);  // 99 (不受影响)
    }
}

// ========== Spliterator (可分割迭代器) ==========
// Arrays.spliterator() 返回 Spliterator
// 用于并行遍历, Stream 底层使用

// Spliterator 特性:
// tryAdvance  - 逐个遍历
// trySplit    - 分割成两部分 (并行)
// estimateSize - 剩余元素估计
// characteristics - 特征 (ORDERED/DISTINCT/SORTED/SIZED)

// ========== 数组与泛型 ==========
// 不能直接创建泛型数组:
// T[] arr = new T[10];  // 编译错误!

// 正确方式:
// T[] arr = (T[]) new Object[10];  // 强制转换
// 或使用 Arrays.copyOf 模板方法:
// String[] strs = Arrays.copyOf(new String[0], 10);
```


> **Note:** 💡 Arrays 进阶: parallelSort 多线程排序; Comparator.comparing 自定义排序; binarySearch 未找到返回 -(插入点+1); mismatch/compare 逐元素比较; System.arraycopy 原生复制; copyOfRange 范围复制; setAll/parallelSetAll 生成; parallelPrefix 累加/累乘; Arrays.stream 转流操作; deepToString/deepEquals 多维; 多维数组深复制需手动。


## 练习


<!-- Converted from: 7_Java Arrays 类深入.html -->
