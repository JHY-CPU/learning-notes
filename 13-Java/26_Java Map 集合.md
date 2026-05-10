# Java Map 集合


## 🗺️ Java Map 集合


Map 接口、HashMap (哈希表/原理/扩容)、LinkedHashMap (插入/访问顺序)、TreeMap (红黑树/排序)、ConcurrentHashMap、Map 遍历方式、HashMap 原理 (桶/红黑树/扩容)。


## Map 基础


```
// ========== Map ==========
// 键值对存储, 每个键映射到一个值
// 键不重复, 值可重复
// 键可以为 null (HashMap), 值可以为 null

import java.util.*;

public class MapBasics {
    public static void main(String[] args) {
        // ========== 创建 Map ==========
        Map map = new HashMap<>();
        Map map2 = Map.of("a", 1, "b", 2);  // 不可变 (Java 9+)

        // ========== 基本操作 ==========
        // 添加/修改
        map.put("Alice", 25);
        map.put("Bob", 30);
        map.put("Alice", 26);  // 覆盖: value 变为 26

        // 读取
        int age = map.get("Alice");       // 26
        int defaultValue = map.getOrDefault("Charlie", 0);  // 0

        // 判断
        boolean hasKey = map.containsKey("Alice");    // true
        boolean hasValue = map.containsValue(30);     // true
        int size = map.size();                        // 2
        boolean empty = map.isEmpty();                // false

        // 删除
        map.remove("Bob");          // 按键删除
        map.remove("Alice", 25);    // 键值都匹配才删除

        // 清空
        // map.clear();

        // ========== 键值视图 ==========
        System.out.println(map);  // {Alice=26}

        Map data = new HashMap<>(Map.of("a", 1, "b", 2, "c", 3));

        // keySet: 所有键
        Set keys = data.keySet();
        System.out.println(keys);  // [a, b, c]

        // values: 所有值
        Collection values = data.values();
        System.out.println(values);  // [1, 2, 3]

        // entrySet: 所有键值对
        Set> entries = data.entrySet();
        for (Map.Entry e : entries) {
            System.out.println(e.getKey() + " = " + e.getValue());
        }
    }
}
```


## HashMap 原理


```
// ========== HashMap 原理 ==========
// 底层: 数组 + 链表 + 红黑树 (Java 8+)
// 默认容量: 16
// 加载因子: 0.75 (扩容阈值 = 容量 × 0.75)
// 扩容: 2 倍 (新容量 = 旧容量 << 1)

public class HashMapPrinciple {
    // ========== 存储结构 ==========
    // Java 8+:
    // 数组 (Node[] table)
    //   ├─ Node1 → Node2 → Node3   (链表: ≤7)
    //   ├─ TreeNode1 ↔ TreeNode2   (红黑树: ≥8)
    //   └─ null

    // ========== put(key, value) 流程 ==========
    // 1. 计算 hashCode(key) → int h
    // 2. 扰动: (h = key.hashCode()) ^ (h >>> 16)
    // 3. 定位: index = (n - 1) & hash  (n = table.length)
    // 4. 如果桶为空 → 新建 Node
    // 5. 如果桶不空:
    //    a. 检查第一个节点 (key 相同 → 覆盖 value)
    //    b. 如果是 TreeNode → 红黑树插入
    //    c. 否则遍历链表:
    //       - key 相同 → 覆盖
    //       - 到尾端 → 新建节点插入尾部
    //       - 链表长度 ≥ 8 → 树化 (红黑树)
    // 6. 检查 size > threshold → resize (扩容 2 倍)

    // ========== get(key) 流程 ==========
    // 1. 计算 hash
    // 2. 定位桶
    // 3. 第一个节点 → 匹配返回
    // 4. 红黑树 → treeNode.getTreeNode()
    // 5. 链表 → 遍历查找

    // ========== 扩容 resize() ==========
    // 1. 新数组 = 旧数组 × 2
    // 2. 重新计算每个元素位置
    // 3. 元素在新数组位置: 原位置 或 原位置+旧容量
    //   (利用: hash & oldCap == 0 → 原位; == 1 → 原位置+oldCap)
    // 4. 链表可能会拆分 → 高低位两条链表

    // ========== 线程安全 ==========
    // HashMap 非线程安全!
    // Java 7: resize 时可能死循环 (环形链表)
    // Java 8: 不会死循环, 但可能丢失数据

    // 替代方案:
    // 1. ConcurrentHashMap
    // 2. Collections.synchronizedMap(new HashMap<>())
    // 3. synchronized 块
}

// ========== HashMap 并发问题演示 ==========
class HashMapConcurrency {
    public static void main(String[] args) throws Exception {
        // ❌ 不要在并发中直接使用 HashMap
        Map unsafe = new HashMap<>();
        Set results = ConcurrentHashMap.newKeySet();

        Runnable task = () -> {
            for (int i = 0; i < 1000; i++) {
                unsafe.put("key-" + i, i);  // 可能出问题
            }
        };

        Thread t1 = new Thread(task);
        Thread t2 = new Thread(task);
        t1.start();
        t2.start();
        t1.join();
        t2.join();

        System.out.println("Expected: 1000, Actual: " + unsafe.size());
        // 可能 < 1000 或抛异常
    }
}
```


## LinkedHashMap & TreeMap


```
// ========== LinkedHashMap ==========
// 继承 HashMap, 额外维护双向链表
// 可维护插入顺序或访问顺序

public class LinkedHashMapDemo {
    public static void main(String[] args) {
        // ========== 插入顺序 (默认) ==========
        LinkedHashMap map = new LinkedHashMap<>();
        map.put("A", 1);
        map.put("B", 2);
        map.put("C", 3);
        System.out.println(map);  // {A=1, B=2, C=3}

        // ========== 访问顺序 (LRU 缓存) ==========
        LinkedHashMap lru = new LinkedHashMap<>(
            16, 0.75f, true  // accessOrder = true
        );
        lru.put("A", 1);
        lru.put("B", 2);
        lru.put("C", 3);
        lru.get("A");         // 访问 A → A 移到最后
        System.out.println(lru);  // {B=2, C=3, A=1}

        // ========== 实现 LRU 缓存 ==========
        class LRUCache extends LinkedHashMap {
            private final int maxCapacity;

            public LRUCache(int maxCapacity) {
                super(16, 0.75f, true);  // accessOrder
                this.maxCapacity = maxCapacity;
            }

            @Override
            protected boolean removeEldestEntry(Map.Entry eldest) {
                return size() > maxCapacity;  // 超过容量移除最旧的
            }
        }

        LRUCache cache = new LRUCache<>(3);
        cache.put("a", "1");
        cache.put("b", "2");
        cache.put("c", "3");
        cache.get("a");           // 访问 a
        cache.put("d", "4");      // 超过容量, 移除最久未访问的 b
        System.out.println(cache);  // {c=3, a=1, d=4}
    }
}

// ========== TreeMap ==========
// 底层: 红黑树
// 键排序 (自然顺序或 Comparator)
// O(log n) 操作

class TreeMapDemo {
    public static void main(String[] args) {
        // 自然排序
        TreeMap map = new TreeMap<>();
        map.put("Charlie", 3);
        map.put("Alice", 1);
        map.put("Bob", 2);
        System.out.println(map);  // {Alice=1, Bob=2, Charlie=3}

        // 自定义排序 (按值)
        TreeMap byValue = new TreeMap<>(
            Comparator.comparingInt(map::get)
        );
        // 注意: 一般不用 TreeMap 按键以外的排序

        // TreeMap 特有方法
        System.out.println(map.firstKey());        // "Alice"
        System.out.println(map.lastKey());         // "Charlie"
        System.out.println(map.lowerKey("Bob"));   // "Alice" (Bob)

        // 子图
        System.out.println(map.subMap("Alice", "Charlie"));  // {Alice=1, Bob=2}
        System.out.println(map.headMap("Bob"));     // {Alice=1}
        System.out.println(map.tailMap("Bob"));     // {Bob=2, Charlie=3}
    }
}
```


## ConcurrentHashMap & 遍历


```
// ========== ConcurrentHashMap ==========
// 线程安全, 高性能 (分段锁/无锁)
// 读: 无锁 (volatile)
// 写: 锁住单个桶 (细粒度)
// 迭代器: 弱一致性 (不抛 ConcurrentModificationException)

import java.util.concurrent.ConcurrentHashMap;

public class CHMDemo {
    public static void main(String[] args) {
        ConcurrentHashMap map = new ConcurrentHashMap<>();
        map.put("A", 1);
        map.put("B", 2);

        // 原子操作
        map.putIfAbsent("A", 10);           // 不存在才放 (A不变)
        map.replace("A", 1, 100);            // 值匹配才替换
        map.remove("B", 2);                  // 键值都匹配才删除

        // Java 8+ 并发方法
        map.compute("A", (k, v) -> v == null ? 1 : v + 1);
        map.computeIfAbsent("C", k -> 3);    // 键不存在才计算
        map.computeIfPresent("A", (k, v) -> v + 1);

        map.merge("A", 1, Integer::sum);     // 合并

        // 遍历 (弱一致性, 安全)
        map.forEach(2, (k, v) ->
            System.out.println(k + " = " + v)
        );

        // 搜索
        String result = map.search(2, (k, v) -> v > 10 ? k : null);
    }
}

// ========== 完整遍历方式 ==========
class MapIteration {
    public static void main(String[] args) {
        Map map = new HashMap<>(Map.of("a", 1, "b", 2, "c", 3));

        // 1. entrySet + for-each
        for (Map.Entry e : map.entrySet()) {
            System.out.println(e.getKey() + "=" + e.getValue());
        }

        // 2. keySet + get (慢, 不推荐)
        for (String key : map.keySet()) {
            System.out.println(key + "=" + map.get(key));
        }

        // 3. values (只有值)
        for (int v : map.values()) {
            System.out.println(v);
        }

        // 4. Iterator (可删除)
        Iterator> it = map.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry e = it.next();
            if (e.getValue() == 2) it.remove();
        }

        // 5. forEach (Java 8+)
        map.forEach((k, v) -> System.out.println(k + "=" + v));

        // 6. Stream
        map.entrySet().stream()
            .filter(e -> e.getValue() > 1)
            .forEach(e -> System.out.println(e.getKey()));
    }
}

// ========== Map 实现对比 ==========
// ┌──────────────┬──────────┬──────────┬──────────┬──────────────┐
// │ 特性         │ HashMap  │ LHM      │ TreeMap  │ ConcurrentHM  │
// ├──────────────┼──────────┼──────────┼──────────┼──────────────┤
// │ 顺序         │ 无序     │ 插入/访问 │ 排序     │ 无序         │
// │ 性能         │ O(1)     │ O(1)     │ O(log n) │ O(1)         │
// │ 线程安全     │ ❌       │ ❌       │ ❌       │ ✅           │
// │ null 键      │ ✅ 1个   │ ✅ 1个   │ ❌       │ ❌           │
// └──────────────┴──────────┴──────────┴──────────┴──────────────┘
```


> **Note:** 💡 Map 要点: HashMap O(1) 基于哈希表 (数组+链表+红黑树); 初始容量 16, 加载因子 0.75, 扩容 2 倍; LinkedHashMap 维护插入/访问顺序; TreeMap 红黑树 O(log n); ConcurrentHashMap 线程安全; Map 遍历用 entrySet 效率最高; putIfAbsent/compute/merge 原子操作; HashMap 非线程安全。


## 练习


<!-- Converted from: 26_Java Map 集合.html -->
