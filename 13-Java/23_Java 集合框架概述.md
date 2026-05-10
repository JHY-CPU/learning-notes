# Java 集合框架概述


## 🗂️ Java 集合框架概述


集合框架体系、Iterable/Collection/Map 接口层次、List/Set/Map 特性对比、泛型与集合、集合 vs 数组、Collections 工具类。


## 集合框架结构


```
// ========== Java 集合框架 ==========
// 一组接口、实现类、算法, 用于存储和操作对象集合
// java.util 包核心

// ========== 接口层次 ==========
//
// Iterable (可迭代, 支持 foreach)
//   └─ Collection (集合根接口)
//        ├─ List (有序, 可重复, 有索引)
//        │    ├─ ArrayList (数组实现)
//        │    ├─ LinkedList (双向链表)
//        │    └─ Vector (线程安全, 已过时)
//        │       └─ Stack (栈, 已过时)
//        │
//        ├─ Set (无序/有序, 不重复)
//        │    ├─ HashSet (哈希表, 最快)
//        │    ├─ LinkedHashSet (插入顺序)
//        │    └─ TreeSet (排序, 红黑树)
//        │
//        └─ Queue (队列, FIFO)
//             ├─ LinkedList (也可作队列)
//             ├─ PriorityQueue (优先级队列)
//             └─ ArrayDeque (双端队列)
//
// Map (键值对, 不属于 Collection)
//   ├─ HashMap (哈希表, 最快)
//   ├─ LinkedHashMap (插入/访问顺序)
//   ├─ TreeMap (排序, 红黑树)
//   └─ ConcurrentHashMap (线程安全)

import java.util.*;

public class CollectionOverview {
    public static void main(String[] args) {
        // ========== List: 有序可重复 ==========
        List list = new ArrayList<>();
        list.add("Apple");
        list.add("Banana");
        list.add("Apple");         // ✅ 可重复
        list.get(1);               // ✅ 有索引

        // ========== Set: 不可重复 ==========
        Set set = new HashSet<>();
        set.add("Apple");
        set.add("Banana");
        set.add("Apple");          // ❌ 不会添加 (已存在)
        System.out.println(set.size());  // 2

        // ========== Queue: 队列 ==========
        Queue queue = new LinkedList<>();
        queue.offer("First");
        queue.offer("Second");
        String head = queue.poll();  // "First" (FIFO)

        // ========== Map: 键值对 ==========
        Map map = new HashMap<>();
        map.put("Alice", 25);
        map.put("Bob", 30);
        System.out.println(map.get("Alice"));  // 25
    }
}
```


## Iterable 与 Collection


```
// ========== Iterable ==========
// 可迭代: 支持增强 for 循环
// 核心方法: Iterator iterator()

public class IterableDemo {
    public static void main(String[] args) {
        List list = Arrays.asList("A", "B", "C");

        // ========== 增强 for (foreach) ==========
        for (String s : list) {
            System.out.println(s);
        }

        // ========== 迭代器 Iterator ==========
        Iterator it = list.iterator();
        while (it.hasNext()) {
            String s = it.next();
            System.out.println(s);
        }

        // ========== forEach + Lambda (Java 8+) ==========
        list.forEach(System.out::println);

        // ========== Iterator 的 remove ==========
        List nums = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5));
        Iterator iter = nums.iterator();
        while (iter.hasNext()) {
            if (iter.next() % 2 == 0) {
                iter.remove();  // ✅ 安全移除
            }
        }
        System.out.println(nums);  // [1, 3, 5]

        // ❌ 不要在 foreach 中修改集合!
        // for (int n : nums) {
        //     nums.remove(n);  // ConcurrentModificationException!
        // }
    }
}

// ========== Collection 接口核心方法 ==========
// boolean add(E e)
// boolean remove(Object o)
// boolean contains(Object o)
// int size()
// boolean isEmpty()
// void clear()
// Iterator iterator()
// Object[] toArray()
//  T[] toArray(T[] a)
// boolean addAll(Collection c)
// boolean removeAll(Collection c)
// boolean retainAll(Collection c)  // 保留交集
// boolean containsAll(Collection c)

// ========== Collections 工具类 ==========
class CollectionsUtil {
    public static void main(String[] args) {
        List list = new ArrayList<>();

        // 空集合
        List empty = Collections.emptyList();
        Set single = Collections.singleton("only");

        // 排序
        Collections.sort(list);
        Collections.reverse(list);
        Collections.shuffle(list);  // 打乱
        Collections.rotate(list, 2);  // 旋转

        // 查找
        Collections.max(list);
        Collections.min(list);
        Collections.binarySearch(list, 5);

        // 包装
        Collections.unmodifiableList(list);   // 不可变视图
        Collections.synchronizedList(list);   // 线程安全
        Collections.checkedList(list, Integer.class);  // 类型检查
    }
}
```


## 集合特性对比


```
// ========== List vs Set vs Map vs Queue ==========
//
// ┌────────────┬──────────┬──────────┬──────────┬──────────┐
// │ 特性       │ List     │ Set      │ Map      │ Queue    │
// ├────────────┼──────────┼──────────┼──────────┼──────────┤
// │ 重复       │ ✅       │ ❌       │ ❌ (key) │ ✅       │
// │ 顺序       │ 有序     │ 不一定   │ 不一定   │ FIFO     │
// │ 索引       │ ✅ (int) │ ❌       │ ✅ (key) │ ❌       │
// │ null       │ ✅       │ ✅ (1个) │ ✅ (1key)│ ❌ (某些)│
// │ 主要实现   │ ArrayList│ HashSet  │ HashMap  │ ArrayDeq │
// └────────────┴──────────┴──────────┴──────────┴──────────┘

// ========== 数组 vs 集合 ==========
// 数组: 固定长度, 基本类型, 性能好
// 集合: 可变大小, 只能存对象 (自动装箱), 功能丰富

// 数组:
int[] arr = new int[10];           // 固定
arr[0] = 42;
int len = arr.length;               // 属性

// 集合:
List nums = new ArrayList<>();  // 可变
nums.add(42);
int size = nums.size();                   // 方法

// ========== 集合初始化 ==========
// 传统:
List list1 = new ArrayList<>();
list1.add("a");
list1.add("b");
list1.add("c");

// Arrays.asList (固定大小, 不能 add/remove)
List list2 = Arrays.asList("a", "b", "c");

// List.of (Java 9+, 不可变)
List list3 = List.of("a", "b", "c");

// Set.of / Map.of (Java 9+)
Set set1 = Set.of("a", "b", "c");
Map map1 = Map.of("a", 1, "b", 2);

// ========== 集合选择指南 ==========
// 需要索引访问       → ArrayList
// 频繁插入/删除中间   → LinkedList
// 不重复             → HashSet (无序) / TreeSet (排序)
// 键值对             → HashMap (无序) / TreeMap (排序)
// 线程安全           → ConcurrentHashMap / CopyOnWriteArrayList
// FIFO 队列          → ArrayDeque / LinkedList
// 优先级队列         → PriorityQueue
// 不可变             → List.of / Set.of / Map.of
```


> **Note:** 💡 集合总览: 4 大接口 (List/Set/Map/Queue); Collection 根接口继承 Iterable; 数组固定 vs 集合可变; Arrays.asList 固定大小, List.of 不可变; 集合只存对象 (基本类型自动装箱); Iterator 迭代, 不能在 foreach 中修改; Collections 工具类排序/查找/同步/不可变包装; 遍历方式: for/foreach/Iterator/forEach。


## 练习


<!-- Converted from: 23_Java 集合框架概述.html -->
