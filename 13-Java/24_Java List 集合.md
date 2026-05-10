# Java List 集合


## 📋 Java List 集合


List 接口、ArrayList (数组实现)、LinkedList (链表实现)、ListIterator、子列表 subList、ArrayList vs LinkedList 性能对比、线程安全。


## List 接口


```
// ========== List 接口特性 ==========
// 1. 有序 (插入顺序)
// 2. 可重复
// 3. 有索引 (int index)

import java.util.*;

public class ListBasics {
    public static void main(String[] args) {
        // ========== 创建 List ==========
        List list1 = new ArrayList<>();
        List list2 = new LinkedList<>();
        List list3 = Arrays.asList("a", "b", "c");  // 固定大小
        List list4 = List.of("x", "y", "z");        // 不可变 (Java 9+)

        // ========== 基本操作 ==========
        List list = new ArrayList<>();

        // 添加
        list.add("Apple");         // 末尾
        list.add("Banana");
        list.add(1, "Avocado");    // 指定位置插入
        System.out.println(list);  // [Apple, Avocado, Banana]

        // 读取
        String fruit = list.get(1);      // "Avocado"
        int idx = list.indexOf("Banana"); // 2
        int lastIdx = list.lastIndexOf("Apple"); // 0

        // 修改
        list.set(0, "Apricot");    // 替换

        // 删除
        list.remove(0);            // 按索引
        list.remove("Banana");     // 按对象 (删除第一个匹配)
        // list.clear();           // 清空

        // 判断
        boolean has = list.contains("Apple");
        boolean empty = list.isEmpty();
        int size = list.size();

        // ========== 范围操作 ==========
        List nums = new ArrayList<>(List.of(0, 1, 2, 3, 4, 5));

        // subList (视图, 修改会影响原列表!)
        List sub = nums.subList(2, 5);  // [2, 3, 4]
        sub.set(0, 99);
        System.out.println(nums.get(2));  // 99 (原列表也被改!)

        // subList 转独立 List
        List independent = new ArrayList<>(nums.subList(2, 5));
    }
}
```


## ArrayList


```
// ========== ArrayList ==========
// 底层: Object[] 数组
// 容量: 默认 10, 扩容 1.5 倍
// 特点: 随机访问 O(1), 插入/删除 O(n)

public class ArrayListDemo {
    public static void main(String[] args) {
        // ========== 构造 ==========
        ArrayList list1 = new ArrayList<>();           // 默认容量 10
        ArrayList list2 = new ArrayList<>(1000);       // 指定初始容量
        ArrayList list3 = new ArrayList<>(list2);      // 从其他集合

        // ========== 扩容机制 ==========
        // 1. add 时检查容量
        // 2. 不够则扩容: new = old + (old >> 1)  (≈1.5 倍)
        // 3. 复制旧数组到新数组 (Arrays.copyOf)
        // 4. 大量添加时可 ensureCapacity 预分配

        ArrayList list = new ArrayList<>();
        list.ensureCapacity(10000);  // 预分配, 减少扩容次数
        list.trimToSize();          // 缩容到当前大小

        // ========== 批量操作 ==========
        List a = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        List b = new ArrayList<>(List.of(4, 5, 6, 7));

        a.addAll(b);                     // [1,2,3,4,5,4,5,6,7]
        a.removeAll(List.of(1, 2));      // [3,4,5,4,5,6,7]
        a.retainAll(List.of(4, 5));      // 只保留交集 [4,5,4,5]
        a.removeIf(n -> n > 4);          // 条件删除 [4,4]

        // ========== 性能 ==========
        // add(E)        O(1) 摊销
        // add(i, E)     O(n)  (移动元素)
        // get(i)        O(1)  (快速!)
        // set(i, E)     O(1)
        // remove(i)     O(n)  (移动)
        // remove(E)     O(n)  (先查找, 再删除)
        // contains(E)   O(n)
        // indexOf(E)    O(n)
        // size()        O(1)
    }
}
```


## LinkedList


```
// ========== LinkedList ==========
// 底层: 双向链表 (Node)
// 特点: 插入/删除 O(1), 随机访问 O(n)
// 同时实现 List, Deque, Queue

public class LinkedListDemo {
    public static void main(String[] args) {
        // ========== 构造 ==========
        LinkedList list = new LinkedList<>();
        LinkedList fromColl = new LinkedList<>(List.of("a", "b"));

        // ========== List 操作 ==========
        list.add("B");
        list.addFirst("A");    // 头部插入
        list.addLast("C");     // 尾部插入
        System.out.println(list);  // [A, B, C]

        String first = list.getFirst();  // "A"
        String last = list.getLast();    // "C"
        String removed = list.removeFirst(); // "A"

        // ========== Queue 操作 ==========
        list.offer("D");          // 尾部插入
        String head = list.peek(); // 读取头部 (不移除)
        String poll = list.poll(); // 读取并移除头部

        // ========== Deque 操作 ==========
        list.push("Top");         // 头部插入 (栈)
        String pop = list.pop();  // 头部移除 (栈)

        // ========== 作为栈使用 ==========
        LinkedList stack = new LinkedList<>();
        stack.push("First");
        stack.push("Second");
        stack.push("Third");
        while (!stack.isEmpty()) {
            System.out.println(stack.pop());  // Third, Second, First
        }

        // ========== 性能 ==========
        // add(E)/addLast(E)   O(1)
        // addFirst(E)         O(1)
        // get(i)              O(n)  (慢!)
        // getFirst/getLast    O(1)
        // remove(i)           O(n)  (先找到位置)
        // removeFirst/Last    O(1)
        // contains(E)         O(n)
    }
}

// ========== ArrayList vs LinkedList 对比 ==========
// ┌─────────────┬─────────────┬──────────────┐
// │ 操作        │ ArrayList   │ LinkedList   │
// ├─────────────┼─────────────┼──────────────┤
// │ get(i)      │ O(1)  ★     │ O(n)         │
// │ add(E)      │ O(1) 摊销   │ O(1)         │
// │ add(i, E)   │ O(n)        │ O(1)  ★      │
// │ remove(i)   │ O(n)        │ O(n)         │
// │ 内存        │ 少 (数组)   │ 多 (Node对象)│
// │ 适用场景    │ 随机访问为主 │ 频繁头尾插入 │
// └─────────────┴─────────────┴──────────────┘

// 结论: 90% 场景用 ArrayList
```


## List 高级用法


```
// ========== List 迭代与遍历 ==========
import java.util.*;

public class ListIterate {
    public static void main(String[] args) {
        List list = new ArrayList<>(List.of("A", "B", "C", "D", "E"));

        // ========== 方式汇总 ==========
        // 1. for 循环 (有索引)
        for (int i = 0; i < list.size(); i++) {
            System.out.println(list.get(i));
        }

        // 2. 增强 for
        for (String s : list) {
            System.out.println(s);
        }

        // 3. Iterator
        Iterator it = list.iterator();
        while (it.hasNext()) {
            System.out.println(it.next());
        }

        // 4. ListIterator (双向迭代)
        ListIterator li = list.listIterator();
        while (li.hasNext()) {
            int idx = li.nextIndex();
            String s = li.next();
            if (s.equals("C")) {
                li.set("CCC");     // 修改当前元素
                li.add("NEW");     // 在当前位置插入
            }
        }
        System.out.println(list);  // [A, B, CCC, NEW, D, E]

        // 5. forEach + Lambda
        list.forEach(System.out::println);

        // 6. Stream
        list.stream()
            .filter(s -> s.length() > 1)
            .forEach(System.out::println);

        // ========== List 排序 ==========
        List nums = new ArrayList<>(List.of(5, 3, 1, 4, 2));

        Collections.sort(nums);                 // 自然排序 [1,2,3,4,5]
        Collections.sort(nums, Collections.reverseOrder()); // 逆序
        nums.sort(Comparator.naturalOrder());   // List.sort (Java 8+)
        nums.sort(Comparator.reverseOrder());

        // 自定义排序
        List words = List.of("banana", "apple", "cherry", "date");
        words.sort(Comparator.comparingInt(String::length)); // 按长度

        // ========== 线程安全 ==========
        // ArrayList 非线程安全
        // 方案1: Collections.synchronizedList
        List syncList = Collections.synchronizedList(new ArrayList<>());

        // 方案2: CopyOnWriteArrayList (读多写少场景)
        List cowList = new CopyOnWriteArrayList<>();

        // 方案3: 自己加锁 (synchronized 块)
    }
}

// ========== Vector & Stack (历史遗留) ==========
// Vector: ArrayList 的线程安全版本 (方法 synchronized)
// Stack: 继承 Vector 的栈实现
// 性能差, 不推荐使用!
// 替代: ArrayList (非并发), ArrayDeque (栈/队列)
//      CopyOnWriteArrayList (并发)
```


> **Note:** 💡 List 要点: ArrayList 数组实现, 随机访问快 O(1); LinkedList 链表实现, 头尾操作快 O(1); ArrayList 扩容 1.5 倍; subList 是视图, 修改影响原列表; ListIterator 双向迭代; Vector/Stack 已过时; 排序用 sort / Comparator; 线程安全用 CopyOnWriteArrayList 或 Collections.synchronizedList; 优先用 ArrayList。


## 练习


<!-- Converted from: 24_Java List 集合.html -->
