# Java Queue 与 Deque


## ⏳ Java Queue 与 Deque


Queue 接口 (offer/poll/peek)、Deque 接口 (双端队列)、ArrayDeque、PriorityQueue (优先级队列)、BlockingQueue、Queue 在 BFS 中的应用。


## Queue 接口


```
// ========== Queue (队列) ==========
// FIFO: First-In-First-Out
// 不能在中间插入

import java.util.*;

public class QueueBasics {
    public static void main(String[] args) {
        // ========== Queue 方法 ==========
        // ┌──────────┬──────────────┬──────────────┐
        // │ 操作      │ 失败抛异常   │ 失败返特殊值 │
        // ├──────────┼──────────────┼──────────────┤
        // │ 插入      │ add(e)       │ offer(e)     │
        // │ 移除      │ remove()     │ poll()       │
        // │ 检查      │ element()    │ peek()       │
        // └──────────┴──────────────┴──────────────┘

        // ========== LinkedList 作为 Queue ==========
        Queue queue = new LinkedList<>();

        // offer: 添加 (返回 false 如果容量满)
        queue.offer("First");
        queue.offer("Second");
        queue.offer("Third");

        // peek: 查看头部 (不移除)
        System.out.println(queue.peek());  // "First"

        // poll: 取出并移除头部
        System.out.println(queue.poll());  // "First"
        System.out.println(queue.poll());  // "Second"
        System.out.println(queue.poll());  // "Third"
        System.out.println(queue.poll());  // null (空队列)

        // ========== 其他 Queue 实现 ==========
        // PriorityQueue: 优先级队列 (见下文)
        // ArrayDeque:    双端队列 (见下文)
        // DelayQueue:    延迟队列 (元素到期才可取出)
        // LinkedBlockingQueue: 线程安全阻塞队列

        // ========== 队列遍历 ==========
        Queue q = new LinkedList<>(List.of("A", "B", "C"));
        // for-each 安全 (不会移除元素)
        for (String s : q) {
            System.out.println(s);
        }
        // 逐个取出:
        while (!q.isEmpty()) {
            System.out.println(q.poll());
        }
    }
}

// ========== Queue 应用: BFS ==========
class BFSExample {
    static void bfs(Map> graph, String start) {
        Queue queue = new LinkedList<>();
        Set visited = new HashSet<>();

        queue.offer(start);
        visited.add(start);

        while (!queue.isEmpty()) {
            String node = queue.poll();
            System.out.println("Visit: " + node);

            for (String neighbor : graph.getOrDefault(node, List.of())) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.offer(neighbor);
                }
            }
        }
    }

    public static void main(String[] args) {
        Map> graph = new HashMap<>();
        graph.put("A", List.of("B", "C"));
        graph.put("B", List.of("A", "D", "E"));
        graph.put("C", List.of("A", "F"));
        graph.put("D", List.of("B"));
        graph.put("E", List.of("B"));
        graph.put("F", List.of("C"));

        bfs(graph, "A");  // A, B, C, D, E, F
    }
}
```


## PriorityQueue


```
// ========== PriorityQueue (优先级队列) ==========
// 底层: 二叉堆 (数组实现)
// 默认: 最小堆 (自然顺序, 小根堆)
// 每次 poll 取出最小/优先级最高的元素
// 非线程安全

public class PriorityQueueDemo {
    public static void main(String[] args) {
        // ========== 最小堆 (自然排序) ==========
        PriorityQueue minHeap = new PriorityQueue<>();
        minHeap.offer(5);
        minHeap.offer(1);
        minHeap.offer(3);
        minHeap.offer(2);
        minHeap.offer(4);

        System.out.println(minHeap.poll());  // 1 (最小)
        System.out.println(minHeap.poll());  // 2
        System.out.println(minHeap.poll());  // 3

        // ========== 最大堆 (逆序) ==========
        PriorityQueue maxHeap = new PriorityQueue<>(
            Comparator.reverseOrder()
        );
        maxHeap.offerAll(List.of(5, 1, 3, 2, 4));
        System.out.println(maxHeap.poll());  // 5 (最大)

        // ========== 自定义优先级 ==========
        // 按字符串长度排序 (短优先)
        PriorityQueue byLength = new PriorityQueue<>(
            Comparator.comparingInt(String::length)
        );
        byLength.offerAll(List.of("banana", "apple", "cherry", "date"));
        System.out.println(byLength.poll());  // "date" (长度4)

        // ========== 对象优先级 ==========
        class Task implements Comparable {
            String name;
            int priority;

            Task(String name, int priority) {
                this.name = name;
                this.priority = priority;
            }

            @Override
            public int compareTo(Task o) {
                return Integer.compare(this.priority, o.priority);
            }

            @Override
            public String toString() {
                return name + "(" + priority + ")";
            }
        }

        PriorityQueue tasks = new PriorityQueue<>();
        tasks.offer(new Task("Low", 3));
        tasks.offer(new Task("High", 1));
        tasks.offer(new Task("Medium", 2));

        while (!tasks.isEmpty()) {
            System.out.println(tasks.poll());  // High, Medium, Low
        }

        // ========== 性能 ==========
        // offer/poll/remove: O(log n)
        // peek/element/size: O(1)
        // 不保证相同优先级元素的顺序
    }
}
```


## Deque (双端队列)


```
// ========== Deque (Double Ended Queue) ==========
// 两端都可插入/删除
// ArrayDeque: 数组实现, 性能优于 Stack/LinkedList
// LinkedList: 也实现 Deque

import java.util.*;

public class DequeDemo {
    public static void main(String[] args) {
        // ========== ArrayDeque ==========
        // 推荐: 比 Stack (同步) 快, 比 LinkedList (节点) 省内存
        Deque deque = new ArrayDeque<>();

        // ========== 双端操作 ==========
        // ┌─────────────┬──────────────┬──────────────┐
        // │ 操作        │ 头部         │ 尾部         │
        // ├─────────────┼──────────────┼──────────────┤
        // │ 插入        │ addFirst/offerFirst │ addLast/offerLast │
        // │ 移除        │ removeFirst/pollFirst│ removeLast/pollLast│
        // │ 检查        │ getFirst/peekFirst  │ getLast/peekLast   │
        // └─────────────┴──────────────┴──────────────┘

        deque.addLast("A");         // 尾部 → [A]
        deque.addLast("B");         // 尾部 → [A, B]
        deque.addFirst("Z");        // 头部 → [Z, A, B]
        System.out.println(deque);  // [Z, A, B]

        System.out.println(deque.peekFirst());  // "Z"
        System.out.println(deque.peekLast());   // "B"
        System.out.println(deque.pollFirst());  // "Z" (移除)
        System.out.println(deque.pollLast());   // "B" (移除)

        // ========== 作为栈 (Stack) ==========
        // push/pop/peek 操作头部 (LIFO)
        Deque stack = new ArrayDeque<>();
        stack.push("First");        // addFirst
        stack.push("Second");       // addFirst
        stack.push("Third");        // addFirst

        System.out.println(stack.peek());  // "Third" (头部)
        while (!stack.isEmpty()) {
            System.out.println(stack.pop());  // Third, Second, First
        }

        // ========== 作为队列 (Queue) ==========
        Deque q = new ArrayDeque<>();
        q.offer("A");              // addLast
        q.offer("B");              // addLast
        q.offer("C");              // addLast

        while (!q.isEmpty()) {
            System.out.println(q.poll());  // A, B, C (FIFO)
        }

        // ========== ArrayDeque 性能 ==========
        // - 无容量限制 (自动扩容)
        // - 初始容量 16, 扩容 2 倍
        // - 所有操作 O(1) 摊销
        // - 比 Stack 快 (无 synchronized)
        // - 比 LinkedList 省内存 (无 Node 对象)
    }
}
```


## BlockingQueue


```
// ========== BlockingQueue ==========
// 线程安全的阻塞队列
// 当队列空: take() 阻塞等待
// 当队列满: put() 阻塞等待
// 用于生产者-消费者模式

import java.util.concurrent.*;

public class BlockingQueueDemo {

    // ========== 常见实现 ==========
    // ArrayBlockingQueue:    有界, 数组实现
    // LinkedBlockingQueue:   可选有界/无界, 链表实现
    // PriorityBlockingQueue: 有优先级, 无界
    // SynchronousQueue:      无容量 (直接传递)
    // DelayQueue:            延迟队列 (元素到期才可取出)

    static class Producer implements Runnable {
        private final BlockingQueue queue;
        private final String name;

        Producer(BlockingQueue queue, String name) {
            this.queue = queue;
            this.name = name;
        }

        @Override
        public void run() {
            try {
                for (int i = 1; i <= 5; i++) {
                    String item = name + "-" + i;
                    queue.put(item);  // 队列满时阻塞
                    System.out.println("Produced: " + item);
                    Thread.sleep(100);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    static class Consumer implements Runnable {
        private final BlockingQueue queue;

        Consumer(BlockingQueue queue) {
            this.queue = queue;
        }

        @Override
        public void run() {
            try {
                while (true) {
                    String item = queue.take();  // 队列空时阻塞
                    System.out.println("Consumed: " + item);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    public static void main(String[] args) {
        BlockingQueue queue = new ArrayBlockingQueue<>(10);

        // 启动生产者和消费者
        new Thread(new Producer(queue, "Task-A")).start();
        new Thread(new Producer(queue, "Task-B")).start();
        new Thread(new Consumer(queue)).start();

        // BlockingQueue 方法对比:
        // ┌──────────┬──────────────┬──────────────┬─────────────┐
        // │          │ 抛异常       │ 返回特殊值   │ 阻塞        │
        // ├──────────┼──────────────┼──────────────┼─────────────┤
        // │ 插入     │ add(e)       │ offer(e)     │ put(e)      │
        // │ 移除     │ remove()     │ poll()       │ take()      │
        // │ 检查     │ element()    │ peek()       │ 无          │
        // └──────────┴──────────────┴──────────────┴─────────────┘
    }
}

// ========== Queue 选择指南 ==========
// FIFO 队列          → LinkedList 或 ArrayDeque
// 优先级处理         → PriorityQueue
// 双端操作/栈        → ArrayDeque
// 并发生产者-消费者  → BlockingQueue
// 延迟任务           → DelayQueue
// 直接传递           → SynchronousQueue
```


> **Note:** 💡 Queue/Deque 要点: Queue (FIFO): offer/poll/peek; PriorityQueue 二叉堆, 默认最小堆, poll 取最小元素 O(log n); Deque 双端操作: addFirst/pollLast 等; ArrayDeque 比 Stack/LinkedList 更优; BlockingQueue put/take 阻塞; 生产者-消费者模式; BFS 用 LinkedList 作队列; 栈用 ArrayDeque。


## 练习


<!-- Converted from: 27_Java Queue 与 Deque.html -->
