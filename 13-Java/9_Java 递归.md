# Java 递归


## 🔄 Java 递归


递归原理 (基线条件 + 递归步骤)、调用栈、常见递归 (阶乘/斐波那契/汉诺塔/回文)、尾递归、递归 vs 迭代、递归优化 (记忆化)。


## 递归基础


```
// ========== 递归 ==========
// 方法调用自身
// 两个要素:
//   1. 基线条件 (base case) — 停止递归
//   2. 递归步骤 (recursive case) — 向基线推进

public class RecursionBasics {

    // ========== 阶乘 ==========
    // n! = n × (n-1) × ... × 1
    static int factorial(int n) {
        // 基线条件: n <= 1
        if (n <= 1) return 1;

        // 递归步骤: n × factorial(n-1)
        return n * factorial(n - 1);
    }
    // factorial(4) 调用栈:
    // factorial(4) = 4 * factorial(3)
    //   factorial(3) = 3 * factorial(2)
    //     factorial(2) = 2 * factorial(1)
    //       factorial(1) = 1        ← 基线
    //     factorial(2) = 2 * 1 = 2
    //   factorial(3) = 3 * 2 = 6
    // factorial(4) = 4 * 6 = 24

    // ========== 递归 vs 迭代 ==========
    static int factorialIter(int n) {
        int result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }

    // ========== 递归打印 ==========
    static void countDown(int n) {
        if (n <= 0) {
            System.out.println("Go!");
            return;
        }
        System.out.println(n);
        countDown(n - 1);
    }
    // 输出: 5 4 3 2 1 Go!

    static void countUp(int n) {
        if (n <= 0) return;
        countUp(n - 1);        // 先递归
        System.out.println(n); // 回溯时打印
    }
    // countUp(5): 1 2 3 4 5

    public static void main(String[] args) {
        System.out.println(factorial(5));     // 120
        System.out.println(factorialIter(5)); // 120
        countDown(5);
        countUp(5);
    }
}
```


## 经典递归


```
// ========== 斐波那契数列 ==========
// F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2)

public class ClassicRecursion {

    // ========== 朴素递归 (指数级 O(2ⁿ)) ==========
    static long fib(int n) {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    }
    // 大量重复计算! fib(5):
    // fib(5) = fib(4) + fib(3)
    //   fib(4) = fib(3) + fib(2)  ← fib(3) 重复!
    //   fib(3) = fib(2) + fib(1)  ← fib(2) 重复!

    // ========== 记忆化递归 (O(n)) ==========
    static long fibMemo(int n, long[] memo) {
        if (n <= 1) return n;
        if (memo[n] != 0) return memo[n];
        memo[n] = fibMemo(n - 1, memo) + fibMemo(n - 2, memo);
        return memo[n];
    }
    static long fibMemo(int n) {
        return fibMemo(n, new long[n + 1]);
    }

    // ========== 迭代版 ==========
    static long fibIter(int n) {
        if (n <= 1) return n;
        long a = 0, b = 1;
        for (int i = 2; i <= n; i++) {
            long c = a + b;
            a = b;
            b = c;
        }
        return b;
    }

    // ========== 汉诺塔 ==========
    // 将 n 个盘从 A 移到 C, 借助 B
    // 规则: 每次移1盘, 大不能压小
    static void hanoi(int n, char from, char to, char aux) {
        if (n == 1) {
            System.out.println(from + " → " + to);
            return;
        }
        // 1. 将 n-1 个盘从 from 移到 aux (借助 to)
        hanoi(n - 1, from, aux, to);
        // 2. 将最大盘从 from 移到 to
        System.out.println(from + " → " + to);
        // 3. 将 n-1 个盘从 aux 移到 to (借助 from)
        hanoi(n - 1, aux, to, from);
    }
    // hanoi(3, 'A', 'C', 'B'):
    // A→C, A→B, C→B, A→C, B→A, B→C, A→C

    // ========== 回文判断 ==========
    static boolean isPalindrome(String s) {
        if (s.length() <= 1) return true;
        if (s.charAt(0) != s.charAt(s.length() - 1)) return false;
        return isPalindrome(s.substring(1, s.length() - 1));
    }
    // isPalindrome("racecar") → true
    // isPalindrome("hello") → false

    public static void main(String[] args) {
        System.out.println(fib(10));       // 55
        System.out.println(fibMemo(100));  // 354224848179261915075
        System.out.println(fibIter(100));

        hanoi(3, 'A', 'C', 'B');

        System.out.println(isPalindrome("racecar")); // true
        System.out.println(isPalindrome("abcba"));   // true
    }
}
```


## 调用栈与栈溢出


```
// ========== 调用栈 (Call Stack) ==========
// 每次方法调用在栈上分配栈帧 (stack frame)
// 栈帧包含: 局部变量, 参数, 返回地址
// 递归深度过大 → StackOverflowError

public class StackDemo {

    // 无终止递归 → 栈溢出
    static void infiniteRecursion(int n) {
        System.out.println(n);
        infiniteRecursion(n + 1);  // 永不停止!
    }
    // 抛出 StackOverflowError

    // ========== 尾递归 ==========
    // 尾递归: 递归调用是方法的最后一个操作
    // Java 不支持尾递归优化 (TCO)
    // 但了解概念有助于理解递归

    // 普通递归 (有乘法待处理)
    static int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);  // 乘法在递归返回后
    }

    // 尾递归形式 (累加器参数)
    static int factorialTail(int n, int acc) {
        if (n <= 1) return acc;
        return factorialTail(n - 1, n * acc);  // 最后一步是递归
    }
    // 调用: factorialTail(5, 1)

    // ========== 递归深度限制 ==========
    // 默认栈大小 ~1MB (取决于 JVM 配置)
    // 递归深度通常 10000-30000 (int 参数)
    // 更多参数/局部变量 → 深度更小

    // 查看栈深度:
    static int depth = 0;
    static void measureDepth() {
        depth++;
        try {
            measureDepth();
        } catch (StackOverflowError e) {
            System.out.println("Max depth: " + depth);
        }
    }

    // ========== 递归优化思路 ==========
    // 1. 记忆化 (Memoization) — 缓存中间结果
    // 2. 尾递归形式 — Java 虽不优化, 但易转迭代
    // 3. 转迭代 — 手动模拟栈
    // 4. 增大栈空间: -Xss2m
}
```


## 递归实战应用


```
// ========== 递归实战模式 ==========
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class RecursionPatterns {

    // ========== 1. 树形结构遍历 ==========
    // 文件系统遍历
    static void listFiles(File dir, String indent) {
        File[] files = dir.listFiles();
        if (files == null) return;

        for (File file : files) {
            System.out.println(indent + (file.isDirectory() ? "📁 " : "📄 ") + file.getName());
            if (file.isDirectory()) {
                listFiles(file, indent + "  ");
            }
        }
    }

    // ========== 2. 链表操作 (递归) ==========
    static class Node {
        int value;
        Node next;
        Node(int v) { this.value = v; }
    }

    // 递归反转链表
    static Node reverse(Node head) {
        if (head == null || head.next == null) return head;
        Node newHead = reverse(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }

    // 递归打印链表
    static void printList(Node head) {
        if (head == null) return;
        System.out.print(head.value + " → ");
        printList(head.next);
    }

    // ========== 3. 分治模式 (Divide & Conquer) ==========
    // 二分查找 (递归版)
    static int binarySearch(int[] arr, int target, int lo, int hi) {
        if (lo > hi) return -1;
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] > target) return binarySearch(arr, target, lo, mid - 1);
        return binarySearch(arr, target, mid + 1, hi);
    }

    // ========== 4. 回溯 (Backtracking) ==========
    // N 皇后: 每行放一个皇后, 不能同行同列同对角线
    static void solveNQueens(int n) {
        int[] board = new int[n];  // board[row] = col
        placeQueen(board, 0, n);
    }

    static void placeQueen(int[] board, int row, int n) {
        if (row == n) {
            printBoard(board);
            return;
        }
        for (int col = 0; col < n; col++) {
            if (isSafe(board, row, col)) {
                board[row] = col;           // 放置
                placeQueen(board, row + 1, n); // 递归下一行
                // 回溯: 自动覆盖 board[row]
            }
        }
    }

    static boolean isSafe(int[] board, int row, int col) {
        for (int i = 0; i < row; i++) {
            if (board[i] == col) return false;                 // 同列
            if (Math.abs(board[i] - col) == Math.abs(i - row)) return false; // 对角线
        }
        return true;
    }

    static void printBoard(int[] board) {
        for (int row = 0; row < board.length; row++) {
            for (int col = 0; col < board.length; col++) {
                System.out.print(board[row] == col ? "Q " : ". ");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void main(String[] args) {
        // 分治: 二分查找
        int[] sorted = {1, 3, 5, 7, 9, 11, 13};
        int idx = binarySearch(sorted, 7, 0, sorted.length - 1);
        System.out.println("Found at: " + idx);  // 3

        // 回溯: 4 皇后
        solveNQueens(4);
    }
}
```


> **Note:** 💡 递归要点: 必须先有基线条件; 每次递归向基线推进; 调用栈有限防 StackOverflow; 斐波那契朴素递归 O(2ⁿ) 需记忆化 → O(n); 尾递归 Java 不优化但易转迭代; 递归适合树/图/分治/回溯; 实战用到文件遍历/链表反转/二分查找/N皇后; 能用迭代尽量用迭代 (性能更好)。


## 练习


<!-- Converted from: 9_Java 递归.html -->
