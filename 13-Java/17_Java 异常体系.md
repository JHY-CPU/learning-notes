# Java 异常体系


## ⚠️ Java 异常体系


异常层次 (Throwable/Error/Exception/RuntimeException)、受检异常 vs 非受检异常、try-catch-finally、try-with-resources、自定义异常、异常链。


## 异常层次结构


```
// ========== Java 异常层次 ==========
//
//                     Throwable
//                    /         \
//               Error         Exception
//              (不可处理)      (可处理)
//              /    \          /        \
//      OutOfMemoryError  RuntimeException  IOException, SQLException...
//      StackOverflowError  (非受检)           (受检异常)
//      ...
//      NullPointerException
//      ArrayIndexOutOfBoundsException
//      IllegalArgumentException
//      ClassCastException
//      ArithmeticException

// ========== 三种类型 ==========
// 1. Error: JVM 严重错误, 不可处理
// 2. 受检异常 (Checked): 编译时检查, 必须处理
// 3. 非受检异常 (Unchecked/Runtime): 运行时检查, 可选处理

public class ExceptionHierarchy {
    public static void main(String[] args) {
        // ========== 1. Error (不可恢复) ==========
        // StackOverflowError
        // OutOfMemoryError
        // 不应该捕获!

        // ========== 2. 受检异常 (必须处理) ==========
        // IOException, SQLException, ClassNotFoundException
        // 编译时检查: 必须 try-catch 或 throws

        // ========== 3. 非受检异常 (可选处理) ==========
        // RuntimeException 的子类
        // NullPointerException
        // ArrayIndexOutOfBoundsException
        // ArithmeticException (除零)
        // IllegalArgumentException
        // NumberFormatException
    }
}
```


## try-catch-finally


```
// ========== try-catch-finally ==========

public class TryCatchDemo {

    public static void main(String[] args) {
        // ========== 基本 try-catch ==========
        try {
            int result = 10 / 0;  // ArithmeticException
            System.out.println(result);  // 不执行
        } catch (ArithmeticException e) {
            System.out.println("Cannot divide by zero!");
            System.out.println(e.getMessage());  // "/ by zero"
        }
        System.out.println("Continuing...");  // 正常执行

        // ========== 多 catch 块 ==========
        try {
            String[] arr = new String[3];
            arr[5] = "hello";  // ArrayIndexOutOfBoundsException
        } catch (NullPointerException e) {
            System.out.println("Null pointer");
        } catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("Array index out of bounds");
        } catch (Exception e) {
            // 兜底: 必须放在最后!
            System.out.println("General error: " + e.getMessage());
        }

        // ========== | 合并异常 (Java 7+) ==========
        try {
            int[] arr = new int[3];
            arr[5] = 10;
        } catch (ArrayIndexOutOfBoundsException | NullPointerException e) {
            System.out.println("Array error: " + e.getClass().getSimpleName());
            // e 是 final (不能重新赋值)
        }

        // ========== finally ==========
        // 不管是否异常, 都执行
        // 通常用于释放资源 (关闭文件/连接)
        try {
            System.out.println("Try");
            // int x = 1/0;  // 取消注释测试
        } catch (Exception e) {
            System.out.println("Catch");
        } finally {
            System.out.println("Finally (always runs)");
        }
    }

    // ========== finally 陷阱 ==========
    static int finallyTrap() {
        try {
            return 1;
        } finally {
            System.out.println("Finally runs before return");
            // return 2;  // ❌ 会覆盖 try 中的 return!
        }
    }

    // ========== finally 与资源 ==========
    static void readFile() {
        java.io.BufferedReader reader = null;
        try {
            reader = new java.io.BufferedReader(
                new java.io.FileReader("test.txt")
            );
            String line = reader.readLine();
            System.out.println(line);
        } catch (java.io.IOException e) {
            System.out.println("IO Error: " + e.getMessage());
        } finally {
            // 手动关闭资源
            if (reader != null) {
                try {
                    reader.close();  // close 也抛异常!
                } catch (java.io.IOException e) {
                    System.out.println("Close error");
                }
            }
        }
    }
}
```


## try-with-resources (Java 7+)


```
// ========== try-with-resources ==========
// 自动关闭实现了 AutoCloseable 的资源
// 不需要 finally 手动 close

import java.io.*;

public class TryWithResources {

    // ========== 传统方式 (啰嗦) ==========
    static void oldWay() {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader("test.txt"));
            System.out.println(reader.readLine());
        } catch (IOException e) {
            System.out.println("Error: " + e.getMessage());
        } finally {
            if (reader != null) {
                try { reader.close(); } catch (IOException e) { }
            }
        }
    }

    // ========== try-with-resources (简洁) ==========
    static void newWay() {
        // 在 try () 中声明资源
        // 资源必须实现 AutoCloseable
        try (BufferedReader reader =
                new BufferedReader(new FileReader("test.txt"))) {
            System.out.println(reader.readLine());
        } catch (IOException e) {
            System.out.println("Error: " + e.getMessage());
        }
        // reader 自动关闭! (即使发生异常)
    }

    // ========== 多个资源 ==========
    static void multipleResources() {
        try (FileInputStream input = new FileInputStream("in.txt");
             FileOutputStream output = new FileOutputStream("out.txt")) {

            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = input.read(buffer)) != -1) {
                output.write(buffer, 0, bytesRead);
            }
        } catch (IOException e) {
            System.out.println("Copy error: " + e.getMessage());
        }
        // input 和 output 都会自动关闭 (逆序)
    }

    // ========== 自定义 AutoCloseable ==========
    static class MyResource implements AutoCloseable {
        String name;
        MyResource(String name) { this.name = name; }

        @Override
        public void close() {
            System.out.println("Closing: " + name);
        }
    }

    static void customResource() {
        try (MyResource r1 = new MyResource("DB");
             MyResource r2 = new MyResource("File")) {
            System.out.println("Using resources");
        }
        // 输出:
        // Using resources
        // Closing: File   (逆序!)
        // Closing: DB
    }
}
```


## throw 与 throws 与自定义异常


```
// ========== throw ==========
// 手动抛出异常

public class ThrowThrowsDemo {

    static void validateAge(int age) {
        if (age < 0) {
            throw new IllegalArgumentException("Age must be >= 0");
        }
        if (age > 150) {
            throw new IllegalArgumentException("Age must be <= 150");
        }
        System.out.println("Age " + age + " is valid");
    }

    // ========== throws ==========
    // 声明方法可能抛出的受检异常
    // 调用方必须处理 (try-catch 或继续 throws)

    static void readConfig(String path) throws IOException {
        if (!new File(path).exists()) {
            throw new FileNotFoundException("Config not found: " + path);
        }
        // ...
    }

    // 调用方处理
    static void loadConfig() {
        try {
            readConfig("app.properties");
        } catch (IOException e) {
            System.out.println("Failed to load config: " + e.getMessage());
        }
    }

    // 或继续抛出
    static void initApp() throws IOException {
        readConfig("app.properties");
    }

    // ========== 自定义异常 ==========
    // 继承 Exception → 受检异常
    // 继承 RuntimeException → 非受检异常

    // 自定义受检异常
    static class InsufficientFundsException extends Exception {
        public InsufficientFundsException(String message) {
            super(message);
        }

        public InsufficientFundsException(String message, double deficit) {
            super(String.format("%s (deficit: $%.2f)", message, deficit));
        }
    }

    // 自定义非受检异常
    static class InvalidTransactionException extends RuntimeException {
        private final String transactionId;

        public InvalidTransactionException(String transactionId, String reason) {
            super("Transaction " + transactionId + " invalid: " + reason);
            this.transactionId = transactionId;
        }

        public String getTransactionId() {
            return transactionId;
        }
    }

    // ========== 使用自定义异常 ==========
    static class BankAccount {
        private double balance;

        void withdraw(double amount) throws InsufficientFundsException {
            if (amount > balance) {
                double deficit = amount - balance;
                throw new InsufficientFundsException(
                    "Cannot withdraw $" + amount, deficit
                );
            }
            balance -= amount;
        }

        void transfer(String txId, double amount) {
            if (amount <= 0) {
                throw new InvalidTransactionException(txId, "Amount must be positive");
            }
            // ...
        }
    }
}
```


## 异常链与最佳实践


```
// ========== 异常链 (Exception Chaining) ==========
// 保留原始异常信息

public class ExceptionChain {
    static class DataAccessException extends Exception {
        public DataAccessException(String message, Throwable cause) {
            super(message, cause);  // 保留 cause
        }
    }

    static void readData() throws DataAccessException {
        try {
            new java.io.FileReader("data.txt");
        } catch (FileNotFoundException e) {
            throw new DataAccessException("Cannot read data file", e);
        }
    }

    static void process() {
        try {
            readData();
        } catch (DataAccessException e) {
            System.out.println("Error: " + e.getMessage());
            System.out.println("Cause: " + e.getCause().getMessage());
            e.printStackTrace();  // 打印完整调用栈
        }
    }

    // ========== try-catch 性能 ==========
    // try 没开销 (JVM 优化)
    // 异常对象创建有开销 (填充栈轨迹)
    // 不要用异常控制正常流程!

    // ❌ 不要这样:
    // try {
    //     int i = Integer.parseInt(s);
    // } catch (NumberFormatException e) {
    //     // 做其他处理
    // }

    // ✅ 用条件判断:
    // if (s != null && s.matches("\\d+")) {
    //     int i = Integer.parseInt(s);
    // } else { ... }

    // ========== 异常最佳实践 ==========
    // 1. 捕获特定异常, 不捕获 Exception/Throwable
    // 2. 记录异常日志, 不吞异常
    // 3. 抛异常时包含有意义的 message
    // 4. 保持异常链 (cause)
    // 5. 用 try-with-resources 管理资源
    // 6. 不要用异常控制流程
    // 7. 清理资源在 finally 或 try-with-resources
    // 8. 自定义异常用 Exception/RuntimeException 作父类

    // ========== 日志记录 (替代 printStackTrace) ==========
    // 实际项目不要用 e.printStackTrace()
    // 使用 SLF4J + Logback:
    // logger.error("Error processing request: {}", requestId, e);
}
```


> **Note:** 💡 异常要点: Throwable → Error (不可处理) / Exception (可处理); 受检异常必须 try-catch/throws; 非受检异常 (RuntimeException) 可选处理; try-with-resources 自动关闭 AutoCloseable; throw 手动抛, throws 声明; 自定义异常继承 Exception 或 RuntimeException; 保持异常链; 不要用异常控制流程; 优先用日志框架。


## 练习


<!-- Converted from: 17_Java 异常体系.html -->
