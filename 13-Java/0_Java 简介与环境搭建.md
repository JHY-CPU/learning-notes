# Java 简介与环境搭建


## ☕ Java 简介与环境搭建


Java 发展史、JVM/JRE/JDK 区别、安装配置、第一个 Java 程序、编译执行流程、IDE (IntelliJ IDEA/Eclipse)、main 方法签名、包声明。


## Java 概述


```
// ========== Java ==========
// 1995 年由 Sun Microsystems 发布
// 2010 年 Oracle 收购 Sun
// 全球最流行的企业级开发语言

// ========== 核心特点 ==========
// 1. 跨平台 (Write Once, Run Anywhere)
// 2. 面向对象 (封装/继承/多态)
// 3. 自动内存管理 (GC)
// 4. 强类型静态语言
// 5. 丰富的标准库
// 6. 庞大的生态系统

// ========== JVM / JRE / JDK ==========
// ┌─────────────────────────────┐
// │          JDK               │  Java Development Kit
// │  ┌───────────────────────┐  │
// │  │        JRE            │  │  Java Runtime Environment
// │  │  ┌─────────────────┐  │  │
// │  │  │      JVM        │  │  │  Java Virtual Machine
// │  │  │  (字节码执行)    │  │  │
// │  │  └─────────────────┘  │  │
// │  │  + 核心类库           │  │
// │  └───────────────────────┘  │
// │  + 开发工具 (javac/jar)     │
// └─────────────────────────────┘

// JDK = JRE + 开发工具 (javac, jar, javadoc)
// JRE = JVM + 核心类库
// JVM = 字节码执行引擎

// ========== 版本历史 ==========
// Java 8  (2014) — Lambda, Stream (最大多数)
// Java 11 (2018) — LTS, HTTP Client
// Java 17 (2021) — LTS, 密封类, 模式匹配
// Java 21 (2023) — LTS, 虚拟线程, Record Pattern

// 当前推荐: Java 17 或 Java 21 (LTS)
```


## 安装与配置


```
// ========== 安装 Java ==========

// 1. 下载 JDK
// https://adoptium.net/ (Eclipse Temurin, 推荐)
// 或 https://www.oracle.com/java/technologies/downloads/

// 2. 安装后验证:
// java -version
// javac -version

// ========== 环境变量 ==========
// Windows:
// JAVA_HOME = C:\Program Files\Eclipse Adoptium\jdk-17.0.9
// PATH      = %JAVA_HOME%\bin

// macOS/Linux (~/.zshrc):
// export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
// export PATH=$JAVA_HOME/bin:$PATH

// ========== 第一个程序 ==========
// HelloWorld.java:

public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, Java!");
    }
}

// ========== 编译执行 ==========
// 1. 编译: javac HelloWorld.java
//    → 生成 HelloWorld.class (字节码)
//
// 2. 执行: java HelloWorld
//    → JVM 加载 .class 并执行
//
// 注意: java 命令不带 .class 后缀!

// ========== main 方法签名 ==========
// public:    JVM 可访问
// static:    无需实例化
// void:      不返回值
// main:      固定的方法名
// String[]:  命令行参数

// public class Main {
//     public static void main(String[] args) {
//         for (String arg : args) {
//             System.out.println(arg);
//         }
//     }
// }
// 运行: java Main hello world
```


## IDE 与项目结构


```
// ========== IDE 选择 ==========

// IntelliJ IDEA (推荐)
// - Community (免费) / Ultimate (付费)
// - 最好的 Java IDE
// - 智能提示, 重构, 调试

// Eclipse
// - 开源免费
// - 老牌 IDE
// - 插件丰富

// VS Code
// - 轻量
// - Extension Pack for Java
// - 适合学习和简单项目

// ========== Java 项目结构 ==========
// my-app/
// ├── src/
// │   ├── main/
// │   │   ├── java/           # Java 源码
// │   │   │   └── com/
// │   │   │       └── myapp/
// │   │   │           ├── Main.java
// │   │   │           ├── controller/
// │   │   │           ├── service/
// │   │   │           └── model/
// │   │   └── resources/       # 配置文件
// │   │       ├── application.properties
// │   │       └── static/
// │   └── test/
// │       └── java/            # 测试代码
// ├── target/                  # 编译输出
// ├── pom.xml                  # Maven (依赖管理)
// └── build.gradle             # Gradle (替代)

// ========== 包声明 ==========
// 包: 命名空间, 避免类名冲突
// 命名规范: 域名倒写

package com.myapp.controller;

import com.myapp.service.UserService;

public class UserController {
    // ...
}

// 编译后的目录结构自动匹配包名
// src/com/myapp/controller/UserController.java
```


## 基本语法入门


```
// ========== Java 基本语法 ==========

public class SyntaxDemo {

    // main 方法入口
    public static void main(String[] args) {
        // ========== 变量 ==========
        int age = 25;                    // 整数
        double price = 19.99;            // 浮点
        boolean isActive = true;         // 布尔
        char grade = 'A';                // 字符
        String name = "Alice";           // 字符串 (不是基本类型)

        // ========== 常量 ==========
        final int MAX_USERS = 1000;       // 不可修改

        // ========== 基本输出 ==========
        System.out.println("Hello");      // 输出 + 换行
        System.out.print("World");        // 输出 (不换行)
        System.out.printf("Age: %d", age); // 格式化

        // ========== 注释 ==========
        // 单行注释

        /*
         * 多行注释
         */

        /**
         * Javadoc 文档注释
         * @param args 命令行参数
         */

        // ========== 输入 ==========
        // Scanner scanner = new Scanner(System.in);
        // System.out.print("Enter name: ");
        // String input = scanner.nextLine();
        // int number = scanner.nextInt();
        // scanner.close();
    }
}

// ========== Java 与 JavaScript 关键区别 ==========
// Java:                JavaScript:
// 静态类型             动态类型
// int x = 5;           let x = 5;
// 编译型 (javac)       解释型
// 类必须与文件名一致   自由命名
// main 方法入口        全局代码执行
// JVM 运行             Node.js / 浏览器
// 多线程               事件循环
```


> **Note:** 💡 Java 要点: JVM 跨平台; JDK = JRE + 开发工具; main 方法是程序入口; javac 编译 → java 运行; 包名域名倒写; 静态强类型; 推荐 Java 17 LTS; IntelliJ IDEA 最佳 IDE; System.out.println 输出; Scanner 输入。


## 练习


<!-- Converted from: 0_Java 简介与环境搭建.html -->
