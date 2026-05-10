# Java 封装与访问控制


## 🔒 Java 封装与访问控制


封装 (Encapsulation)、访问修饰符 (private/default/protected/public)、getter/setter、JavaBean 规范、package 与 import、包访问控制。


## 封装 (Encapsulation)


```
// ========== 封装 ==========
// 将数据 (字段) 和行为 (方法) 打包在类中
// 隐藏内部实现, 通过公共接口访问
// 好处: 数据安全, 实现解耦, 可维护

// ========== 未封装的类 (问题) ==========
class BadStudent {
    String name;
    int age;
    // 直接暴露字段:
    // 1. 无法验证数据合法性
    // 2. 修改实现影响所有调用方
    // 3. 无法添加额外逻辑 (日志/通知)
}

// 使用:
// BadStudent s = new BadStudent();
// s.age = -5;  // 不合法的年龄! 但能设置

// ========== 封装的类 ==========
class Student {
    // ========== private 字段 ==========
    // 只能在类内部访问
    private String name;
    private int age;
    private String email;
    private double score;

    // ========== 构造器 ==========
    public Student(String name, int age) {
        this.name = name;
        setAge(age);  // 复用验证逻辑
    }

    // ========== public getter/setter ==========
    // getter — 读取
    // setter — 写入 (可添加验证)

    public String getName() {
        return name;
    }

    public void setName(String name) {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Name cannot be empty");
        }
        this.name = name.trim();
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        if (age < 0 || age > 150) {
            throw new IllegalArgumentException("Invalid age: " + age);
        }
        this.age = age;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        // 简单邮箱验证
        if (email != null && !email.contains("@")) {
            throw new IllegalArgumentException("Invalid email");
        }
        this.email = email;
    }

    // ========== 只读属性 (只有 getter) ==========
    public double getScore() {
        return score;
    }

    // 没有 setScore — 通过方法来修改
    public void updateScore(double newScore) {
        if (newScore < 0 || newScore > 100) {
            throw new IllegalArgumentException("Score must be 0-100");
        }
        this.score = newScore;
        System.out.println("Score updated to: " + newScore);
    }

    // ========== 业务方法 (封装行为) ==========
    public boolean isAdult() {
        return age >= 18;
    }

    public String getGrade() {
        if (score >= 90) return "A";
        if (score >= 80) return "B";
        if (score >= 70) return "C";
        if (score >= 60) return "D";
        return "F";
    }

    @Override
    public String toString() {
        return String.format("Student{name='%s', age=%d, grade=%s}",
            name, age, getGrade());
    }
}
```


## 访问修饰符


```
// ========== 4 种访问修饰符 ==========
//
// ┌──────────────┬──────┬──────────┬───────────┬──────┐
// │ 修饰符       │ 同类 │ 同包     │ 子类      │ 全局 │
// ├──────────────┼──────┼──────────┼───────────┼──────┤
// │ private      │ ✅   │ ❌       │ ❌        │ ❌   │
// │ default(无)  │ ✅   │ ✅       │ ❌        │ ❌   │
// │ protected    │ ✅   │ ✅       │ ✅        │ ❌   │
// │ public       │ ✅   │ ✅       │ ✅        │ ✅   │
// └──────────────┴──────┴──────────┴───────────┴──────┘

package com.example.access;  // 包声明

public class AccessModifiers {

    private int privateVar = 1;     // 仅本类
    int defaultVar = 2;             // 本类 + 同包
    protected int protectedVar = 3; // 本类 + 同包 + 子类
    public int publicVar = 4;       // 任意

    void demoAccess() {
        // 同类: 全部可访问
        System.out.println(privateVar);    // ✅
        System.out.println(defaultVar);    // ✅
        System.out.println(protectedVar);  // ✅
        System.out.println(publicVar);     // ✅
    }
}

// ========== 同包中的类 ==========
class SamePackageClass {
    void demo() {
        AccessModifiers am = new AccessModifiers();
        // System.out.println(am.privateVar);    // ❌
        System.out.println(am.defaultVar);        // ✅ 同包
        System.out.println(am.protectedVar);      // ✅ 同包
        System.out.println(am.publicVar);         // ✅ 全局
    }
}

// ========== 设计原则 ==========
// 1. 最小权限原则: 能 private 就 private
// 2. 字段: private (除常量 public static final)
// 3. 方法: 公开的用 public, 辅助的用 private
// 4. protected: 子类需要访问时用
// 5. default (包级): 包内协作时用 (不常用)
```


## package 与 import


```
// ========== package (包) ==========
// 包: 类的命名空间/组织方式
// 域名反转: com.example.project.module
// 对应文件目录: com/example/project/module/

package com.example.school;  // 必须在文件第一行 (除了注释)

// 完整类名: com.example.school.Student
// 简名单: Student (同包内)

// ========== import ==========
// 引入其他包的类

import java.util.Scanner;           // 引入单个类
import java.util.*;                 // 引入所有类 (不推荐, 降低可读性)
import java.util.List;              // 明确引入
import static java.lang.Math.PI;    // 静态导入 (静态成员)
import static java.lang.Math.sqrt;

// 使用静态导入后:
// double r = sqrt(25);  // 不用 Math.sqrt(25)
// double area = PI * r * r;

public class PackageDemo {
    public static void main(String[] args) {
        // 使用 import 的类
        Scanner scanner = new Scanner(System.in);

        // 完整限定名 (不用 import)
        java.util.ArrayList list = new java.util.ArrayList<>();

        // 同包类无需 import
        Student s = new Student("Alice", 20);
    }
}

// ========== 常见 Java 包 ==========
// java.lang        — String, Math, System, Thread (自动导入)
// java.util        — List, Map, Set, Scanner, Collections
// java.io          — File, InputStream, OutputStream
// java.nio.file    — Path, Files (Java 7+)
// java.time        — LocalDate, LocalDateTime (Java 8+)
// java.net         — URL, Socket
// java.sql         — Connection, ResultSet
// javax.swing      — GUI 组件

// ========== 命名规范 ==========
// 包: 全小写, com.example.project
// 类: 大驼峰, StudentManager
// 方法: 小驼峰, getName()
// 变量: 小驼峰, studentName
// 常量: 大写_下划线, MAX_VALUE
}
```


## JavaBean 规范


```
// ========== JavaBean 规范 ==========
// 可重用的 Java 组件规范

public class PersonBean {
    // 1. 类必须是 public
    // 2. 必须有无参构造器

    // 3. 属性 private, 通过 getter/setter 访问
    private String name;
    private int age;
    private boolean active;  // boolean 的 getter: isXxx()

    // 4. 必须实现 Serializable (可选但推荐)
    // implements java.io.Serializable

    // 无参构造器 (重要!)
    public PersonBean() {}

    // getter/setter 命名规则:
    // getXxx() / setXxx()  — 一般属性
    // isXxx() / setXxx()   — boolean 属性

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    // boolean: isXxx() 而不是 getXxx()
    public boolean isActive() {
        return active;
    }

    public void setActive(boolean active) {
        this.active = active;
    }

    // getter/setter 命名违反规范时, 框架可能无法识别!
    // ❌ getN() → 不标准
    // ✅ getName() → 标准
}

// ========== 记录类 (Record, Java 16+) ==========
// 不可变数据载体, 自动生成构造器/getter/equals/hashCode/toString
// record 是 final class, 不能继承

// public record Point(int x, int y) { }
// // 自动生成:
// // - 构造器 Point(int x, int y)
// // - 访问器 x() / y()  (不是 getX() / getY()!)
// // - equals(), hashCode(), toString()

// 使用:
// Point p = new Point(3, 4);
// System.out.println(p.x());  // 3
// System.out.println(p);       // Point[x=3, y=4]

// 可以添加方法:
// public record Rectangle(double width, double height) {
//     public double area() {
//         return width * height;
//     }
// }
```


> **Note:** 💡 封装要点: 字段 private + getter/setter 实现封装; 4 种访问修饰符 (private/default/protected/public) 控制可见性; 最小权限原则; package 组织类, import 引入类; 静态导入 static import; JavaBean 规范 (无参构造 + getter/setter + 序列化); record (Java 16+) 简化不可变数据类。


## 练习


<!-- Converted from: 11_Java 封装与访问控制.html -->
