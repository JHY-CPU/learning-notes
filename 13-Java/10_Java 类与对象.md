# Java 类与对象


## 📦 Java 类与对象


类定义、对象创建 (new)、构造器、this 关键字、实例 vs 静态成员、对象生命周期、垃圾回收、finalize。


## 类定义与对象创建


```
// ========== 类 (Class) 与 对象 (Object) ==========
// 类: 模板/蓝图
// 对象: 类的实例 (实例化)

// ========== 定义一个类 ==========
class Student {
    // ========== 字段 (Fields / 成员变量) ==========
    String name;      // 实例变量 (每个对象独有)
    int age;
    String studentId;

    // ========== 构造器 (Constructor) ==========
    // 与类同名, 无返回类型
    // 用于初始化对象

    // 默认无参构造器 (如果不定义任何构造器, Java 自动提供)
    Student() {
        this.name = "Unknown";
        this.age = 0;
    }

    // 有参构造器
    Student(String name, int age, String studentId) {
        this.name = name;         // this.字段 = 参数
        this.age = age;
        this.studentId = studentId;
    }

    // ========== 方法 ==========
    void introduce() {
        System.out.println("Hi, I'm " + name + ", " + age + " years old");
    }

    boolean isAdult() {
        return age >= 18;
    }

    void birthday() {
        age++;
        System.out.println("Happy birthday! Now " + age);
    }
}

// ========== 使用类 ==========
public class ClassDemo {
    public static void main(String[] args) {
        // ========== 创建对象 (new) ==========
        // new 关键字: 分配堆内存, 调用构造器, 返回引用

        Student s1 = new Student();  // 无参构造器
        s1.name = "Alice";           // 直接访问字段 (不推荐)
        s1.age = 20;

        Student s2 = new Student("Bob", 22, "S2024002");  // 有参构造器

        // ========== 调用方法 ==========
        s1.introduce();  // "Hi, I'm Alice, 20 years old"
        s2.introduce();  // "Hi, I'm Bob, 22 years old"

        System.out.println(s1.isAdult());  // true
        s2.birthday();                     // age: 22 → 23

        // ========== 多个对象 ==========
        Student s3 = s1;  // 引用复制, 指向同一对象
        s3.name = "Charlie";
        System.out.println(s1.name);  // "Charlie" (s1 也被改了!)
    }
}
```


## 构造器详解


```
// ========== 构造器深入 ==========

class ConstructorDemo {
    String name;
    int age;
    String email;

    // ========== 构造器重载 ==========
    ConstructorDemo() {
        this("Guest", 0);  // 用 this() 调用另一个构造器!
    }

    ConstructorDemo(String name) {
        this(name, 0, null);  // 链式调用
    }

    ConstructorDemo(String name, int age) {
        this(name, age, null);
    }

    // 全参构造器 (终极目标)
    ConstructorDemo(String name, int age, String email) {
        this.name = name;
        this.age = age;
        this.email = email;
    }

    // ========== this 关键字 ==========
    // 1. this.字段 — 区分字段和参数
    // 2. this() — 调用本类其他构造器 (必须在第一行)
    // 3. this 引用本身

    void printThis() {
        System.out.println("this = " + this);  // 对象地址
        System.out.println("name = " + this.name);
    }

    // 返回 this (链式调用)
    ConstructorDemo setName(String name) {
        this.name = name;
        return this;
    }

    ConstructorDemo setAge(int age) {
        this.age = age;
        return this;
    }

    void show() {
        System.out.println(name + " (" + age + ")");
    }

    // 使用链式:
    // new ConstructorDemo()
    //     .setName("Alice")
    //     .setAge(25)
    //     .show();
}

// ========== 默认构造器 ==========
class DefaultConstructor {
    // 没有定义构造器时, Java 自动生成:
    // DefaultConstructor() { super(); }
    // 一旦定义了任何构造器, 默认不再生成
}

class NoDefault {
    int x;
    NoDefault(int x) {
        this.x = x;
    }
    // 没有无参构造器!
    // new NoDefault()  // 编译错误!
}
```


## 实例 vs 静态成员


```
// ========== static 关键字 ==========
// 静态成员属于类, 不属于对象

class StaticDemo {
    // ========== 实例变量 ==========
    // 每个对象独有, 通过对象访问
    String instanceVar = "I belong to object";

    // ========== 静态变量 (类变量) ==========
    // 所有对象共享, 通过类名访问
    static int objectCount = 0;
    static String classDesc = "This is a demo class";

    // ========== 实例方法 ==========
    // 需要对象调用
    void instanceMethod() {
        System.out.println("Instance method: " + instanceVar);
        System.out.println("Can access static: " + objectCount);  // ✅
    }

    // ========== 静态方法 (类方法) ==========
    // 通过类名调用
    static void staticMethod() {
        System.out.println("Static method: " + classDesc);
        // System.out.println(instanceVar);  // ❌ 不能访问实例变量!
        // instanceMethod();  // ❌ 不能直接调用实例方法!
    }

    // ========== 静态方法用途 ==========
    static int getObjectCount() {
        return objectCount;
    }

    // ========== 构造器与 static ==========
    StaticDemo() {
        objectCount++;  // 每创建对象, 计数 +1
    }

    public static void main(String[] args) {
        // 静态成员: 类名.成员
        System.out.println(StaticDemo.classDesc);
        StaticDemo.staticMethod();

        // 实例成员: 对象.成员
        StaticDemo obj = new StaticDemo();
        System.out.println(obj.instanceVar);
        obj.instanceMethod();

        System.out.println("Objects created: " + StaticDemo.objectCount);
    }
}

// ========== 访问静态成员的注意事项 ==========
// 1. 静态方法不能使用 this/super
// 2. 静态方法不能访问实例变量/方法
// 3. 静态变量在类加载时初始化 (方法区)
// 4. 静态方法可以被重载, 但不能被重写 (override)
// 5. 通过对象引用访问静态成员 (不推荐)
//    obj.staticMethod()  // 实际等于 ClassName.staticMethod()
```


## 对象生命周期与垃圾回收


```
// ========== 对象生命周期 ==========
// 1. 声明: Student s;
// 2. 创建: new Student()  — 堆内存分配
// 3. 使用: s.study();
// 4. 不可达: s = null; 或超出作用域
// 5. GC 回收: JVM 自动回收

public class LifecycleDemo {

    String name;

    LifecycleDemo(String name) {
        this.name = name;
        System.out.println(name + " created");
    }

    void use() {
        System.out.println(name + " in use");
    }

    @Override
    protected void finalize() throws Throwable {
        // 不建议使用! (Java 9+ 已废弃)
        // GC 回收前调用, 不保证执行
        System.out.println(name + " is being GC'd");
    }

    public static void main(String[] args) {
        // 对象创建
        LifecycleDemo obj1 = new LifecycleDemo("Object 1");
        obj1.use();

        // 引用重新赋值 → 原对象变为垃圾
        obj1 = new LifecycleDemo("Object 2");
        obj1.use();

        // 匿名对象 (没有引用)
        new LifecycleDemo("Anonymous").use();

        // 请求 GC (不保证立即执行)
        System.gc();

        // 推荐: try-with-resources 管理资源
        // 而不是依赖 finalize()
    }
}

// ========== GC 与内存管理 ==========
// 堆内存分代:
// - Young Gen (Eden + Survivor 0/1)
// - Old Gen
// - Metaspace (Java 8+, 取代 PermGen)

// GC 算法:
// - Minor GC: 清理 Young Gen
// - Major GC / Full GC: 清理所有

// 对象成为垃圾的条件:
// 1. 引用赋 null
// 2. 超出作用域
// 3. 对象互相引用但整体不可达 (孤岛)

// 建议:
// - 不需要手动调用 System.gc()
// - 资源管理用 try-with-resources
// - 留意对象引用避免内存泄漏
```


> **Note:** 💡 类与对象要点: class 定义; new 创建对象; 构造器与类同名, 无返回类型, 可重载; this 区分字段/参数 + 构造器链式调用 this(); 实例成员属对象, 静态成员属类; static 方法不能访问实例成员; 对象用完后 GC 自动回收; finalize() 已废弃。


## 练习


<!-- Converted from: 10_Java 类与对象.html -->
