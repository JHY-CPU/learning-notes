# Java 泛型


## 📦 Java 泛型


泛型类/接口/方法、类型擦除、通配符 ? / ? extends / ? super、上下界、泛型限制、PECS 原则、桥方法。


## 泛型类与接口


```
// ========== 泛型 (Generics) ==========
// 类型参数化: 定义时用类型参数, 使用时指定具体类型
// 好处: 编译时类型安全, 避免强制转换

// ========== 泛型类 ==========
// T: Type (类型参数, 习惯单大写字母)
// E: Element (集合元素)
// K: Key, V: Value
// N: Number
// ?: 通配符

class Box {
    private T value;

    public Box(T value) {
        this.value = value;
    }

    public T getValue() {
        return value;
    }

    public void setValue(T value) {
        this.value = value;
    }

    // 泛型方法 (在泛型类中)
    public  Box transform(java.util.function.Function fn) {
        return new Box<>(fn.apply(value));
    }
}

// ========== 使用泛型类 ==========
public class GenericClassDemo {
    public static void main(String[] args) {
        // 类型推断 (Java 7+: 菱形运算符 <>)
        Box stringBox = new Box<>("Hello");
        System.out.println(stringBox.getValue().length());  // 5

        Box intBox = new Box<>(42);
        System.out.println(intBox.getValue() + 1);  // 43

        // 原始类型 (Raw Type, 不推荐!)
        Box rawBox = new Box("raw");  // 编译警告
        // Object val = rawBox.getValue();  // 返回 Object

        // 泛型的好处
        // 没有泛型:
        // Object val = stringBox.getValue();
        // String s = (String) val;  // 需要强制转换

        // 有泛型 (自动):
        String s = stringBox.getValue();  // 自动类型安全!
    }
}

// ========== 泛型接口 ==========
interface Pair {
    K getKey();
    V getValue();
}

class OrderedPair implements Pair {
    private K key;
    private V value;

    public OrderedPair(K key, V value) {
        this.key = key;
        this.value = value;
    }

    @Override
    public K getKey() { return key; }

    @Override
    public V getValue() { return value; }
}

// 使用:
// Pair pair = new OrderedPair<>("age", 25);
// String key = pair.getKey();     // "age"
// Integer val = pair.getValue();  // 25
```


## 泛型方法


```
// ========== 泛型方法 ==========
// 在方法返回类型前声明类型参数

public class GenericMethodDemo {

    // ========== 泛型方法 ==========
    //  放在返回类型前
    public static  T identity(T value) {
        return value;
    }

    // 多个类型参数
    public static  void printPair(K key, V value) {
        System.out.println(key + " = " + value);
    }

    // 泛型 + 可变参数
    @SafeVarargs
    public static  T[] toArray(T... elements) {
        return elements;  // 返回传入的数组
    }

    // ========== 类型参数边界 (Bounded Type Parameters) ==========
    // 限制 T 必须是 Number 的子类
    public static  double sum(T a, T b) {
        return a.doubleValue() + b.doubleValue();
    }

    // 多重边界: T 是 A 的子类且实现 B 和 C
    // public static  void process(T obj)

    // 泛型方法绑定到 Comparable
    public static > T max(T a, T b) {
        return a.compareTo(b) >= 0 ? a : b;
    }

    public static void main(String[] args) {
        System.out.println(identity("Hello"));    // "Hello"
        System.out.println(identity(42));          // 42

        printPair("name", "Alice");
        printPair(1, "one");

        System.out.println(sum(3, 5));     // 8.0 (Integer)
        System.out.println(sum(3.5, 2.1)); // 5.6 (Double)
        // System.out.println(sum("a", "b")); // ❌ 编译错误!

        System.out.println(max(10, 20));     // 20
        System.out.println(max("apple", "banana")); // "banana"
    }
}

// ========== 类型推断 ==========
// Java 7: 菱形 <>
// Java 8: 目标类型推断 (方法参数)
// Java 10+: var 局部变量

// var list = new ArrayList();  // ArrayList
```


## 通配符与上下界


```
// ========== 通配符 ? ==========
// ? : 未知类型 (通配符)
// ? extends T : 上限通配符 (T 或 T 的子类)
// ? super T   : 下限通配符 (T 或 T 的父类)

import java.util.*;

public class WildcardDemo {

    // ========== 无界通配符 ? ==========
    // 只能读 (Object), 不能写 (除 null)
    static void printList(List list) {
        for (Object obj : list) {
            System.out.print(obj + " ");
        }
        // list.add("hello");  // ❌ 不能添加!
        // list.add(null);     // ✅ 只能加 null
    }

    // ========== 上界 ? extends T ==========
    // 可以读 (T 类型), 不能写
    static double sumOfList(List list) {
        double sum = 0;
        for (Number n : list) {
            sum += n.doubleValue();
        }
        // list.add(42);  // ❌ 不能添加!
        return sum;
    }

    // ========== 下界 ? super T ==========
    // 可以写 (T 或子类), 读只能 Object
    static void addNumbers(List list) {
        list.add(1);    // ✅ Integer 可以
        list.add(2);    // ✅ Integer 可以
        // list.add(3.5);  // ❌ Double 不行
    }

    // ========== PECS 原则 ==========
    // Producer Extends, Consumer Super
    // 生产者(读)用 extends, 消费者(写)用 super

    // 示例: Collections.copy
    // public static  void copy(
    //     List dest,    // 消费者: 写入
    //     List src    // 生产者: 读取
    // )

    public static void main(String[] args) {
        // ? extends Number 示例
        List ints = Arrays.asList(1, 2, 3);
        List dbls = Arrays.asList(1.5, 2.5);
        System.out.println(sumOfList(ints));  // 6.0
        System.out.println(sumOfList(dbls));  // 4.0

        // ? super Integer 示例
        List nums = new ArrayList<>();
        addNumbers(nums);
        System.out.println(nums);  // [1, 2]

        // ? 示例
        printList(Arrays.asList("a", "b", "c"));
    }
}
```


## 类型擦除与限制


```
// ========== 类型擦除 (Type Erasure) ==========
// 泛型只在编译时存在, 运行时被擦除
// JVM 不知道泛型类型

public class TypeErasure {

    public static void main(String[] args) {
        ArrayList strList = new ArrayList<>();
        ArrayList intList = new ArrayList<>();

        // 运行时类型相同!
        System.out.println(strList.getClass() == intList.getClass());
        // true (都是 ArrayList.class)

        // 擦除规则:
        // Box       → Box (Object)
        // Box → Box (Number)
        // Box> → Box (Comparable)
    }
}

// ========== 桥方法 (Bridge Method) ==========
// 编译器生成桥方法处理多态

class Node {
    private T data;
    public void setData(T data) { this.data = data; }
}

class MyNode extends Node {
    @Override
    public void setData(Integer data) {  // 参数: Integer
        super.setData(data);
    }

    // 编译器生成桥方法 (合成):
    // public void setData(Object data) {
    //     setData((Integer) data);
    // }
}

// ========== 泛型的限制 ==========
class GenericRestrictions {
    // ❌ 1. 不能 new T()
    // public T create() { return new T(); }

    // ❌ 2. 不能 new T[]
    // public T[] createArray() { return new T[10]; }

    // ❌ 3. 基本类型不能做类型参数
    // Box box;  // 必须用 Box

    // ❌ 4. 不能用在静态变量
    // static T value;  // 编译错误

    // ❌ 5. 不能用在异常类
    // class MyException extends Exception { }

    // ❌ 6. 不能 instanceof 类型参数
    // if (obj instanceof T) { }

    // ✅ 7. 可以通过反射创建
    @SuppressWarnings("unchecked")
    public  T create(Class clazz) throws Exception {
        return clazz.getDeclaredConstructor().newInstance();
    }

    // ✅ 8. 可以通过数组转型
    @SuppressWarnings("unchecked")
    public  T[] createArray(Class clazz, int size) {
        return (T[]) java.lang.reflect.Array.newInstance(clazz, size);
    }
}
```


> **Note:** 💡 泛型要点: 泛型类/接口/方法用
> 声明类型参数; 类型擦除运行时消失; 菱形 <> 类型推断; 通配符 ? / extends / super; PECS (Producer Extends, Consumer Super); 上界可读不可写, 下界可写不可读; 泛型不能基本类型/new实例/静态变量/异常; 桥方法保证多态。


## 练习


<!-- Converted from: 18_Java 泛型.html -->
