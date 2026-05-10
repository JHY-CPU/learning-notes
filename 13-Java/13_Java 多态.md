# Java 多态


## 🎭 Java 多态


编译时多态 (重载) vs 运行时多态 (重写)、动态绑定、向上转型/向下转型、instanceof 模式匹配、多态的实际应用。


## 运行时多态


```
// ========== 多态 (Polymorphism) ==========
// 同一种行为, 多种实现方式
// 编译时多态: 方法重载 (静态绑定)
// 运行时多态: 方法重写 (动态绑定)

// ========== 运行时多态基础 ==========
// 父类引用, 子类对象 → 调用时执行子类重写方法

class Animal {
    String name;
    Animal(String name) { this.name = name; }
    void makeSound() { System.out.println("Some sound"); }
}

class Dog extends Animal {
    Dog(String name) { super(name); }
    @Override
    void makeSound() { System.out.println(name + ": Woof!"); }
}

class Cat extends Animal {
    Cat(String name) { super(name); }
    @Override
    void makeSound() { System.out.println(name + ": Meow!"); }
}

class Cow extends Animal {
    Cow(String name) { super(name); }
    @Override
    void makeSound() { System.out.println(name + ": Moo!"); }
}

public class PolymorphismDemo {
    public static void main(String[] args) {
        // ========== 向上转型 (Upcasting) ==========
        // 子类 → 父类, 自动隐式转换
        Animal a1 = new Dog("Buddy");   // Dog → Animal
        Animal a2 = new Cat("Kitty");   // Cat → Animal
        Animal a3 = new Cow("Bessie");  // Cow → Animal

        // ========== 动态绑定 ==========
        // 编译时看 Animal, 运行时看实际对象
        a1.makeSound();  // "Buddy: Woof!"  (Dog 的)
        a2.makeSound();  // "Kitty: Meow!"  (Cat 的)
        a3.makeSound();  // "Bessie: Moo!"  (Cow 的)

        // ========== 多态数组 ==========
        Animal[] animals = {
            new Dog("Rex"),
            new Cat("Luna"),
            new Cow("Daisy"),
            new Dog("Max")
        };

        for (Animal a : animals) {
            a.makeSound();  // 每个调用正确版本!
        }

        // ========== 多态参数 ==========
        makeItSound(new Dog("Charlie"));
        makeItSound(new Cat("Oliver"));
    }

    // 多态参数: 接受任何 Animal 子类
    static void makeItSound(Animal animal) {
        animal.makeSound();  // 动态绑定
    }

    // ========== 多态返回 ==========
    static Animal getRandomAnimal() {
        int choice = (int)(Math.random() * 3);
        return switch (choice) {
            case 0 -> new Dog("Random Dog");
            case 1 -> new Cat("Random Cat");
            default -> new Cow("Random Cow");
        };
    }
}
```


## 类型转换与 instanceof


```
// ========== 向下转型 (Downcasting) ==========
// 父类 → 子类, 需要显式强制转换
// 可能抛 ClassCastException

public class TypeCasting {
    public static void main(String[] args) {
        Animal a = new Dog("Buddy");

        // ========== 安全的向下转型 ==========
        if (a instanceof Dog) {
            Dog d = (Dog) a;     // 安全!
            d.wagTail();
        }

        // ========== 传统 instanceof ==========
        if (a instanceof Dog) {
            Dog d = (Dog) a;
            d.wagTail();
        } else if (a instanceof Cat) {
            Cat c = (Cat) a;
            c.purr();
        }

        // ========== 模式匹配 instanceof (Java 16+) ==========
        if (a instanceof Dog d) {        // 声明变量 + 自动转型!
            d.wagTail();
        } else if (a instanceof Cat c) {
            c.purr();
        }

        // ========== switch 模式匹配 (Java 21+) ==========
        String sound = switch (a) {
            case Dog d -> d.bark();
            case Cat c -> c.meow();
            case Cow c -> c.moo();
            case null -> "no animal";
            default -> "unknown";
        };
    }
}

class Dog extends Animal {
    Dog(String n) { super(n); }
    void wagTail() { System.out.println("Wagging tail"); }
    String bark() { return "Woof"; }
}

class Cat extends Animal {
    Cat(String n) { super(n); }
    void purr() { System.out.println("Purring"); }
    String meow() { return "Meow"; }
}

class Cow extends Animal {
    Cow(String n) { super(n); }
    void moo() { return "Moo"; }
}

// ========== 类型转换注意事项 ==========
class CastWarnings {
    public static void main(String[] args) {
        // ❌ 错误转换
        Animal a = new Dog("Buddy");
        // Cat c = (Cat) a;   // ClassCastException!

        // ❌ 无关联的类型
        // String s = (String) a;  // 编译错误! (无继承关系)

        // ✅ 安全的转换
        if (a instanceof Dog d) {
            System.out.println("It's a dog!");
        }

        // ========== 编译时类型 vs 运行时类型 ==========
        Animal animal = new Dog("Rex");
        // 编译时类型: Animal (编译器知道的)
        // 运行时类型: Dog (JVM 实际创建的)

        System.out.println(animal.getClass().getName());  // "Dog"
        System.out.println(animal instanceof Animal);      // true
        System.out.println(animal instanceof Dog);         // true
    }
}
```


## 多态实际应用


```
// ========== 多态的实际应用场景 ==========

// ========== 1. 策略模式 ==========
// 不同支付方式
interface PaymentStrategy {
    void pay(double amount);
}

class CreditCard implements PaymentStrategy {
    public void pay(double amount) {
        System.out.println("Paid " + amount + " via Credit Card");
    }
}

class Alipay implements PaymentStrategy {
    public void pay(double amount) {
        System.out.println("Paid " + amount + " via Alipay");
    }
}

class WeChatPay implements PaymentStrategy {
    public void pay(double amount) {
        System.out.println("Paid " + amount + " via WeChat Pay");
    }
}

class ShoppingCart {
    private PaymentStrategy payment;

    void setPayment(PaymentStrategy payment) {
        this.payment = payment;
    }

    void checkout(double total) {
        payment.pay(total);  // 多态调用
    }
}

// ========== 2. 模板方法模式 ==========
abstract class DataParser {
    // 模板方法 (final 防止子类修改)
    public final void parse() {
        openFile();
        readData();
        processData();
        closeFile();
    }

    abstract void readData();
    abstract void processData();

    void openFile() { System.out.println("Opening file..."); }
    void closeFile() { System.out.println("Closing file..."); }
}

class CSVParser extends DataParser {
    void readData() { System.out.println("Reading CSV..."); }
    void processData() { System.out.println("Processing CSV..."); }
}

class JSONParser extends DataParser {
    void readData() { System.out.println("Reading JSON..."); }
    void processData() { System.out.println("Processing JSON..."); }
}

// ========== 3. 集合多态 ==========
import java.util.*;

class CollectionPolymorphism {
    public static void main(String[] args) {
        // 面向接口/父类编程
        List list = new ArrayList<>();    // 可换成 LinkedList
        Map map = new HashMap<>(); // 可换成 TreeMap

        // 方法参数使用父类/接口
        printAll(new ArrayList<>(List.of("a", "b")));
        printAll(new LinkedList<>(List.of("c", "d")));
    }

    // 接受任何 List 实现
    static void printAll(List list) {
        for (String s : list) {
            System.out.println(s);
        }
    }
}

// ========== 多态的好处 ==========
// 1. 可替换性 — 子类替换父类
// 2. 可扩展性 — 新增子类不影响现有代码 (开闭原则)
// 3. 接口统一 — 同一套方法, 不同行为
// 4. 代码简化 — 减少 if-else/switch-case

// ========== 动态绑定原理 ==========
// JVM 使用虚方法表 (vtable)
// 每个类有虚方法表, 存储方法实际入口地址
// 调用时从 vtable 查找实际方法
// 比静态方法调用略慢 (但可忽略)
```


> **Note:** 💡 多态要点: 父类引用指向子类对象; 动态绑定调用子类重写方法; instanceof 安全向下转型; instanceof 模式匹配 (Java 16+) 自动变量声明; 编绎时多态 (重载) vs 运行时多态 (重写); 向上转型自动, 向下转型需强制; 多态应用: 策略模式/模板方法/集合框架; 开闭原则基础。


## 练习


<!-- Converted from: 13_Java 多态.html -->
