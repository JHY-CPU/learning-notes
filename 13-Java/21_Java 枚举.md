# Java 枚举


## 📋 Java 枚举


enum 定义、枚举常量、字段/方法/构造器、switch 与枚举、EnumMap/EnumSet、枚举实现接口、策略枚举模式。


## 枚举基础


```
// ========== 枚举 (enum) ==========
// 一组命名的常量
// enum 是 final class, 隐式继承 java.lang.Enum
// 枚举常量是 public static final 实例

// ========== 定义枚举 ==========
enum Day {
    MONDAY, TUESDAY, WEDNESDAY, THURSDAY,
    FRIDAY, SATURDAY, SUNDAY
}

enum Season {
    SPRING, SUMMER, AUTUMN, WINTER
}

// ========== 使用枚举 ==========
public class EnumBasics {
    public static void main(String[] args) {
        // ========== 引用枚举常量 ==========
        Day today = Day.MONDAY;
        System.out.println(today);  // "MONDAY"

        // ========== 枚举方法 ==========
        System.out.println(today.name());     // "MONDAY"
        System.out.println(today.ordinal());  // 0 (顺序)
        System.out.println(today.toString());// "MONDAY"

        // ========== 遍历 ==========
        System.out.println("=== Days of week ===");
        for (Day d : Day.values()) {  // values() 返回所有常量
            System.out.println(d.ordinal() + ": " + d);
        }

        // ========== 字符串 → 枚举 ==========
        Day day = Day.valueOf("MONDAY");  // "MONDAY" → Day.MONDAY
        System.out.println(day);  // "MONDAY"

        // Day.valueOf("monday");   // ❌ IllegalArgumentException! 区分大小写
        // Day.valueOf("invalid");  // ❌ IllegalArgumentException

        // ========== switch 与枚举 ==========
        switch (today) {
            case MONDAY:
                System.out.println("Start of work week");
                break;
            case FRIDAY:
                System.out.println("TGIF!");
                break;
            case SATURDAY:
            case SUNDAY:
                System.out.println("Weekend!");
                break;
            default:
                System.out.println("Midweek");
        }
    }
}

// ========== 枚举与 switch 表达式 ==========
class EnumSwitch {
    static String describe(Day day) {
        return switch (day) {
            case MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY -> "Workday";
            case SATURDAY, SUNDAY -> "Weekend";
            // 不需要 default (覆盖了所有分支)
        };
    }
}
```


## 枚举进阶: 字段与方法


```
// ========== 枚举可以有字段、构造器、方法 ==========
// 构造器默认 private (枚举不能被 new)

enum Planet {
    // 枚举常量可以传参给构造器
    MERCURY(3.303e23, 2.4397e6),
    VENUS(4.869e24, 6.0518e6),
    EARTH(5.976e24, 6.37814e6),
    MARS(6.421e23, 3.3972e6),
    JUPITER(1.9e27, 7.1492e7),
    SATURN(5.688e26, 6.0268e7),
    URANUS(8.686e25, 2.5559e7),
    NEPTUNE(1.024e26, 2.4746e7);

    // ========== 字段 ==========
    private final double mass;    // kg
    private final double radius;  // m

    // ========== 构造器 (private) ==========
    Planet(double mass, double radius) {
        this.mass = mass;
        this.radius = radius;
    }

    // ========== 方法 ==========
    public double getMass() { return mass; }
    public double getRadius() { return radius; }

    // 表面重力: G * mass / r²
    public double surfaceGravity() {
        return 6.67300E-11 * mass / (radius * radius);
    }

    // 物体在行星上的重量
    public double weight(double earthMass) {
        return earthMass * surfaceGravity();
    }
}

// ========== 使用带字段的枚举 ==========
public class EnumFieldsDemo {
    public static void main(String[] args) {
        // 访问枚举常量
        Planet p = Planet.EARTH;
        System.out.println(p.name() + ": " + p.surfaceGravity());

        // 计算在地球上 68kg 的人在各行星的重量
        double earthWeight = 68.0;
        for (Planet planet : Planet.values()) {
            System.out.printf("Weight on %s: %.2f kg%n",
                planet, planet.weight(earthWeight));
        }

        // 枚举常量可以比较 (== 和 equals 都行)
        System.out.println(p == Planet.EARTH);       // true
        System.out.println(p.equals(Planet.EARTH));  // true
    }
}
```


## 枚举抽象方法与接口


```
// ========== 枚举可以实现接口 ==========

interface Describable {
    String getDescription();
}

enum Status implements Describable {
    PENDING("Waiting for approval"),
    APPROVED("Application approved"),
    REJECTED("Application rejected"),
    CANCELLED("Cancelled by user");

    private final String description;

    Status(String description) {
        this.description = description;
    }

    @Override
    public String getDescription() {
        return description;
    }
}

// ========== 枚举抽象方法 (每个常量不同行为) ==========

enum Operation {
    PLUS {
        @Override
        public double apply(double x, double y) {
            return x + y;
        }
    },
    MINUS {
        @Override
        public double apply(double x, double y) {
            return x - y;
        }
    },
    TIMES {
        @Override
        public double apply(double x, double y) {
            return x * y;
        }
    },
    DIVIDE {
        @Override
        public double apply(double x, double y) {
            return x / y;
        }
    };

    // 抽象方法: 每个常量必须实现
    public abstract double apply(double x, double y);
}

// ========== 策略枚举 ==========
// 工资计算策略

enum PayrollDay {
    MONDAY(PayType.WEEKDAY),
    TUESDAY(PayType.WEEKDAY),
    WEDNESDAY(PayType.WEEKDAY),
    THURSDAY(PayType.WEEKDAY),
    FRIDAY(PayType.WEEKDAY),
    SATURDAY(PayType.WEEKEND),
    SUNDAY(PayType.WEEKEND);

    private final PayType payType;

    PayrollDay(PayType payType) {
        this.payType = payType;
    }

    double pay(double hoursWorked, double rate) {
        return payType.pay(hoursWorked, rate);
    }

    // 嵌套枚举
    private enum PayType {
        WEEKDAY {
            @Override
            double overtimePay(double hours, double rate) {
                return hours <= 8 ? 0 : (hours - 8) * rate * 1.5;
            }
        },
        WEEKEND {
            @Override
            double overtimePay(double hours, double rate) {
                return hours * rate * 2.0;
            }
        };

        abstract double overtimePay(double hours, double rate);

        double pay(double hoursWorked, double rate) {
            double base = Math.min(hoursWorked, 8) * rate;
            return base + overtimePay(hoursWorked, rate);
        }
    }
}

// ========== 使用 ==========
// double pay = PayrollDay.MONDAY.pay(10, 100);
// System.out.println(pay);
// // 工作日: 前 8h 正常, 后 2h 1.5 倍
// // = 8×100 + 2×100×1.5 = 1100
```


## EnumMap 与 EnumSet


```
// ========== EnumMap ==========
// 专门为枚举键设计的 Map
// 内部用数组实现, 性能优于 HashMap
// 键顺序 = 枚举声明顺序

import java.util.*;

public class EnumMapSetDemo {

    enum Color { RED, GREEN, BLUE, YELLOW, ORANGE }

    public static void main(String[] args) {
        // ========== EnumMap ==========
        EnumMap schedule = new EnumMap<>(Day.class);
        schedule.put(Day.MONDAY, "Work on project");
        schedule.put(Day.WEDNESDAY, "Team meeting");
        schedule.put(Day.FRIDAY, "Review PRs");

        System.out.println(schedule.get(Day.MONDAY));  // "Work on project"

        // 遍历 (按声明顺序)
        for (Map.Entry entry : schedule.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }

        // ========== EnumSet ==========
        // 高效的位向量实现

        // 创建包含所有常量的 Set
        EnumSet allDays = EnumSet.allOf(Day.class);
        System.out.println(allDays);  // [MON..SUN]

        // 创建空 Set
        EnumSet empty = EnumSet.noneOf(Day.class);

        // 范围
        EnumSet workdays = EnumSet.range(Day.MONDAY, Day.FRIDAY);
        System.out.println(workdays);  // [MON..FRI]

        // 特定集合
        EnumSet weekend = EnumSet.of(Day.SATURDAY, Day.SUNDAY);

        // 补集
        EnumSet notWeekend = EnumSet.complementOf(weekend);

        // ========== 实际应用 ==========
        // 权限控制
        enum Permission { READ, WRITE, EXECUTE, DELETE }

        class Role {
            private String name;
            private EnumSet permissions;

            Role(String name, Permission... perms) {
                this.name = name;
                this.permissions = EnumSet.copyOf(Arrays.asList(perms));
            }

            boolean hasPermission(Permission p) {
                return permissions.contains(p);
            }

            void addPermission(Permission p) {
                permissions.add(p);
            }
        }

        Role admin = new Role("Admin", Permission.values());
        Role user = new Role("User", Permission.READ, Permission.WRITE);
        System.out.println(admin.hasPermission(Permission.DELETE)); // true
        System.out.println(user.hasPermission(Permission.EXECUTE)); // false
    }
}

// ========== 枚举最佳实践 ==========
// 1. 需要常量组时优先用 enum, 不用 static final int
// 2. 枚举常量用大写字母
// 3. 需要行为时用枚举抽象方法或接口
// 4. 用 EnumMap/EnumSet 提高性能
// 5. 枚举不能继承, 但可以实现接口
// 6. switch 枚举省略 default 时需覆盖所有常量
```


> **Note:** 💡 枚举要点: enum 关键字; 常量是 public static final 实例; values()/valueOf()/name()/ordinal(); 枚举可以有字段/构造器(默认private)/方法; 枚举抽象方法让常量有不同行为; enum 实现接口; EnumMap 高性能; EnumSet 位向量; 枚举比 int 常量安全; 策略枚举模式嵌套枚举实现策略。


## 练习


<!-- Converted from: 21_Java 枚举.html -->
