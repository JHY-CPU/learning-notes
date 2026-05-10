# Java 流程控制 if-else switch


## 🔄 Java 流程控制 if-else switch


if/else if/else 条件判断、switch 语句 (传统/箭头语法/表达式)、三元运算符、switch 模式匹配 (Java 17+)、yield 返回值。


## if-else 条件判断


```
public class IfElseDemo {
    public static void main(String[] args) {
        int score = 85;

        // ========== 基本 if ==========
        if (score >= 60) {
            System.out.println("Pass");
        }

        // ========== if-else ==========
        if (score >= 60) {
            System.out.println("Pass");
        } else {
            System.out.println("Fail");
        }

        // ========== if-else if-else ==========
        if (score >= 90) {
            System.out.println("A");
        } else if (score >= 80) {
            System.out.println("B");   // ← 执行这个
        } else if (score >= 70) {
            System.out.println("C");
        } else if (score >= 60) {
            System.out.println("D");
        } else {
            System.out.println("F");
        }

        // ========== 单行简化 ==========
        if (score >= 60) System.out.println("Pass");  // 只有一行

        // ========== 三元运算符 ==========
        String result = score >= 60 ? "Pass" : "Fail";
        System.out.println(result);

        // 嵌套三元:
        String grade = score >= 90 ? "A" :
                       score >= 80 ? "B" :
                       score >= 70 ? "C" : "D";

        // ========== 常见错误 ==========
        // if (score = 60)  // ❌ 编译错误! = 是赋值
        // if (score == 60) // ✅ 正确

        // 浮点数比较:
        double a = 0.1 + 0.2;
        // if (a == 0.3)    // false! 浮点精度问题
        double eps = 1e-10;
        if (Math.abs(a - 0.3) < eps) {  // ✅
            System.out.println("Equal");
        }
    }
}

// ========== 逻辑组合 ==========
public class LogicDemo {
    public static void main(String[] args) {
        int age = 25;
        boolean hasLicense = true;
        boolean isInsured = false;

        // && (AND): 都满足
        if (age >= 18 && hasLicense) {
            System.out.println("Can drive");
        }

        // || (OR): 任一满足
        if (isInsured || age > 25) {
            System.out.println("Covered");
        }

        // ! (NOT)
        if (!hasLicense) {
            System.out.println("No license");
        }

        // 短路: 左 false 不评估右
        String s = null;
        if (s != null && s.length() > 0) {  // 安全!
            System.out.println("Not empty");
        }

        // 括号明确优先级
        if ((age > 18 && hasLicense) || isInsured) {
            System.out.println("OK");
        }
    }
}
```


## switch 语句


```
// ========== switch 传统 ==========

public class SwitchDemo {
    public static void main(String[] args) {
        String day = "MONDAY";

        // ========== 传统 switch (break) ==========
        switch (day) {
            case "MONDAY":
            case "TUESDAY":
            case "WEDNESDAY":
            case "THURSDAY":
            case "FRIDAY":
                System.out.println("Weekday");
                break;
            case "SATURDAY":
            case "SUNDAY":
                System.out.println("Weekend");
                break;
            default:
                System.out.println("Invalid day");
        }

        // ========== 支持的类型 ==========
        // byte, short, int, char
        // String (Java 7+)
        // enum
        // var (推断为支持的)

        // ========== 箭头语法 (Java 14+) ==========
        switch (day) {
            case "MONDAY", "TUESDAY" -> System.out.println("Early week");
            case "WEDNESDAY", "THURSDAY" -> System.out.println("Mid week");
            case "FRIDAY" -> System.out.println("TGIF!");
            case "SATURDAY", "SUNDAY" -> System.out.println("Weekend!");
            default -> System.out.println("Unknown");
        }
        // 箭头: 不需要 break, 不会 fall-through
    }
}

// ========== switch 表达式 (Java 14+) ==========
public class SwitchExpression {
    public static void main(String[] args) {
        String day = "MONDAY";

        // ========== 用 yield 返回值 ==========
        String type = switch (day) {
            case "MONDAY":
            case "TUESDAY":
            case "WEDNESDAY":
            case "THURSDAY":
            case "FRIDAY":
                yield "Weekday";
            case "SATURDAY":
            case "SUNDAY":
                yield "Weekend";
            default:
                yield "Unknown";
        };
        System.out.println(type);  // "Weekday"

        // ========== 箭头 + 返回值 ==========
        int numLetters = switch (day) {
            case "MONDAY", "FRIDAY", "SUNDAY" -> 6;
            case "TUESDAY" -> 7;
            case "THURSDAY", "SATURDAY" -> 8;
            case "WEDNESDAY" -> 9;
            default -> 0;
        };
        System.out.println(numLetters);  // 6

        // ========== 代码块 yield ==========
        int result = switch (day) {
            case "MONDAY" -> {
                System.out.println("Start of week");
                yield 1;
            }
            case "FRIDAY" -> {
                System.out.println("End of week");
                yield 5;
            }
            default -> 0;
        };
    }
}

// ========== switch 模式匹配 (Java 17+ preview, 21+ final) ==========
// public class PatternSwitch {
//     static String format(Object obj) {
//         return switch (obj) {
//             case Integer i -> "Integer: " + i;
//             case String s -> "String: " + s;
//             case Long l -> "Long: " + l;
//             case null -> "null!";
//             default -> "Unknown: " + obj;
//         };
//     }
//
//     public static void main(String[] args) {
//         System.out.println(format(42));        // "Integer: 42"
//         System.out.println(format("hello"));   // "String: hello"
//         System.out.println(format(null));      // "null!"
//     }
// }
```


> **Note:** 💡 流程控制要点: if-else if-else; 三元 ?:; 浮点数比较用误差; 短路 && ||; switch 箭头语法 (无 fall-through); switch 表达式 yield 返回值 (Java 14+); switch 支持 String; 模式匹配 (Java 21+)。


## 练习


<!-- Converted from: 5_Java 流程控制 if-else switch.html -->
