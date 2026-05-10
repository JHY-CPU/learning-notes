# Java String 与字符串


## 📝 Java String 与字符串


String 不可变性、常用方法 (length/split/substring/indexOf)、StringBuilder/StringBuffer、字符串常量池、字符串比较、正则表达式。


## String 基础


```
public class StringBasics {
    public static void main(String[] args) {
        // ========== 创建字符串 ==========
        String s1 = "Hello";               // 字面量 (常量池)
        String s2 = new String("Hello");   // 堆上创建
        String s3 = String.valueOf(123);   // "123"

        // ========== 字符串常量池 ==========
        // JVM 维护字符串常量池
        // 字面量创建的字符串在常量池中复用
        String a = "Hello";
        String b = "Hello";
        System.out.println(a == b);        // true (同一引用)

        String c = new String("Hello");
        System.out.println(a == c);        // false (不同对象)

        // intern() → 从常量池取
        String d = c.intern();
        System.out.println(a == d);        // true

        // ========== 不可变性 ==========
        // String 不可变 (final class)
        // 每次修改创建新对象
        String s = "Hello";
        s = s + " World";  // 创建新对象, 原 "Hello" 不变
        // 不可变的好处:
        // 1. 线程安全
        // 2. 常量池复用
        // 3. 哈希缓存

        // ========== 长度 ==========
        String str = "Hello 世界";
        int len = str.length();           // 8 (字符数)
        boolean empty = str.isEmpty();    // false
        boolean blank = "  ".isBlank();   // true (Java 11+)
    }
}

// ========== 常用方法 ==========
public class StringMethods {
    public static void main(String[] args) {
        String str = "  Hello, Java World!  ";

        // 去空格
        str.trim();              // "Hello, Java World!"
        str.strip();             // Java 11+ (支持 Unicode)
        str.stripLeading();      // 去左空格
        str.stripTrailing();     // 去右空格

        // 大小写
        str.toUpperCase();       // "  HELLO, JAVA WORLD!  "
        str.toLowerCase();       // "  hello, java world!  "

        // 判断
        str.contains("Java");       // true
        str.startsWith("  He");     // true
        str.endsWith("!  ");       // true
        str.equals("hello");        // false
        str.equalsIgnoreCase("  hello, java world!  "); // true

        // 搜索
        str.indexOf("Java");        // 9  (位置)
        str.indexOf("o", 5);        // 从5开始找
        str.lastIndexOf("o");       // 最后出现位置
        // 没找到返回 -1

        // 截取
        str.substring(2, 7);        // "Hello" [2,7)
        str.substring(9);           // "Java World!  "

        // 替换
        str.replace("Java", "Kotlin"); // "...Kotlin World!"
        str.replaceAll("[aeiou]", "*"); // 正则替换
        str.replaceFirst("l", "L");    // 替换第一个

        // 拆分
        String[] parts = "a,b,c".split(",");      // ["a","b","c"]
        String[] words = str.split("\\s+");        // 按空白拆分

        // 合并
        String joined = String.join("-", "a", "b", "c"); // "a-b-c"
        String csv = String.join(",", parts);             // "a,b,c"

        // 重复 (Java 11+)
        String stars = "*".repeat(5);  // "*****"

        // 格式化
        String fmt = String.format("Name: %s, Age: %d", "Alice", 25);

        // 字符数组
        char[] chars = str.toCharArray();
        // chars[2] → 'H'
    }
}
```


## StringBuilder 与 StringBuffer


```
// ========== StringBuilder ==========
// 可变字符串, 适合大量拼接
// 非线程安全 (但更快)

public class StringBuilderDemo {
    public static void main(String[] args) {
        // 创建
        StringBuilder sb = new StringBuilder();      // 默认16
        StringBuilder sb2 = new StringBuilder(100);  // 指定容量
        StringBuilder sb3 = new StringBuilder("Hello");

        // 追加
        sb.append("Hello");
        sb.append(' ');
        sb.append("World");
        sb.append(123);
        // sb = "Hello World123"

        // 链式调用
        StringBuilder chain = new StringBuilder()
            .append("[")
            .append("data")
            .append("]");
        // "[data]"

        // 插入
        sb.insert(5, ",");        // "Hello, World123"

        // 删除
        sb.delete(5, 7);          // 删除 [5,7)
        sb.deleteCharAt(0);       // 删除第一个

        // 替换
        sb.replace(0, 5, "Hi");   // 替换 [0,5)

        // 反转
        sb.reverse();             // 反转字符串

        // 转 String
        String result = sb.toString();

        // 容量
        int cap = sb.capacity();  // 当前容量
        sb.trimToSize();          // 缩容到实际大小

        // ========== StringBuilder vs StringBuffer ==========
        // StringBuilder:  非线程安全, 更快 (推荐)
        // StringBuffer:   线程安全 (同步方法), 较慢

        // 一般用 StringBuilder
        // 多线程共享用 StringBuffer

        // ========== 性能对比 ==========
        // + 拼接:    O(n²)  - 每次创建新对象
        // StringBuilder: O(n)  - 原地修改

        // ❌ 不要这样:
        String s = "";
        for (int i = 0; i < 1000; i++) {
            s += i;  // 创建 1000 个临时对象!
        }

        // ✅ 用 StringBuilder:
        StringBuilder sb4 = new StringBuilder();
        for (int i = 0; i < 1000; i++) {
            sb4.append(i);
        }
        String result2 = sb4.toString();
    }
}

// ========== 文本块 (Java 13+, 正式 15) ==========
public class TextBlockDemo {
    public static void main(String[] args) {
        String json = """
            {
                "name": "Alice",
                "age": 25,
                "city": "Beijing"
            }
            """;

        String html = """


Hello


            """;

        // 自动处理缩进和换行
        System.out.println(json);
    }
}
```


## 正则表达式


```
// ========== 正则表达式 ==========

import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class RegexDemo {
    public static void main(String[] args) {
        // 1. String 方法 (简单)
        "abc123".matches("\\w+\\d+");     // true
        "a,b,c".split(",");               // 拆分
        "hello".replaceAll("l", "L");     // "heLLo"

        // 2. Pattern/Matcher (复杂)
        String text = "My email: alice@test.com, bob@test.com";
        Pattern pattern = Pattern.compile("\\w+@\\w+\\.\\w+");
        Matcher matcher = pattern.matcher(text);

        while (matcher.find()) {
            System.out.println(matcher.group());  // alice@test.com
            System.out.println(matcher.start());  // 起始位置
            System.out.println(matcher.end());    // 结束位置
        }

        // 3. 分组
        Pattern p2 = Pattern.compile("(\\w+)@(\\w+)\\.(\\w+)");
        Matcher m2 = p2.matcher("alice@test.com");
        if (m2.matches()) {
            System.out.println(m2.group(0));  // 完整
            System.out.println(m2.group(1));  // alice
            System.out.println(m2.group(2));  // test
            System.out.println(m2.group(3));  // com
        }

        // 4. 常用正则
        // 邮箱: ^\\w+@\\w+\\.\\w+$
        // 手机: ^1[3-9]\\d{9}$
        // URL:  ^https?://.+$
        // IP:   ^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$
    }
}

// ========== String 方法速查 ==========
// length()        长度
// isEmpty()       空
// charAt(i)       取字符
// substring(s,e)  子串 [s,e)
// indexOf(s)      位置 (-1 未找到)
// contains(s)     包含
// startsWith(s)   前缀
// endsWith(s)     后缀
// equals(s)       相等
// equalsIgnoreCase 忽略大小写
// compareTo(s)    字典序比较
// toUpperCase()   大写
// toLowerCase()   小写
// trim()          去空格
// strip()         去空格(Java11)
// replace(a,b)    替换
// replaceAll(r,s) 正则替换
// split(r)        正则拆分
// join(d,arr)     合并
// repeat(n)       重复
// format(f,args)  格式化
```


> **Note:** 💡 String 要点: 不可变 final 类; 常量池 intern; equals() 比较内容; StringBuilder 拼接; 文本块 """ """ (Java15+); Pattern/Matcher 正则; 大量拼接用 StringBuilder; StringBuffer 线程安全; 常用方法 length/indexOf/substring/split/replace。


## 练习


<!-- Converted from: 3_Java String 与字符串.html -->
