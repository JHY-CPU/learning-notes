# Java 注解


## 🏷️ Java 注解


注解定义、内置注解 (@Override/@Deprecated/@SuppressWarnings)、元注解 (@Retention/@Target/@Inherited)、自定义注解、反射读取注解、注解处理器。


## 内置注解


```
// ========== 注解 (Annotation) ==========
// 给代码添加元数据 (metadata)
// 不影响程序运行 (除非通过反射处理)
// 编译器/框架/工具使用

import java.util.ArrayList;
import java.util.List;

public class BuiltInAnnotations {

    // ========== @Override ==========
    // 标记方法重写父类方法
    // 编译器检查签名是否匹配
    @Override
    public String toString() {
        return "BuiltInAnnotations instance";
    }

    // @Override 防止拼写错误:
    // @Override
    // public String tostring() { }  // ❌ 编译错误! 没有重写

    // ========== @Deprecated ==========
    // 标记已过时, 不推荐使用
    // 使用处会有编译警告
    @Deprecated
    static void oldMethod() {
        System.out.println("This method is deprecated");
    }

    // 可以加上说明 (Java 9+)
    @Deprecated(since = "2.0", forRemoval = true)
    static void legacyMethod() {
        System.out.println("Will be removed in future");
    }

    // ========== @SuppressWarnings ==========
    // 抑制编译器警告
    @SuppressWarnings("unchecked")           // 未检查转换
    @SuppressWarnings("deprecation")         // 使用了过期 API
    @SuppressWarnings("rawtypes")            // 原始类型
    @SuppressWarnings("all")                 // 所有警告

    @SuppressWarnings("unchecked")
    static void suppressDemo() {
        List rawList = new ArrayList();  // 通常会有警告
        List strList = rawList;  // unchecked 警告
    }

    // ========== @SafeVarargs ==========
    // 抑制可变参数泛型的堆污染警告
    @SafeVarargs
    static  void safeVarargs(T... args) {
        for (T arg : args) {
            System.out.println(arg);
        }
    }

    // ========== @FunctionalInterface ==========
    // 标记函数式接口 (只有一个抽象方法)
    @FunctionalInterface
    interface MyFunction {
        void execute();
        // void other();  // ❌ 编译错误
    }

    public static void main(String[] args) {
        oldMethod();  // 编译时有 deprecation 警告
    }
}
```


## 自定义注解


```
// ========== 定义注解 ==========
// 使用 @interface 关键字

import java.lang.annotation.*;

// ========== 元注解 (注解的注解) ==========

// @Retention: 注解保留策略
//   RetentionPolicy.SOURCE   — 仅源码 (编译丢弃)
//   RetentionPolicy.CLASS    — 字节码 (默认, 运行时不可反射)
//   RetentionPolicy.RUNTIME  — 运行时保留 (可反射读取)

// @Target: 注解适用目标
//   ElementType.TYPE         — 类/接口/枚举
//   ElementType.FIELD        — 字段
//   ElementType.METHOD       — 方法
//   ElementType.PARAMETER    — 参数
//   ElementType.CONSTRUCTOR  — 构造器
//   ElementType.LOCAL_VARIABLE — 局部变量
//   ElementType.ANNOTATION_TYPE — 注解
//   ElementType.PACKAGE      — 包

// @Inherited: 子类继承父类的注解
// @Documented: 包含在 Javadoc 中
// @Repeatable: 可在同一位置重复使用 (Java 8+)

// ========== 自定义注解示例 ==========

@Retention(RetentionPolicy.RUNTIME)  // 运行时保留
@Target(ElementType.METHOD)           // 只能用于方法
@interface Loggable {
    // 注解元素 (类似方法)
    String value() default "INFO";    // 默认 "INFO"
    boolean showArgs() default false;
}

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
@interface NotNull {
    String message() default "Field must not be null";
}

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
@interface Table {
    String name();
    String schema() default "public";
}

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
@interface Column {
    String name();
    boolean primaryKey() default false;
    boolean nullable() default true;
    int length() default 255;
}

// ========== 使用自定义注解 ==========

@Table(name = "users", schema = "app")
class User {
    @Column(name = "user_id", primaryKey = true)
    private Long id;

    @Column(name = "username", nullable = false, length = 50)
    private String username;

    @Column(name = "email")
    private String email;

    @Loggable("DEBUG")
    public void login() {
        System.out.println("User logged in");
    }

    @Loggable(value = "ERROR", showArgs = true)
    public void failedLogin(String reason) {
        System.out.println("Login failed: " + reason);
    }
}
```


## 反射读取注解


```
// ========== 反射读取注解 ==========
// 注解本身不会做任何事
// 需要工具/框架通过反射读取并处理

import java.lang.reflect.*;

public class AnnotationProcessor {

    // ========== 读取类注解 ==========
    static void readTableAnnotation(Class clazz) {
        // 获取类上的注解
        Table table = clazz.getAnnotation(Table.class);
        if (table != null) {
            System.out.println("Table: " + table.schema() + "." + table.name());
        }
    }

    // ========== 读取字段注解 ==========
    static void readColumnAnnotations(Class clazz) {
        for (Field field : clazz.getDeclaredFields()) {
            Column col = field.getAnnotation(Column.class);
            if (col != null) {
                System.out.printf("Field: %s → Column: %s (pk=%b, nullable=%b)%n",
                    field.getName(), col.name(), col.primaryKey(), col.nullable());
            }
        }
    }

    // ========== 读取方法注解 ==========
    static void readMethodAnnotations(Class clazz) {
        for (Method method : clazz.getDeclaredMethods()) {
            Loggable log = method.getAnnotation(Loggable.class);
            if (log != null) {
                System.out.printf("Method: %s → Log level: %s (showArgs=%b)%n",
                    method.getName(), log.value(), log.showArgs());
            }
        }
    }

    // ========== 检查是否有注解 ==========
    static void validateAnnotations(Object obj) throws IllegalAccessException {
        Class clazz = obj.getClass();
        for (Field field : clazz.getDeclaredFields()) {
            NotNull notNull = field.getAnnotation(NotNull.class);
            if (notNull != null) {
                field.setAccessible(true);
                if (field.get(obj) == null) {
                    throw new IllegalArgumentException(
                        notNull.message() + ": " + field.getName()
                    );
                }
            }
        }
    }

    // ========== 生成 SQL ==========
    static String generateCreateTableSQL(Class clazz) {
        Table table = clazz.getAnnotation(Table.class);
        if (table == null) return "";

        StringBuilder sql = new StringBuilder();
        sql.append("CREATE TABLE ").append(table.schema()).append(".")
           .append(table.name()).append(" (\n");

        for (Field field : clazz.getDeclaredFields()) {
            Column col = field.getAnnotation(Column.class);
            if (col == null) continue;

            sql.append("  ").append(col.name()).append(" ")
               .append(mapType(field.getType()))
               .append(col.primaryKey() ? " PRIMARY KEY" : "")
               .append(col.nullable() ? "" : " NOT NULL")
               .append(",\n");
        }
        sql.append(");");
        return sql.toString();
    }

    static String mapType(Class type) {
        if (type == Long.class || type == long.class) return "BIGINT";
        if (type == String.class) return "VARCHAR(255)";
        if (type == Integer.class || type == int.class) return "INT";
        if (type == Boolean.class || type == boolean.class) return "BOOLEAN";
        return "TEXT";
    }

    public static void main(String[] args) throws Exception {
        readTableAnnotation(User.class);
        readColumnAnnotations(User.class);
        readMethodAnnotations(User.class);

        System.out.println("\n--- Generated SQL ---");
        System.out.println(generateCreateTableSQL(User.class));

        // 验证 @NotNull
        User user = new User();
        // validateAnnotations(user);  // 抛出异常: username is null
    }
}
```


## 注解进阶


```
// ========== @Repeatable (Java 8+) ==========
// 重复使用同一注解

import java.lang.annotation.Repeatable;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
@Repeatable(Schedules.class)  // 容器注解
@interface Schedule {
    String day();
    String time();
}

// 容器注解 (存储重复注解)
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
@interface Schedules {
    Schedule[] value();
}

class RepeatableDemo {
    @Schedule(day = "MON", time = "09:00")
    @Schedule(day = "WED", time = "14:00")
    @Schedule(day = "FRI", time = "11:00")
    void scheduledTask() {
        System.out.println("Running task...");
    }

    static void readSchedules() throws Exception {
        Method method = RepeatableDemo.class
            .getMethod("scheduledTask");

        // 方式1: 获取重复注解
        Schedule[] schedules = method.getAnnotationsByType(Schedule.class);
        for (Schedule s : schedules) {
            System.out.println(s.day() + " @ " + s.time());
        }

        // 方式2: 获取容器注解
        Schedules container = method.getAnnotation(Schedules.class);
        if (container != null) {
            for (Schedule s : container.value()) {
                System.out.println(s.day() + " @ " + s.time());
            }
        }
    }
}

// ========== 注解注解 ==========
// 注解元素类型限制:
// - 基本类型 (int, boolean, double...)
// - String
// - Class
// - enum
// - 注解
// - 以上类型的数组

// 默认值: 不能为 null, 必须是非空常量

// ========== 常见框架注解 ==========
// JUnit: @Test, @Before, @After, @BeforeEach, @AfterEach
// Spring: @Component, @Service, @Autowired, @GetMapping
// JPA: @Entity, @Table, @Column, @Id
// Lombok: @Data, @Getter, @Setter, @Builder, @Slf4j
// Jackson: @JsonIgnore, @JsonProperty, @JsonFormat
// Validation: @NotNull, @NotBlank, @Email, @Pattern
```


> **Note:** 💡 注解要点: @interface 定义; 元注解 @Retention/@Target/@Inherited/@Documented/@Repeatable; @Retention RUNTIME 才能反射; 注解元素类型有限制; @Repeatable 重复使用需容器注解; 框架大量使用注解 (Spring/JPA/JUnit/Lombok); 注解本身无逻辑, 需要处理器通过反射读取。


## 练习


<!-- Converted from: 19_Java 注解.html -->
