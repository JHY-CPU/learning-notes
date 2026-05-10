# Spring Boot 入门


## 🚀 Spring Boot 入门


Spring Boot 概述、@SpringBootApplication 注解、自动配置原理简介、Spring Initializr 项目生成、标准目录结构。


## Spring Boot 概述


```
// ========== Spring Boot ==========
// Spring 的"约定优于配置"解决方案
// 目标: 快速创建生产级 Spring 应用

// ========== 核心优势 ==========
// 1. 自动配置 — 根据依赖自动配置 Bean
// 2. 起步依赖 — 一组常用依赖的整合 (starter)
// 3. 嵌入式服务器 — 内嵌 Tomcat/Jetty/Undertow
// 4. Actuator — 生产级监控 (健康/指标/环境)
// 5. 外部化配置 — properties/yml 灵活配置
// 6. 无需 XML — 纯 Java 配置 + 注解

// ========== 版本选择 ==========
// Spring Boot 3.x (2022+) — 要求 Java 17+
//   - Jakarta EE 9+ (javax.* → jakarta.*)
//   - GraalVM Native Image 支持
//   - 基于 Spring Framework 6
//
// Spring Boot 2.x — 要求 Java 8+ (已逐渐停止维护)

// ========== 核心注解 ==========
@SpringBootApplication  // 组合注解, 包含 3 个:
// @SpringBootConfiguration — 标记为配置类
// @EnableAutoConfiguration  — 启用自动配置
// @ComponentScan           — 扫描当前包及子包的 @Component

// ========== 最小应用 ==========
@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}

// ========== 启动流程 ==========
// 1. 加载 SpringApplication
// 2. 推断 Web 应用类型 (Reactive/Servlet/None)
// 3. 加载 ApplicationContextInitializer
// 4. 加载 ApplicationListener
// 5. 推断 main 类
// 6. 启动内嵌服务器 (Web 应用)
// 7. 执行自动配置
// 8. 运行 ApplicationRunner / CommandLineRunner
```


## Spring Initializr


```
// ========== Spring Initializr ==========
// 项目脚手架: 快速生成 Spring Boot 项目

// ========== 访问方式 ==========
// 1. Web: https://start.spring.io
// 2. IDE: IntelliJ IDEA → New Project → Spring Initializr
// 3. CLI: curl https://start.spring.io/starter.zip -d parameters...

// ========== 选择项 ==========
// Project:     Maven / Gradle
// Language:    Java / Kotlin / Groovy
// Spring Boot: 3.2.x / 3.1.x
// Group:       com.example
// Artifact:    my-app
// Packaging:   Jar / War
// Java:        17 / 21

// ========== 常用 Starter ==========
// Web:
//   spring-boot-starter-web           — Web (Tomcat + Spring MVC)
//   spring-boot-starter-webflux       — Reactive Web (Netty)
//   spring-boot-starter-websocket     — WebSocket

// 数据:
//   spring-boot-starter-data-jpa      — JPA + Hibernate
//   spring-boot-starter-data-redis    — Redis
//   spring-boot-starter-data-mongodb  — MongoDB
//   spring-boot-starter-jdbc          — JDBC + HikariCP

// 安全:
//   spring-boot-starter-security      — Spring Security

// 测试:
//   spring-boot-starter-test          — JUnit 5 + Mockito + AssertJ

// 生产:
//   spring-boot-starter-actuator      — 监控端点
//   spring-boot-starter-validation    — Bean Validation

// 开发者工具:
//   spring-boot-devtools              — 热重载 + LiveReload

// ========== 项目目录结构 ==========
src/
├── main/
│   ├── java/com/example/myapp/
│   │   ├── MyApplication.java          // 启动类
│   │   ├── controller/                 // 控制器
│   │   ├── service/                    // 服务层
│   │   ├── repository/                 // 数据访问
│   │   ├── model/                      // 实体/DTO
│   │   └── config/                     // 配置类
│   └── resources/
│       ├── application.yml             // 配置文件
│       ├── static/                     // 静态资源
│       └── templates/                  // 模板 (Thymeleaf)
└── test/java/com/example/myapp/
    └── MyApplicationTests.java         // 测试类
```


## @SpringBootApplication 详解


```
// ========== @SpringBootApplication 分解 ==========

// 1. @SpringBootConfiguration
//    本质是 @Configuration, 标记为配置类
//    与普通 @Configuration 区别: 自动配置会寻找 @SpringBootConfiguration

// 2. @EnableAutoConfiguration
//    核心: 启用 Spring Boot 自动配置
//    通过 AutoConfigurationImportSelector 加载
//    META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports
//    文件列出所有自动配置类
//    条件匹配 (@Conditional) 决定是否生效

// 3. @ComponentScan
//    扫描 basePackages (默认: 启动类所在包及子包)
//    自动注册 @Component / @Service / @Repository / @Controller / @Configuration

// ========== 启动类位置 ==========
// ✅ 正确: 放在根包, 自动扫描所有子包
// com.example.myapp/
//   ├── MyApplication.java           // 启动类放这里
//   ├── controller/...
//   ├── service/...
//   └── repository/...

// ❌ 错误: 放在子包, 扫描不到其他包
// com.example.myapp.config/
//   └── MyApplication.java           // 扫描不到 controller 等!

// ========== 排除自动配置 ==========
@SpringBootApplication(exclude = {
    DataSourceAutoConfiguration.class,  // 排除数据源配置
    SecurityAutoConfiguration.class     // 排除安全配置
})

// ========== 自定义扫描 ==========
@SpringBootApplication
@ComponentScan(basePackages = {
    "com.example.myapp",
    "com.example.common"
})

// ========== SpringApplication 配置 ==========
@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication app = new SpringApplication(Application.class);
        app.setBannerMode(Banner.Mode.OFF);   // 关闭 Banner
        app.setAdditionalProfiles("dev");     // 设置 Profile
        app.setDefaultProperties(Map.of(      // 设置默认属性
            "server.port", "8081"
        ));
        app.run(args);
    }
}

// ========== ApplicationRunner 与 CommandLineRunner ==========
// 启动后执行特定逻辑
@Bean
ApplicationRunner initDatabase() {
    return args -> {
        System.out.println("启动后执行: " + Arrays.toString(args.getSourceArgs()));
    };
}

@Bean
CommandLineRunner hello() {
    return args -> {
        System.out.println("Hello from CommandLineRunner!");
    };
}
```


## 内嵌服务器与 DevTools


```
// ========== 内嵌服务器 ==========
// Spring Boot 内嵌 Servlet 容器, 直接 java -jar 启动
// 无需部署到外部 Tomcat

// ========== 支持的服务器 ==========
// Tomcat   — 默认 (spring-boot-starter-web)
// Jetty    — 需替换:
//   <exclusion>
//     <groupId>org.springframework.boot</groupId>
//     <artifactId>spring-boot-starter-tomcat</artifactId>
//   </exclusion>
//   添加 spring-boot-starter-jetty
// Undertow — 高性能, 替代 Tomcat:
//   类似方式排除 Tomcat 添加 Undertow

// ========== 服务器配置 ==========
server:
  port: 8080                         # 端口
  servlet:
    context-path: /api               # 上下文路径
    session:
      timeout: 30m                   # 会话超时
  tomcat:
    max-connections: 10000           # 最大连接数
    max-threads: 200                 # 最大线程数
    min-spare-threads: 10            # 最小空闲线程
    connection-timeout: 5s           # 连接超时
  compression:
    enabled: true                    # 启用压缩
    mime-types: text/html,text/css,application/json
    min-response-size: 1024

// ========== DevTools ==========
// 开发时自动重启 + LiveReload
// 依赖:
// <dependency>
//     <groupId>org.springframework.boot>
//     <artifactId>spring-boot-devtools</artifactId>
//     <optional>true</optional>
// </dependency>

// 功能:
// - 自动重启: 文件变化后自动重启 (比手动快)
// - LiveReload: 浏览器自动刷新 (需插件)
// - 禁用缓存: Thymeleaf/Freemarker 模板禁用缓存
// - 远程调试: 远程应用的 DevTools 支持

// 自动重启排除:
spring.devtools.restart.exclude=static/**,public/**,templates/**

// ========== 打包与运行 ==========
// Maven: mvn clean package → java -jar target/app.jar
// Gradle: ./gradlew build → java -jar build/libs/app.jar

// 运行参数:
java -jar myapp.jar --server.port=8081                    // 命令行覆盖
java -jar myapp.jar --spring.profiles.active=prod          // Profile
java -Dspring.profiles.active=prod -jar myapp.jar          // JVM 参数
java -jar myapp.jar &> app.log &                           // 后台运行
```


> **Note:** 💡 Spring Boot 要点: @SpringBootApplication = @SpringBootConfig + @EnableAutoConfig + @ComponentScan; 自动配置根据 classpath 依赖自动配置 Bean; Spring Initializr 快速生成项目; 标准目录结构; 内嵌 Tomcat 无需外部容器; java -jar 直接运行; DevTools 自动重启 + LiveReload; application.yml 集中配置。


## 练习


<!-- Converted from: 0_Spring Boot 入门.html -->
