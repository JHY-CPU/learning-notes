# Spring Boot 日志


## 📝 Spring Boot 日志


SLF4J + Logback 日志体系、日志级别、配置格式、文件输出与轮转、MDC 追踪、日志最佳实践。


## 日志体系


```
// ========== Spring Boot 日志 ==========
// 默认: SLF4J (门面) + Logback (实现)
// SLF4J — 日志门面 (提供统一 API)
// Logback — 实现 (Spring Boot 默认)
// Log4j2 — 可选替代 (性能更高)

// ========== 使用日志 ==========
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UserService {
    // 方式 1: 声明 Logger
    private static final Logger log = LoggerFactory.getLogger(UserService.class);

    public void createUser(String name) {
        log.trace("开始创建用户: {}", name);  // 跟踪 (最低)
        log.debug("处理参数: {}", name);      // 调试
        log.info("用户创建成功: {}", name);   // 信息 (默认)
        log.warn("用户名过长: {}", name);     // 警告
        log.error("创建用户失败", exception); // 错误
    }
}

// ========== Lombok @Slf4j ==========
// 更简洁的日志声明
import lombok.extern.slf4j.Slf4j;

@Slf4j                              // 自动生成 log 字段
@Service
public class UserService {

    public void createUser(String name) {
        log.info("创建用户: {}", name);  // 直接用 log
    }
}

// 其他 Lombok 日志注解:
// @Log       — java.util.logging
// @Log4j2    — Log4j2
// @XSlf4j    — 扩展 SLF4J

// ========== {} 占位符 ==========
// 使用 {} 避免字符串拼接 (性能更好)
log.info("User {} logged in at {}", username, LocalDateTime.now());
// 等价于: log.info("User " + username + " logged in at " + ...);
// 区别: {} 方式只在日志级别启用时才构建字符串
```


## application.yml 日志配置


```
// ========== application.yml 日志配置 ==========

logging:
  # ========== 日志级别 ==========
  level:
    root: INFO                          # 全局级别 (默认 INFO)
    com.example: DEBUG                  # 特定包 DEBUG
    org.springframework: WARN           # Spring 框架 WARN
    org.hibernate: ERROR               # Hibernate 只显示 ERROR
    org.springframework.security: TRACE # 安全模块 TRACE

  # ========== 文件输出 ==========
  file:
    name: logs/myapp.log                # 日志文件路径
    path: logs/                         # 日志目录 (默认: 项目根目录)
    max-size: 100MB                     # 单个文件最大 (Logback)
    max-history: 30                    # 保留天数
    total-size-cap: 1GB                 # 总容量上限

  # ========== 控制台格式 ==========
  pattern:
    console: "%d{yyyy-MM-dd HH:mm:ss} [%thread] %-5level %logger{36} - %msg%n"
    file:    "%d{yyyy-MM-dd HH:mm:ss} [%thread] %-5level %logger{36} - %msg%n"

  # ========== 日志分组 ==========
  group:
    myapp: com.example.myapp,com.example.common
    db: org.hibernate,org.springframework.jdbc

  level:
    myapp: DEBUG                        # 组级别
    db: WARN

// ========== 常用格式模式 ==========
// %d{pattern}     — 日期时间
// %thread         — 线程名
// %-5level        — 日志级别 (左对齐 5 字符)
// %logger{36}     — Logger 名 (缩写最多 36 字符)
// %msg            — 日志消息
// %n              — 换行
// %class          — 类名
// %method         — 方法名
// %line           — 行号
// %X{key}         — MDC 值
// %highlight      — 高亮级别 (控制台)
// %clr(..., ...)  — 彩色输出

// ========== 彩色控制台 ==========
// Spring Boot 默认启用彩色输出
// 可在 application.yml 中控制:
spring:
  output:
    ansi:
      enabled: always                   # ALWAYS/NEVER/DETECT
```


## Logback 高级配置


```
// ========== logback-spring.xml ==========
// 如果需要更细致的配置, 使用 logback-spring.xml
// 放在 src/main/resources/ 下
// 支持 Spring Profile 扩展

<?xml version="1.0" encoding="UTF-8"?>
<configuration>

    <!-- 引入 Spring Boot 默认配置 -->
    <include resource="org/springframework/boot/logging/logback/base.xml"/>

    <!-- ========== 控制台输出 ========== -->
    <appender name="CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>
                %d{HH:mm:ss.SSS} %highlight(%-5level) [%thread] %cyan(%logger{36}) - %msg%n
            </pattern>
        </encoder>
    </appender>

    <!-- ========== 滚动文件输出 ========== -->
    <appender name="FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <file>logs/myapp.log</file>
        <rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
            <!-- 每天滚动 + 压缩 -->
            <fileNamePattern>logs/myapp.%d{yyyy-MM-dd}.%i.gz</fileNamePattern>
            <timeBasedFileNamingAndTriggeringPolicy class=
                "ch.qos.logback.core.rolling.SizeAndTimeBasedFNATP">
                <maxFileSize>100MB</maxFileSize>
            </timeBasedFileNamingAndTriggeringPolicy>
            <maxHistory>30</maxHistory>
        </rollingPolicy>
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss} [%thread] %-5level %logger{36} - %msg%n</pattern>
        </encoder>
    </appender>

    <!-- ========== 异步 Appender (高性能) ========== -->
    <appender name="ASYNC" class="ch.qos.logback.classic.AsyncAppender">
        <appender-ref ref="FILE"/>
        <queueSize>512</queueSize>              <!-- 队列大小 -->
        <discardingThreshold>0</discardingThreshold>
        <neverBlock>true</neverBlock>           <!-- 不阻塞主线程 -->
    </appender>

    <!-- ========== Profile 特定配置 ========== -->
    <springProfile name="dev">
        <root level="INFO">
            <appender-ref ref="CONSOLE"/>
        </root>
        <logger name="com.example" level="DEBUG"/>
    </springProfile>

    <springProfile name="prod">
        <root level="WARN">
            <appender-ref ref="ASYNC"/>
        </root>
        <logger name="com.example" level="INFO"/>
    </springProfile>

</configuration>
```


## MDC 与跟踪


```
// ========== MDC (Mapped Diagnostic Context) ==========
// 在日志中关联请求上下文
// 如: 请求 ID, 用户 ID, 会话 ID

// ========== 1. 拦截器设置 MDC ==========
@Component
public class MdcFilter implements Filter {

    @Override
    public void doFilter(ServletRequest request, ServletResponse response,
                         FilterChain chain) throws IOException, ServletException {

        try {
            // 生成请求 ID 并放入 MDC
            String requestId = UUID.randomUUID().toString().substring(0, 8);
            MDC.put("requestId", requestId);

            // 如果有用户认证信息
            Authentication auth = SecurityContextHolder.getContext().getAuthentication();
            if (auth != null) {
                MDC.put("userId", auth.getName());
            }

            chain.doFilter(request, response);
        } finally {
            // 必须清理 MDC!
            MDC.clear();
        }
    }
}

// ========== 2. 日志配置中引用 MDC ==========
// pattern 中使用 %X{key}
%X{requestId}  → 输出 MDC 中的 requestId
%X{userId}     → 输出 MDC 中的 userId

// 完整模式:
"%d{yyyy-MM-dd HH:mm:ss} [%thread] %-5level [%X{requestId}] %logger{36} - %msg%n"

// 输出示例:
// 2026-04-29 14:30:00 [http-nio-8080] INFO [a1b2c3d4] com.example.UserService - 创建用户成功

// ========== 3. Spring Boot 3.x 自动跟踪 ID ==========
// 添加依赖: micrometer-tracing
// <dependency>
//     <groupId>io.micrometer</groupId>
//     <artifactId>micrometer-tracing-bridge-brave</artifactId>
// </dependency>

// 配置文件:
logging:
  pattern:
    level: "%5p [%X{traceId:-},%X{spanId:-}]"
// 输出: INFO [traceId,spanId]

// ========== 4. 异步任务 MDC 传递 ==========
// 异步线程 MDC 不会自动传递
// 需要手动包装:

@Bean
public Executor taskExecutor() {
    ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
    executor.setTaskDecorator(runnable -> {
        Map<String, String> contextMap = MDC.getCopyOfContextMap();
        return () -> {
            try {
                MDC.setContextMap(contextMap);
                runnable.run();
            } finally {
                MDC.clear();
            }
        };
    });
    return executor;
}
```


## 日志最佳实践


```
// ========== 日志最佳实践 ==========

// ========== 1. 使用 SLF4J API ==========
// 不直接依赖 Logback/Log4j
// 保持实现可替换

// ========== 2. 使用 {} 占位符 ==========
// log.info("User: {}", name);          // ✅ 正确
// log.info("User: " + name);           // ❌ 字符串拼接

// ========== 3. 级别使用指南 ==========
// ERROR — 需要立即处理的错误 (系统不可用, 操作失败)
// WARN  — 潜在问题 (配置降级, 重试, 不常见情况)
// INFO  — 重要业务事件 (创建/删除/启动/停止)
// DEBUG — 开发调试信息 (SQL, 请求参数, 状态变化)
// TRACE — 最详细跟踪 (循环内, 非常频繁)

// ========== 4. 避免重复日志 ==========
// ❌ 错误:
try {
    // ...
} catch (Exception e) {
    log.error("操作失败", e);              // 上层又抛异常
    throw new RuntimeException("操作失败", e); // 上层又记录一次!
}

// ✅ 正确: 在最终捕获处记录
// 或抛出自定义异常, 由全局异常处理器统一记录

// ========== 5. 保护敏感信息 ==========
// ❌ log.info("用户密码: {}", password);
// ✅ log.info("用户认证成功");
// 不记录: 密码, 令牌, 支付信息, 个人身份信息

// ========== 6. 记录关键上下文 ==========
// 包含足够的上下文信息以便排查
log.warn("支付失败 userId={} orderId={} reason={}",
    userId, orderId, errorReason);

// 不够:
log.warn("支付失败");  // 无法排查!

// ========== 7. 使用 MDC 关联请求 ==========
// 将请求 ID/用户 ID 自动注入到所有日志

// ========== 8. 切面日志 ==========
// 使用 AOP 统一记录 Controller 日志
@Aspect
@Component
public class LoggingAspect {

    @Before("execution(* com.example..*Controller.*(..))")
    public void logRequest(JoinPoint joinPoint) {
        log.info("方法: {} 参数: {}",
            joinPoint.getSignature().toShortString(),
            joinPoint.getArgs());
    }
}

// ========== 9. 条件日志 ==========
// 高频日志增加采样
if (log.isDebugEnabled()) {           // 避免不必要的参数计算
    log.debug("处理大数据: {}", computeExpensive());
}

// ========== 10. 统一异常日志 ==========
// 使用 @ControllerAdvice 统一记录异常
@ControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(Exception.class)
    public ResponseEntity<?> handleException(Exception e, HttpServletRequest request) {
        log.error("请求 {} 异常", request.getRequestURI(), e);
        return ResponseEntity.internalServerError().body("系统异常");
    }
}
```


> **Note:** 💡 日志要点: Spring Boot 默认 SLF4J + Logback; 级别 TRACE < DEBUG < INFO < WARN < ERROR; {} 占位符避免拼接; application.yml 配置 logging.level/pattern/file; logback-spring.xml 高级配置; MDC 关联请求 ID; AsyncAppender 高性能; 保护敏感信息; 记录关键上下文便于排查。


## 练习


<!-- Converted from: 3_Spring Boot 日志.html -->
