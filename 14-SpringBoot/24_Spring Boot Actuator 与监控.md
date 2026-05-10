# Spring Boot Actuator 与监控


## 📊 Spring Boot Actuator 与监控


Actuator 端点 (health/metrics/info/env)、自定义 HealthIndicator、Metrics 指标、Micrometer 集成 Prometheus、动态日志级别、应用信息与版本。


## Actuator 入门


```
// ========== Spring Boot Actuator ==========
// 生产级监控和管理端点
// 提供: 健康检查/指标/环境信息/日志/线程转储等

// ========== 依赖 ==========
// <dependency>
//     <groupId>org.springframework.boot</groupId>
//     <artifactId>spring-boot-starter-actuator</artifactId>
// </dependency>

// ========== 配置端点 ==========
// application.yml:

management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics,env,loggers,threaddump,heapdump,conditions,mappings,configprops,beans
        # 包含所有: include: "*"
        # 排除: exclude: env,beans
      base-path: /actuator                  # 默认 /actuator
      cors:
        allowed-origins: http://localhost:3000
        allowed-methods: GET,POST
  endpoint:
    health:
      show-details: always                 # 显示详细健康信息 (默认 never)
      show-components: always              # 显示组件状态
    info:
      enabled: true
    env:
      show-values: when-authorized         # 环境变量显示策略
    configprops:
      show-values: when-authorized
    shutdown:
      enabled: false                       # 关闭端点默认禁用 (危险)

// ========== 常用端点 ==========
// GET  /actuator/health        — 健康检查
// GET  /actuator/info          — 应用信息
// GET  /actuator/metrics       — 指标列表
// GET  /actuator/metrics/{name} — 特定指标详情
// GET  /actuator/env           — 环境属性
// GET  /actuator/env/{name}    — 特定属性
// GET  /actuator/loggers       — 日志级别
// POST /actuator/loggers/{name} — 修改日志级别
// GET  /actuator/conditions    — 自动配置条件
// GET  /actuator/beans         — 所有 Bean
// GET  /actuator/mappings      — 请求映射
// GET  /actuator/threaddump    — 线程转储
// GET  /actuator/heapdump      — 堆转储 (.hprof)
// POST /actuator/shutdown      — 关闭应用

// ========== 端点 URL 示例 ==========
// http://localhost:8080/actuator/health
// {
//   "status": "UP",
//   "components": {
//     "db": { "status": "UP", "database": "MySQL", "validationQuery": "isValid()" },
//     "redis": { "status": "UP" },
//     "diskSpace": { "status": "UP", "total": 500107862016, "free": 320182329344 },
//     "ping": { "status": "UP" }
//   }
// }
```


## 自定义 HealthIndicator


```
// ========== 健康检查 ==========
// Actuator 聚合多个 HealthIndicator 判断应用状态

// ========== 内置 HealthIndicator ==========
// DiskSpaceHealthIndicator      — 磁盘空间
// DataSourceHealthIndicator     — 数据源
// RedisHealthIndicator          — Redis
// MongoHealthIndicator          — MongoDB
// ElasticsearchHealthIndicator  — ES
// RabbitHealthIndicator         — RabbitMQ
// CassandraHealthIndicator      — Cassandra

// ========== 自定义 HealthIndicator ==========
// 检查外部 API 是否可用

@Component
public class ExternalApiHealthIndicator implements HealthIndicator {

    private final RestTemplate restTemplate;

    @Override
    public Health health() {
        try {
            ResponseEntity<String> response = restTemplate
                .getForEntity("https://api.example.com/health", String.class);

            if (response.getStatusCode().is2xxSuccessful()) {
                return Health.up()
                    .withDetail("url", "https://api.example.com")
                    .withDetail("statusCode", response.getStatusCodeValue())
                    .withDetail("responseTime", "50ms")
                    .build();
            } else {
                return Health.down()
                    .withDetail("url", "https://api.example.com")
                    .withDetail("statusCode", response.getStatusCodeValue())
                    .build();
            }
        } catch (Exception e) {
            return Health.down(e)
                .withDetail("url", "https://api.example.com")
                .withDetail("error", e.getMessage())
                .build();
        }
    }
}

// ========== 详细 HealthIndicator ==========
// 检查关键业务指标

@Component
public class BusinessHealthIndicator implements HealthIndicator {

    private final UserRepository userRepository;
    private final OrderRepository orderRepository;

    @Override
    public Health health() {
        try {
            long userCount = userRepository.count();
            long orderCount = orderRepository.count();

            return Health.up()
                .withDetail("totalUsers", userCount)
                .withDetail("totalOrders", orderCount)
                .withDetail("databaseStatus", "connected")
                .build();
        } catch (Exception e) {
            return Health.down(e)
                .withDetail("databaseStatus", "disconnected")
                .build();
        }
    }
}

// ========== 健康状态权重 ==========
// 某些服务不可用影响不同
// 自定义 Status 聚合规则:

@Component
public class CustomHealthAggregator implements HealthAggregator {

    @Override
    public Health aggregate(Map<String, Health> healths) {
        for (Health health : healths.values()) {
            // 任何服务 DOWN, 整体 DOWN
            if (Status.DOWN.equals(health.getStatus())) {
                return Health.down().build();
            }
            // 如果有服务 OUT_OF_SERVICE
            if (Status.OUT_OF_SERVICE.equals(health.getStatus())) {
                return Health.outOfService().build();
            }
        }
        return Health.up().build();
    }
}

// ========== 启用所有健康详情 ==========
// management.endpoint.health.show-details=always
// 或:
// management.endpoint.health.show-details=when-authorized
// management.endpoint.health.roles=ADMIN
```


## Info 端点与自定义信息


```
// ========== Info 端点 ==========
// 显示应用构建信息、Git 提交、Java 版本等

// ========== 配置静态信息 ==========
// application.yml:

info:
  app:
    name: '@project.name@'                # 引用 Maven 属性
    version: '@project.version@'
    description: '@project.description@'
  java:
    version: '@java.version@'
  build:
    artifact: '@project.artifactId@'
    groupId: '@project.groupId@'
  contact:
    email: dev@example.com
    team: backend-team

// ========== Git 信息 ==========
// 添加插件 (在 pom.xml):
// <plugin>
//     <groupId>pl.project13.maven</groupId>
//     <artifactId>git-commit-id-plugin</artifactId>
//     <configuration>
//         <generateGitPropertiesFile>true</generateGitPropertiesFile>
//     </configuration>
// </plugin>
// 自动生成 git.properties, info 端点显示:
// info.git.commit.id, info.git.branch, info.git.commit.time

// ========== 自定义 InfoContributor ==========
@Component
public class CustomInfoContributor implements InfoContributor {

    @Override
    public void contribute(Info.Builder builder) {
        builder
            .withDetail("deployTime", LocalDateTime.now().toString())
            .withDetail("activeProfile",
                String.join(", ",
                    environment.getActiveProfiles()))
            .withDetail("memory",
                Map.of("max", Runtime.getRuntime().maxMemory(),
                       "total", Runtime.getRuntime().totalMemory(),
                       "free", Runtime.getRuntime().freeMemory()))
            .withDetail("systemInfo",
                Map.of("os", System.getProperty("os.name"),
                       "cpu", Runtime.getRuntime().availableProcessors()));
    }
}

// ========== 动态修改日志级别 ==========
// 运行时调整日志级别, 无需重启

// 查看日志级别:
// GET /actuator/loggers
// GET /actuator/loggers/com.example

// 修改日志级别 (POST):
// POST /actuator/loggers/com.example
// Content-Type: application/json
// {
//   "configuredLevel": "DEBUG"
// }

// 实现动态修改:
@RestController
@RequestMapping("/api/admin")
public class LogLevelController {

    private final LoggersEndpoint loggersEndpoint;

    // 通过 Actuator 端点的 LoggersEndpoint 编程修改
    @PutMapping("/log-level")
    public ResponseEntity<Void> setLogLevel(
            @RequestParam String packageName,
            @RequestParam String level) {
        LoggersEndpoint.LoggerLevels levels =
            new LoggersEndpoint.LoggerLevels(null, LogLevel.valueOf(level));
        loggersEndpoint.configureLogLevel(packageName, levels);
        return ResponseEntity.ok().build();
    }
}
```


## Micrometer 与 Metrics


```
// ========== Micrometer ==========
// Spring Boot Actuator 的指标系统
// 支持: Prometheus / Graphite / Datadog / Influx / JMX

// ========== 核心指标 ==========
// /actuator/metrics 列出所有指标:
// jvm.memory.used         — JVM 内存使用
// jvm.memory.max          — JVM 最大内存
// jvm.gc.pause            — GC 暂停时间
// jvm.threads.live        — 活跃线程数
// system.cpu.usage        — CPU 使用率
// process.cpu.usage       — 进程 CPU
// http.server.requests    — HTTP 请求统计
// jdbc.connections.active — 活跃数据库连接
// logback.events          — 日志级别统计

// ========== Prometheus 集成 ==========
// 依赖:
// <dependency>
//     <groupId>io.micrometer</groupId>
//     <artifactId>micrometer-registry-prometheus</artifactId>
// </dependency>

// 配置暴露 Prometheus 端点:
// management.endpoints.web.exposure.include=prometheus
// 或包含所有: include=*

// Prometheus 抓取端点:
// GET /actuator/prometheus
// 返回 Prometheus 格式文本:
// # HELP jvm_memory_used_bytes The amount of used memory
// # TYPE jvm_memory_used_bytes gauge
// jvm_memory_used_bytes{area="heap",id="G1 Eden Space",} 2.097152E7

// ========== 自定义 Metrics ==========

@Service
public class OrderMetricsService {

    private final MeterRegistry meterRegistry;

    // 计数器: 订单创建数
    private final Counter orderCreatedCounter;

    // 计时器: 订单处理时间
    private final Timer orderProcessingTimer;

    // 仪表: 待处理订单数
    private final AtomicInteger pendingOrders;

    public OrderMetricsService(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;

        this.orderCreatedCounter = Counter.builder("orders.created.total")
            .description("订单创建总数")
            .tag("type", "all")
            .register(meterRegistry);

        this.orderProcessingTimer = Timer.builder("orders.processing.duration")
            .description("订单处理耗时")
            .publishPercentiles(0.5, 0.95, 0.99)    // P50, P95, P99
            .register(meterRegistry);

        this.pendingOrders = meterRegistry.gauge(
            "orders.pending.count",
            new AtomicInteger(0));
    }

    // 记录订单创建
    public void recordOrderCreated(String status) {
        orderCreatedCounter.increment();
        meterRegistry.counter("orders.created.total",
            "status", status).increment();
    }

    // 记录处理时间
    public void recordProcessingTime(Runnable task) {
        orderProcessingTimer.record(task);
    }

    // 更新待处理数
    public void setPendingOrders(int count) {
        pendingOrders.set(count);
    }

    // 记录自定义分布
    public void recordOrderAmount(double amount, String currency) {
        DistributionSummary summary = DistributionSummary
            .builder("orders.amount")
            .description("订单金额分布")
            .baseUnit("CNY")
            .publishPercentiles(0.5, 0.9, 0.99)
            .tag("currency", currency)
            .register(meterRegistry);
        summary.record(amount);
    }
}

// ========== @Timed 注解 ==========
// AOP 方式自动计时

@RestController
@RequiredArgsConstructor
@Timed                                    // 类级别
public class OrderController {

    @Timed(value = "orders.create",       // 方法级别覆盖
          percentiles = {0.5, 0.95},
          description = "创建订单耗时")
    @PostMapping("/orders")
    public ResponseEntity<Order> createOrder(@RequestBody CreateOrderRequest req) {
        // 自动记录方法执行时间
        return ResponseEntity.ok(orderService.createOrder(req));
    }
}
```


## 最佳实践


```
// ========== Actuator 最佳实践 ==========

// ========== 1. 安全加固 ==========
// Actuator 端点暴露敏感信息, 必须保护

// 方式 1: 独立端口 (推荐)
// management.server.port=8081
// management.server.address=127.0.0.1

// 方式 2: Spring Security 保护
@Configuration
@Order(1)
public class ActuatorSecurityConfig {

    @Bean
    public SecurityFilterChain actuatorFilterChain(HttpSecurity http) throws Exception {
        http
            .securityMatcher("/actuator/**")
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/actuator/health").permitAll()   // 健康检查开放
                .requestMatchers("/actuator/info").permitAll()
                .anyRequest().hasRole("ADMIN")                     // 其他需 ADMIN
            )
            .httpBasic(Customizer.withDefaults());
        return http.build();
    }
}

// 方式 3: 关闭敏感端点
// management.endpoint.shutdown.enabled=false
// management.endpoint.env.enabled=false

// ========== 2. 健康检查定制 ==========
// 配置 K8s 就绪和存活探针
// management.endpoint.health.probes.enabled=true

// Kubernetes liveness: /actuator/health/liveness
// Kubernetes readiness: /actuator/health/readiness

// ========== 3. Prometheus + Grafana ==========
// 集成 Prometheus 抓取 /actuator/prometheus
// Grafana 导入 Spring Boot 仪表盘 (ID: 4701)

// ========== 4. 关键指标 ==========
// 监控重点:
// - 堆内存使用 (OOM 预警)
// - GC 频率和时间 (GC 问题)
// - HTTP 请求延迟 P99 (性能)
// - 错误率 (异常增加)
// - 线程状态 (死锁/阻塞)
// - 数据库连接池 (连接耗尽)

// ========== 5. 日志管理 ==========
// 通过 /actuator/loggers 动态调整日志级别
// 生产环境默认 INFO
// 排查问题时临时降级到 DEBUG

// ========== 6. 端点访问控制 ==========
// 启用认证 + HTTPS
// 独立管理端口
// 内网隔离

// ========== 7. 健康检查自定义 ==========
// 区分关键依赖和非关键依赖
// 关键依赖 DOWN → 整体 DOWN
// 非关键依赖 DOWN → 整体 UP 但告警

// ========== 8. 监控告警 ==========
// Prometheus + Alertmanager
// 配置告警规则:
// - 服务不可用 (health=DOWN > 1m)
// - 高错误率 (http 5xx > 1%)
// - 高延迟 (P99 > 500ms)
// - OOM 风险 (堆 > 90%)
```


> **Note:** 💡 Actuator 要点: /actuator/health 健康检查; /actuator/metrics 指标; /actuator/info 应用信息; /actuator/loggers 动态日志; Micrometer + Prometheus 指标; HealthIndicator 自定义健康; @Timed 方法计时; 独立管理端口; Security 保护敏感端点; K8s 存活/就绪探针。


## 练习


<!-- Converted from: 24_Spring Boot Actuator 与监控.html -->
