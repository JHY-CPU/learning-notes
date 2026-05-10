# Spring Boot 数据源与连接池


## 🗄️ Spring Boot 数据源与连接池


HikariCP 连接池、DataSource 配置、多数据源、JdbcTemplate 使用、事务管理、测试数据源。


## HikariCP 连接池


```
// ========== HikariCP ==========
// Spring Boot 默认连接池 (Hi·ka·ri = "光" in Japanese)
// 性能最好: 最快、最轻量 (约 150KB)
// 零额外开销: 字节码优化

// ========== 自动配置 ==========
// 引入 spring-boot-starter-data-jpa 或 spring-boot-starter-jdbc
// 自动配置 HikariCP + DataSource

// ========== 核心配置 ==========
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb?useSSL=false&serverTimezone=Asia/Shanghai
    username: root
    password: ${DB_PASSWORD}
    driver-class-name: com.mysql.cj.jdbc.Driver

    # HikariCP 特定配置
    hikari:
      # 连接池大小
      maximum-pool-size: 20            # 最大连接数 (默认 10)
      minimum-idle: 5                  # 最小空闲连接 (默认 10)
      connection-timeout: 30000        # 等待连接超时 (毫秒, 默认 30000)
      idle-timeout: 600000             # 连接最大空闲时间 (毫秒, 默认 600000)
      max-lifetime: 1800000            # 连接最大存活时间 (毫秒, 默认 1800000)
      pool-name: MyHikariPool          # 连接池名称 (方便监控)

      # 连接验证
      connection-test-query: SELECT 1  # 连接测试 SQL (H2 不需要)
      validation-timeout: 5000         # 验证超时

      # 性能
      read-only: false                 # 是否只读
      auto-commit: true                # 自动提交

// ========== HikariCP 配置详解 ==========
// maximum-pool-size: 连接池最大连接数
//   公式: T = C * (N + 1)
//     C = 单连接处理请求数/秒
//     N = CPU 核数
//   估算: 4核CPU, 单连接处理100请求/秒, 需要200请求/秒
//   → pool size = 200/100 * (4 + 1) = 10

// minimum-idle: 最小空闲连接数
//   保持一些连接随时可用, 减少建立连接延迟

// connection-timeout: 获取连接超时
//   30000ms = 30秒, 超过抛 SQLException

// max-lifetime: 连接最大存活时间
//   小于数据库的连接超时时间
//   MySQL wait_timeout = 28800 (8小时)
//   HikariCP max-lifetime = 1800000 (30分钟)

// ========== 监控 HikariCP ==========
// 启用 Actuator 查看连接池指标
// /actuator/health — 连接池健康状态
// /actuator/metrics/hikaricp.connections.active — 活跃连接数

// 自定义监控:
@Slf4j
@Component
public class HikariPoolMonitor {

    private final DataSource dataSource;

    public HikariPoolMonitor(DataSource dataSource) {
        this.dataSource = dataSource;
        schedulePoolLogging();
    }

    @Scheduled(fixedRate = 60000)  // 每分钟输出
    public void logPoolStats() {
        if (dataSource instanceof HikariDataSource hikari) {
            HikariPoolMXBean pool = hikari.getHikariPoolMXBean();
            log.info("连接池 {} - 活跃: {}, 空闲: {}, 等待: {}, 总: {}",
                hikari.getPoolName(),
                pool.getActiveConnections(),
                pool.getIdleConnections(),
                pool.getThreadsAwaitingConnection(),
                pool.getTotalConnections());
        }
    }
}
```


## DataSource 配置


```
// ========== 常见数据库配置 ==========

// ========== MySQL ==========
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb?useUnicode=true&characterEncoding=utf8&useSSL=false&serverTimezone=Asia/Shanghai&allowPublicKeyRetrieval=true
    username: root
    password: ${MYSQL_PASSWORD}
    driver-class-name: com.mysql.cj.jdbc.Driver

// ========== PostgreSQL ==========
spring:
  datasource:
    url: jdbc:postgresql://localhost:5432/mydb
    username: postgres
    password: ${PG_PASSWORD}
    driver-class-name: org.postgresql.Driver

// ========== H2 (内存/文件) ==========
spring:
  datasource:
    url: jdbc:h2:mem:testdb               # 内存模式 (重启丢失)
    # url: jdbc:h2:file:./data/testdb     # 文件模式
    driver-class-name: org.h2.Driver
    username: sa
    password:
  h2:
    console:
      enabled: true                        # 开启 H2 控制台
      path: /h2-console                    # http://localhost:8080/h2-console

// ========== PostgreSQL + Docket ==========
// Docker 快速启动测试数据库:
// docker run -d \
//   --name postgres-test \
//   -e POSTGRES_DB=mydb \
//   -e POSTGRES_USER=test \
//   -e POSTGRES_PASSWORD=test \
//   -p 5432:5432 \
//   postgres:16

// ========== 编程式创建 DataSource ==========
@Configuration
public class DataSourceConfig {

    @Bean
    @ConfigurationProperties(prefix = "spring.datasource")
    public DataSource dataSource() {
        return DataSourceBuilder.create().build();  // 自动使用 HikariCP
    }
}

// ========== 使用嵌入式数据库 ==========
// 自动检测 classpath 中的 H2/HSQL/Derby
// 无需配置 url, Spring Boot 自动创建

// 依赖 H2:
// <dependency>
//     <groupId>com.h2database</groupId>
//     <artifactId>h2</artifactId>
//     <scope>runtime</scope>
// </dependency>

// 零配置即可使用!
```


## JdbcTemplate


```
// ========== JdbcTemplate ==========
// Spring JDBC 核心, 简化 JDBC 操作
// 自动管理连接、预处理语句、结果集
// 比原始 JDBC 简洁, 比 JPA 轻量

// ========== 自动注入 ==========
@Slf4j
@Service
public class UserDao {

    private final JdbcTemplate jdbcTemplate;

    public UserDao(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    // ========== 查询 ==========
    public List<User> findAll() {
        return jdbcTemplate.query(
            "SELECT * FROM users",
            new BeanPropertyRowMapper<>(User.class)  // 自动映射
        );
    }

    public Optional<User> findById(Long id) {
        List<User> users = jdbcTemplate.query(
            "SELECT * FROM users WHERE id = ?",
            new BeanPropertyRowMapper<>(User.class),
            id
        );
        return users.stream().findFirst();
    }

    public List<String> findNamesByAge(int age) {
        return jdbcTemplate.queryForList(
            "SELECT name FROM users WHERE age > ?",
            String.class, age
        );
    }

    // ========== 单值查询 ==========
    public int count() {
        return jdbcTemplate.queryForObject(
            "SELECT COUNT(*) FROM users", Integer.class
        );
    }

    // ========== 插入 ==========
    public int insert(User user) {
        return jdbcTemplate.update(
            "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
            user.getName(), user.getEmail(), user.getAge()
        );
    }

    // ========== 更新 ==========
    public int updateEmail(Long id, String email) {
        return jdbcTemplate.update(
            "UPDATE users SET email = ? WHERE id = ?",
            email, id
        );
    }

    // ========== 删除 ==========
    public int deleteById(Long id) {
        return jdbcTemplate.update(
            "DELETE FROM users WHERE id = ?", id
        );
    }

    // ========== 批量操作 ==========
    public int[] batchInsert(List<User> users) {
        return jdbcTemplate.batchUpdate(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            new BatchPreparedStatementSetter() {
                @Override
                public void setValues(PreparedStatement ps, int i) throws SQLException {
                    ps.setString(1, users.get(i).getName());
                    ps.setString(2, users.get(i).getEmail());
                }

                @Override
                public int getBatchSize() {
                    return users.size();
                }
            }
        );
    }

    // ========== NamedParameterJdbcTemplate ==========
    // 使用命名参数而非 ?
    public Optional<User> findByNameAndEmail(String name, String email) {
        var named = new NamedParameterJdbcTemplate(jdbcTemplate);
        Map<String, Object> params = Map.of(
            "name", name,
            "email", email
        );
        List<User> result = named.query(
            "SELECT * FROM users WHERE name = :name AND email = :email",
            params,
            new BeanPropertyRowMapper<>(User.class)
        );
        return result.stream().findFirst();
    }
}
```


## 多数据源配置


```
// ========== 多数据源 ==========
// 同时连接多个数据库

// ========== 1. 配置文件 ==========
spring:
  datasource:
    primary:
      url: jdbc:mysql://localhost:3306/maindb
      username: root
      password: ${DB_PASSWORD}
      driver-class-name: com.mysql.cj.jdbc.Driver

    secondary:
      url: jdbc:postgresql://localhost:5432/analytics
      username: postgres
      password: ${PG_PASSWORD}
      driver-class-name: org.postgresql.Driver

// ========== 2. 配置类 ==========
@Configuration
public class DataSourceConfig {

    @Primary
    @Bean(name = "primaryDataSource")
    @ConfigurationProperties(prefix = "spring.datasource.primary")
    public DataSource primaryDataSource() {
        return DataSourceBuilder.create().build();
    }

    @Bean(name = "secondaryDataSource")
    @ConfigurationProperties(prefix = "spring.datasource.secondary")
    public DataSource secondaryDataSource() {
        return DataSourceBuilder.create().build();
    }

    @Primary
    @Bean(name = "primaryJdbcTemplate")
    public JdbcTemplate primaryJdbcTemplate(
            @Qualifier("primaryDataSource") DataSource ds) {
        return new JdbcTemplate(ds);
    }

    @Bean(name = "secondaryJdbcTemplate")
    public JdbcTemplate secondaryJdbcTemplate(
            @Qualifier("secondaryDataSource") DataSource ds) {
        return new JdbcTemplate(ds);
    }
}

// ========== 3. 使用 ==========
@Service
public class ReportService {

    private final JdbcTemplate primaryJdbcTemplate;
    private final JdbcTemplate secondaryJdbcTemplate;

    public ReportService(
            @Qualifier("primaryJdbcTemplate") JdbcTemplate primary,
            @Qualifier("secondaryJdbcTemplate") JdbcTemplate secondary) {
        this.primaryJdbcTemplate = primary;
        this.secondaryJdbcTemplate = secondary;
    }

    public void generateReport() {
        // 从主库读取
        List<User> users = primaryJdbcTemplate.query(
            "SELECT * FROM users", new BeanPropertyRowMapper<>(User.class));

        // 写入分析库
        secondaryJdbcTemplate.update(
            "INSERT INTO report_log (data, created_at) VALUES (?, NOW())",
            users.toString());
    }
}
```


## 事务与测试数据源


```
// ========== 事务管理 ==========

// ========== 声明式事务 ==========
@Service
@Transactional                          // 类级别: 所有方法启用事务
public class OrderService {

    private final JdbcTemplate jdbcTemplate;

    // 方法级别覆盖
    @Transactional(readOnly = true)     // 只读事务 (性能优化)
    public Order findById(Long id) { ... }

    @Transactional(rollbackFor = Exception.class)  // 所有异常都回滚
    public void createOrder(Order order) {
        // 多个数据库操作在同一个事务中
        jdbcTemplate.update("INSERT INTO orders ...");
        jdbcTemplate.update("UPDATE inventory SET stock = stock - 1 ...");
        // 任何异常都会回滚所有操作
    }

    @Transactional(timeout = 30)        // 超时 30 秒
    public void processPayment(Payment payment) { ... }

    @Transactional(propagation = Propagation.REQUIRES_NEW)  // 新事务
    public void auditLog(String action) { ... }
}

// ========== 编程式事务 ==========
@Autowired
private TransactionTemplate transactionTemplate;

public void transfer(Long fromId, Long toId, BigDecimal amount) {
    transactionTemplate.execute(status -> {
        try {
            jdbcTemplate.update("UPDATE accounts SET balance = balance - ? WHERE id = ?",
                amount, fromId);
            jdbcTemplate.update("UPDATE accounts SET balance = balance + ? WHERE id = ?",
                amount, toId);
            return null;
        } catch (Exception e) {
            status.setRollbackOnly();   // 手动回滚
            throw e;
        }
    });
}

// ========== 测试数据源 ==========
// 使用 Testcontainers 集成测试
@SpringBootTest
@Testcontainers
class UserDaoTest {

    @Container
    static MySQLContainer<?> mysql = new MySQLContainer<>("mysql:8.0")
        .withDatabaseName("testdb");

    @DynamicPropertySource
    static void configureProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", mysql::getJdbcUrl);
        registry.add("spring.datasource.username", mysql::getUsername);
        registry.add("spring.datasource.password", mysql::getPassword);
    }

    @Autowired
    private UserDao userDao;

    @Test
    void shouldInsertAndFindUser() {
        userDao.insert(new User("Alice", "alice@test.com"));
        List<User> users = userDao.findAll();
        assertThat(users).hasSize(1);
        assertThat(users.get(0).getName()).isEqualTo("Alice");
    }
}

// ========== 使用 H2 测试 ==========
@SpringBootTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.ANY)  // 自动替换为 H2
class UserDaoTest {
    // 无需额外配置, 自动使用 H2 内存数据库
}

// 自定义测试配置:
@TestPropertySource(properties = {
    "spring.datasource.url=jdbc:h2:mem:testdb;MODE=MySQL;DB_CLOSE_DELAY=-1",
    "spring.datasource.driver-class-name=org.h2.Driver",
    "spring.jpa.database-platform=org.hibernate.dialect.H2Dialect"
})
```


> **Note:** 💡 数据源要点: HikariCP 默认连接池 (快速轻量); DataSourceBuilder 快速创建; JdbcTemplate 简化 JDBC; @Transactional 声明式事务; @Primary/@Qualifier 多数据源; connection-test-query 验证连接; maximum-pool-size 根据 CPU 计算; 测试用 @AutoConfigureTestDatabase 自动切换 H2; Testcontainers 真实数据库集成测试。


## 练习


<!-- Converted from: 6_Spring Boot 数据源与连接池.html -->
