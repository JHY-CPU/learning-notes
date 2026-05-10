# Spring Data JPA 审计与迁移


## 📋 Spring Data JPA 审计与迁移


@CreatedDate/@LastModifiedDate 审计、@EntityListeners、Flyway 数据库迁移、Liquibase、数据初始化。


## JPA Auditing


```
// ========== JPA Auditing ==========
// 自动填充创建时间/修改时间/创建人/修改人

// ========== 1. 启用审计 ==========
@SpringBootApplication
@EnableJpaAuditing                      // 启用 JPA 审计
public class Application { ... }

// ========== 2. 审计基类 ==========
@MappedSuperclass                       // 父类映射, 子类继承字段
@EntityListeners(AuditingEntityListener.class)  // 审计监听器
@Data
public abstract class BaseEntity {

    @CreatedDate                         // 创建时自动填充
    @Column(updatable = false)
    private LocalDateTime createdAt;

    @LastModifiedDate                    // 每次更新时自动填充
    private LocalDateTime updatedAt;

    @CreatedBy                           // 创建人 (需实现 AuditorAware)
    @Column(updatable = false)
    private String createdBy;

    @LastModifiedBy                      // 修改人
    private String lastModifiedBy;
}

// ========== 3. 实体继承 ==========
@Entity
public class User extends BaseEntity {   // 自动获得审计字段
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String username;
    private String email;
}

// 数据库表自动包含:
// created_at, updated_at, created_by, last_modified_by

// ========== 4. 审计人实现 ==========
@Configuration
public class AuditConfig {

    @Bean
    public AuditorAware<String> auditorAware() {
        // 从 SecurityContext 获取当前用户
        return () -> Optional.ofNullable(SecurityContextHolder.getContext())
            .map(SecurityContext::getAuthentication)
            .filter(Authentication::isAuthenticated)
            .map(Authentication::getName)
            .or(() -> Optional.of("system"));  // 默认系统用户
    }
}

// ========== 5. 使用 ==========
@Service
public class UserService {

    @Transactional
    public User createUser(User user) {
        User saved = userRepository.save(user);
        // createdAt / createdBy 自动填充
        log.info("用户 {} 由 {} 创建于 {}",
            saved.getUsername(), saved.getCreatedBy(), saved.getCreatedAt());
        return saved;
    }
}
```


## @EntityListeners 与回调


```
// ========== JPA 生命周期回调 ==========
// 在实体状态变化时执行自定义逻辑

// ========== 回调注解 ==========
// @PrePersist    — 保存前
// @PostPersist   — 保存后
// @PreUpdate     — 更新前
// @PostUpdate    — 更新后
// @PreRemove     — 删除前
// @PostRemove    — 删除后
// @PostLoad      — 加载后

// ========== 实体内部回调 ==========
@Entity
public class Product {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;
    private BigDecimal price;

    @PrePersist
    public void prePersist() {
        log.info("保存前: {}", name);
        if (price == null) {
            price = BigDecimal.ZERO;
        }
    }

    @PostPersist
    public void postPersist() {
        log.info("保存后, ID: {}", id);
        // 触发索引更新等
    }

    @PreUpdate
    public void preUpdate() {
        log.info("更新前: {}", id);
    }

    @PostLoad
    public void postLoad() {
        log.info("加载后: {}", id);
    }
}

// ========== 外部监听器 ==========
public class ProductEntityListener {

    @PrePersist
    public void onPrePersist(Product product) {
        // 校验逻辑
        if (product.getPrice() != null && product.getPrice().compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("价格不能为负");
        }
    }

    @PostUpdate
    public void onPostUpdate(Product product) {
        // 清除相关缓存
        cacheManager.evict("products", product.getId());
    }
}

// 实体上注册:
@Entity
@EntityListeners(ProductEntityListener.class)
public class Product { ... }
```


## Flyway 数据库迁移


```
// ========== Flyway ==========
// 版本化数据库迁移工具
// 比 ddl-auto 更安全可控
// 替代 ddl-auto=update, 用于生产环境

// ========== 添加依赖 ==========
// Maven:
<dependency>
    <groupId>org.flywaydb</groupId>
    <artifactId>flyway-core</artifactId>
</dependency>
<dependency>
    <groupId>org.flywaydb</groupId>
    <artifactId>flyway-mysql</artifactId>
</dependency>

// ========== 配置 ==========
spring:
  flyway:
    enabled: true                        # 启用 (默认)
    locations: classpath:db/migration    # SQL 文件位置
    baseline-on-migrate: true            # 对已有数据库基线化
    baseline-version: 1                  # 基线版本
    validate-on-migrate: true            # 迁移时校验

// ========== SQL 迁移文件 ==========
// 位置: src/main/resources/db/migration/
// 命名规则: V{版本号}__{描述}.sql
// V1__create_users_table.sql
// V2__add_email_to_users.sql
// V3__create_orders_table.sql

// V1__create_users_table.sql
/*
CREATE TABLE users (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) UNIQUE,
    password VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    created_at DATETIME NOT NULL,
    updated_at DATETIME
);
*/

// V2__add_email_to_users.sql
/*
ALTER TABLE users ADD COLUMN phone VARCHAR(20) AFTER email;
ALTER TABLE users ADD COLUMN age INT AFTER phone;
*/

// V3__create_orders_table.sql
/*
CREATE TABLE orders (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_no VARCHAR(50) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    created_at DATETIME NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
*/

// ========== 常用命令 ==========
// mvn flyway:migrate        — 执行迁移
// mvn flyway:info           — 查看迁移状态
// mvn flyway:validate       — 校验已迁移的 SQL
// mvn flyway:baseline       — 基线化已有数据库
// mvn flyway:repair         — 修复迁移记录

// ========== Flyway vs ddl-auto ==========
// ddl-auto=update: 开发方便, 生产危险
// Flyway: 版本化, 可回滚, 团队协作, 生产安全

// 最佳实践:
// 开发: ddl-auto=update (快速迭代)
// 生产: ddl-auto=none + Flyway (安全可控)
```


## Liquibase 与数据初始化


```
// ========== Liquibase ==========
// 另一种数据库迁移工具
// 支持 XML/YAML/JSON/SQL 多种格式

// ========== 添加依赖 ==========
// <dependency>
//     <groupId>org.liquibase</groupId>
//     <artifactId>liquibase-core</artifactId>
// </dependency>

// ========== 配置 ==========
spring:
  liquibase:
    change-log: classpath:db/changelog/db.changelog-master.xml
    enabled: true

// ========== db.changelog-master.xml ==========
// <databaseChangeLog>
//     <include file="db/changelog/changes/V1_create_users.xml"/>
//     <include file="db/changelog/changes/V2_create_orders.xml"/>
// </databaseChangeLog>

// V1_create_users.xml:
// <changeSet id="1" author="dev">
//     <createTable tableName="users">
//         <column name="id" type="bigint" autoIncrement="true">
//             <constraints primaryKey="true"/>
//         </column>
//         <column name="username" type="varchar(50)">
//             <constraints unique="true"/>
//         </column>
//     </createTable>
// </changeSet>

// ========== 数据初始化 ==========
// 方式 1: data.sql + schema.sql
// 放在 src/main/resources/ 下
// schema.sql — 建表 DDL (需禁用 ddl-auto)
// data.sql   — 初始数据 INSERT

// 配置:
spring:
  sql:
    init:
      mode: always                      # always/embedded/never
      schema-locations: classpath:schema.sql
      data-locations: classpath:data.sql
  jpa:
    defer-datasource-initialization: true  # 让 JPA 先建表, 再执行 data.sql

// data.sql 示例:
// INSERT INTO users (username, email, password) VALUES
// ('admin', 'admin@test.com', '$2a$10$...'),
// ('user1', 'user1@test.com', '$2a$10$...');

// 方式 2: CommandLineRunner / ApplicationRunner
@Component
@Profile("dev")
@RequiredArgsConstructor
public class DataInitializer implements CommandLineRunner {

    private final UserRepository userRepository;

    @Override
    public void run(String... args) {
        if (userRepository.count() > 0) return;  // 已有数据跳过

        User admin = new User("admin", "admin@test.com", passwordEncoder.encode("admin123"));
        admin.setRole(Role.ADMIN);
        userRepository.save(admin);

        log.info("初始化管理员用户");
    }
}
```


## 最佳实践


```
// ========== 数据库迁移最佳实践 ==========

// ========== 1. 生产禁用 ddl-auto ==========
// spring.jpa.hibernate.ddl-auto=none
// 使用 Flyway 或 Liquibase 管理

// ========== 2. 迁移文件不可修改 ==========
// 已执行的迁移文件不能修改内容
// 只能新增迁移文件

// ========== 3. 版本命名规范 ==========
// V1__description.sql
// V1.1__description.sql
// V2__description.sql

// ========== 4. 每次迁移做一件事 ==========
// 一个迁移文件只做一个变更
// 方便排查和回滚

// ========== 5. 审计字段 ==========
// 所有表包含: created_at, updated_at
// 方便问题排查和数据追溯

// ========== 6. 软删除 ==========
// 用 deleted 标记替代物理删除
// @SQLDelete + @Where 实现

@Entity
@SQLDelete(sql = "UPDATE users SET deleted = true WHERE id = ?")
@Where(clause = "deleted = false")
public class User { ... }

// ========== 7. 测试数据初始化 ==========
// 测试用 data.sql 初始化数据
// 或使用 @Sql 注解

@SpringBootTest
@Sql({"/sql/init-users.sql", "/sql/init-orders.sql"})
class OrderServiceTest { ... }
```


> **Note:** 💡 审计与迁移要点: @EnableJpaAuditing 启用; @CreatedDate/@LastModifiedDate 审计字段; @EntityListeners 生命周期回调; Flyway 版本化 SQL 迁移 (V1__xxx.sql); Liquibase 多格式支持; data.sql 初始化数据; 生产用 Flyway 替代 ddl-auto; 软删除 + 审计字段最佳实践。


## 练习


<!-- Converted from: 18_Spring Data JPA 审计与迁移.html -->
