# Spring Data JPA 入门


## 🗄️ Spring Data JPA 入门


JPA/Hibernate 核心概念、@Entity 映射、JpaRepository 接口、CRUD 操作、主键策略、自动 DDL。


## JPA 与 Hibernate


```
// ========== JPA ==========
// Jakarta Persistence API (Java 持久化规范)
// ORM: Object-Relational Mapping (对象关系映射)
// Hibernate: JPA 最流行的实现

// Spring Data JPA: Spring 对 JPA 的封装
//   JPA (规范) ← Hibernate (实现) ← Spring Data JPA (封装)

// ========== 添加依赖 ==========
// Maven:
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>

// 还需数据库驱动:
// <dependency>
//     <groupId>com.mysql</groupId>
//     <artifactId>mysql-connector-j</artifactId>
//     <scope>runtime</scope>
// </dependency>

// ========== 配置 ==========
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb
    username: root
    password: ${DB_PASSWORD}
    driver-class-name: com.mysql.cj.jdbc.Driver
  jpa:
    hibernate:
      ddl-auto: update                 # 自动建表/更新
    show-sql: true                     # 显示 SQL
    properties:
      hibernate:
        format_sql: true               # 格式化 SQL
        dialect: org.hibernate.dialect.MySQLDialect

// ========== ddl-auto 选项 ==========
// none      — 什么也不做 (生产推荐)
// validate  — 验证实体与表结构是否匹配
// update    — 自动更新表结构 (开发)
// create    — 每次启动删除重建
// create-drop — 启动创建, 关闭删除 (测试)
```


## @Entity 映射


```
// ========== 实体映射 ==========

// ========== 基本实体 ==========
@Entity                                   // 声明为 JPA 实体
@Table(name = "users")                    // 对应数据库表名
@Data
@NoArgsConstructor
@AllArgsConstructor
public class User {

    @Id                                   // 主键
    @GeneratedValue(strategy = GenerationType.IDENTITY)  // 自增
    private Long id;

    @Column(name = "username", nullable = false, unique = true, length = 50)
    private String username;

    @Column(nullable = false)             // 列名默认 = 字段名
    private String password;

    @Column(unique = true)
    private String email;

    @Column(name = "display_name")
    private String displayName;

    @Column(length = 20)
    private String phone;

    @Column(nullable = false)
    private Integer age;

    @Enumerated(EnumType.STRING)          // 枚举映射 (存字符串而非序号)
    private UserStatus status;

    @Column(updatable = false)            // 创建后不可更新
    private LocalDateTime createdAt;

    @Column(insertable = false)           // 由数据库维护
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updatedAt;
}

// ========== @GeneratedValue 主键策略 ==========
// IDENTITY     — 数据库自增 (MySQL, SQL Server)
// SEQUENCE     — 序列 (PostgreSQL, Oracle) [性能好]
// TABLE        — 表模拟 (效率低)
// AUTO         — 自动选择 (默认)

// IDENTITY 示例:
@Id
@GeneratedValue(strategy = GenerationType.IDENTITY)
private Long id;

// SEQUENCE 示例 (PostgreSQL 推荐):
@Id
@GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "user_seq")
@SequenceGenerator(name = "user_seq", sequenceName = "user_sequence", allocationSize = 50)
private Long id;

// UUID 主键:
@Id
@GeneratedValue(strategy = GenerationType.UUID)
private String id;

// ========== @Column 常用属性 ==========
@Column(
    name = "column_name",    // 列名
    nullable = false,        // 非空
    unique = true,           // 唯一
    length = 100,            // 字符串长度 (默认 255)
    precision = 10,          // 数字总位数
    scale = 2,               // 小数位数
    updatable = false,       // 不可更新
    insertable = false,      // 不可插入
    columnDefinition = "TEXT" // 自定义 DDL
)

// ========== @Transient ==========
@Transient                                // 不映射到数据库
private String confirmPassword;
```


## JpaRepository


```
// ========== Repository 接口 ==========
// 继承 JpaRepository 获得 CRUD 方法

public interface UserRepository extends JpaRepository<User, Long> {
    // JpaRepository<实体类型, 主键类型>
    // 自动提供: save, findById, findAll, count, deleteById, existsById ...
}

// ========== 继承体系 ==========
// Repository (标记接口)
//   └─ CrudRepository (CRUD 方法)
//        └─ PagingAndSortingRepository (分页+排序)
//             └─ JpaRepository (JPA 特定方法)

// ========== 使用 Repository ==========
@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;

    // ========== CRUD ==========
    public User save(User user) {
        return userRepository.save(user);           // 新增或更新
    }

    public List<User> saveAll(List<User> users) {
        return userRepository.saveAll(users);       // 批量保存
    }

    public Optional<User> findById(Long id) {
        return userRepository.findById(id);         // 按主键查
    }

    public List<User> findAll() {
        return userRepository.findAll();            // 查询全部
    }

    public List<User> findAllById(List<Long> ids) {
        return userRepository.findAllById(ids);     // 批量查
    }

    public boolean existsById(Long id) {
        return userRepository.existsById(id);       // 是否存在
    }

    public long count() {
        return userRepository.count();              // 总数
    }

    public void deleteById(Long id) {
        userRepository.deleteById(id);               // 按主键删
    }

    public void delete(User user) {
        userRepository.delete(user);                 // 删实体
    }

    public void deleteAll() {
        userRepository.deleteAll();                  // 删全部
    }

    // ========== getReferenceById (懒加载) ==========
    public User getReference(Long id) {
        return userRepository.getReferenceById(id); // 只获取引用, 不查询
    }
}

// ========== 分页排序 ==========
@Service
public class ProductService {

    private final ProductRepository productRepository;

    public Page<Product> findPage(int page, int size, String sortBy) {
        Pageable pageable = PageRequest.of(page, size, Sort.by(sortBy).descending());
        return productRepository.findAll(pageable);
        // 返回: content + totalElements + totalPages + ...
    }

    public Slice<Product> findSlice(int page, int size) {
        Pageable pageable = PageRequest.of(page, size);
        return productRepository.findSliceBy(pageable);  // Slice: 只判断是否有下一页
    }
}
```


## 派生查询方法


```
// ========== 派生查询 ==========
// 按方法命名规则自动生成查询
// 无需写 JPQL/SQL!

public interface UserRepository extends JpaRepository<User, Long> {

    // ========== 精确匹配 ==========
    Optional<User> findByUsername(String username);
    Optional<User> findByEmail(String email);
    List<User> findByStatus(UserStatus status);

    // ========== 条件查询 ==========
    List<User> findByAgeGreaterThan(int age);
    List<User> findByAgeBetween(int min, int max);
    List<User> findByAgeLessThanEqual(int age);

    // ========== 字符串匹配 ==========
    List<User> findByUsernameStartingWith(String prefix);
    List<User> findByUsernameEndingWith(String suffix);
    List<User> findByUsernameContaining(String keyword);  // LIKE %keyword%
    List<User> findByUsernameLike(String pattern);

    // ========== 多条件 ==========
    List<User> findByUsernameAndEmail(String username, String email);
    List<User> findByUsernameOrEmail(String username, String email);
    List<User> findByStatusAndAgeGreaterThan(UserStatus status, int age);

    // ========== IN 查询 ==========
    List<User> findByUsernameIn(List<String> usernames);
    List<User> findByAgeIn(List<Integer> ages);

    // ========== 空判断 ==========
    List<User> findByEmailIsNull();
    List<User> findByEmailIsNotNull();

    // ========== 排序 ==========
    List<User> findByStatusOrderByCreatedAtDesc(UserStatus status);
    List<User> findByAgeGreaterThanOrderByUsernameAsc(int age);

    // ========== 去重 + 限制 ==========
    List<User> findDistinctByStatus(UserStatus status);
    Optional<User> findFirstByOrderByCreatedAtDesc();
    List<User> findTop5ByOrderByCreatedAtDesc();
}

// ========== 关键字参考 ==========
// And / Or / Is / Equals
// Between / LessThan / GreaterThan / LessThanEqual / GreaterThanEqual
// After / Before (日期)
// IsNull / IsNotNull / NotNull
// Like / NotLike / StartingWith / EndingWith / Containing
// In / NotIn
// True / False
// OrderBy / Asc / Desc
// IgnoreCase
// Top / First / Distinct
// Count / Exists / Delete

// ========== 复杂分页 ==========
Page<User> findByStatus(UserStatus status, Pageable pageable);
Slice<User> findByAgeGreaterThan(int age, Pageable pageable);
List<User> findByUsernameContaining(String keyword, Sort sort);
```


## 最佳实践


```
// ========== JPA 最佳实践 ==========

// ========== 1. 谨慎使用 @Data 于实体 ==========
// @EqualsAndHashCode 可能触发懒加载
// 推荐: @Getter @Setter @ToString(onlyExplicitlyIncluded = true)

@Entity
@Getter
@Setter
@ToString(onlyExplicitlyIncluded = true)
@NoArgsConstructor
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ToString.Include
    private String username;
}

// ========== 2. 使用 @Builder 时加 @NoArgsConstructor ==========
@Entity
@Builder
@NoArgsConstructor                         // JPA 需要
@AllArgsConstructor
public class Product { ... }

// ========== 3. ddl-auto ==========
// 开发: update
// 生产: validate 或 none
// 生产禁用 ddl-auto, 用 Flyway/Liquibase 管理

// ========== 4. 避免 N+1 查询 ==========
// 使用 @EntityGraph 或 fetch join

// ========== 5. 事务管理 ==========
// Service 层加 @Transactional

// ========== 6. 软删除 ==========
@Column(nullable = false)
private Boolean deleted = false;

// ========== 7. 审计字段 ==========
// 使用 @CreatedDate @LastModifiedDate

// ========== 8. 合理选择主键策略 ==========
// MySQL: IDENTITY
// PostgreSQL: SEQUENCE (性能更好)
// UUID: UUID (分布式)
```


> **Note:** 💡 JPA 要点: @Entity 映射表; @Id 主键; JpaRepository 提供 CRUD+分页; 派生查询按方法名自动生成 SQL; ddl-auto=update 自动建表; IDENTITY/SEQUENCE/UUID 主键策略; @Enumerated 枚举映射; 谨慎使用 @Data 于实体; 生产用 validate+Flyway。


## 练习


<!-- Converted from: 14_Spring Data JPA 入门.html -->
