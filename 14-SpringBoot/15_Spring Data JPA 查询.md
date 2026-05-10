# Spring Data JPA 查询


## 🔍 Spring Data JPA 查询


@Query JPQL/native 查询、@Modifying 更新、分页查询、动态查询 Specification、EntityGraph 解决 N+1。


## @Query 自定义查询


```
// ========== @Query ==========
// 自定义 JPQL 或原生 SQL 查询

public interface UserRepository extends JpaRepository<User, Long> {

    // ========== JPQL (对象查询) ==========
    @Query("SELECT u FROM User u WHERE u.email = ?1")
    Optional<User> findByEmailCustom(String email);

    // ========== 命名参数 ==========
    @Query("SELECT u FROM User u WHERE u.username = :username AND u.age > :minAge")
    List<User> findByUsernameAndAgeMin(
        @Param("username") String username,
        @Param("minAge") int minAge
    );

    // ========== 多表查询 ==========
    @Query("SELECT u FROM User u JOIN u.roles r WHERE r.name = :roleName")
    List<User> findByRoleName(@Param("roleName") String roleName);

    // ========== 部分字段查询 ==========
    @Query("SELECT u.username, u.email FROM User u WHERE u.id = :id")
    List<Object[]> findUsernameAndEmailById(@Param("id") Long id);

    // DTO 投影
    @Query("SELECT new com.example.dto.UserSummary(u.id, u.username, u.email) FROM User u")
    List<UserSummary> findAllSummaries();

    // ========== 原生 SQL ==========
    @Query(value = "SELECT * FROM users WHERE age > :minAge", nativeQuery = true)
    List<User> findUsersByAgeNative(@Param("minAge") int minAge);

    // 原生 SQL + 指定结果类
    @Query(value = "SELECT id, username FROM users WHERE status = :status",
           nativeQuery = true,
           resultSetMapping = "UserSummaryMapping")
    List<UserSummary> findSummariesByStatus(@Param("status") String status);

    // ========== 分页查询 ==========
    @Query("SELECT u FROM User u WHERE u.status = :status")
    Page<User> findByStatusWithPage(@Param("status") UserStatus status, Pageable pageable);

    @Query(value = "SELECT * FROM users WHERE status = :status",
           countQuery = "SELECT COUNT(*) FROM users WHERE status = :status",
           nativeQuery = true)
    Page<User> findByStatusNative(@Param("status") String status, Pageable pageable);

    // ========== 排序 ==========
    @Query("SELECT u FROM User u WHERE u.age > :age")
    List<User> findByAgeGreaterThan(@Param("age") int age, Sort sort);
}
```


## @Modifying 更新


```
// ========== @Modifying ==========
// 执行 UPDATE/DELETE 操作
// 需要配合 @Transactional

public interface UserRepository extends JpaRepository<User, Long> {

    // ========== 批量更新 ==========
    @Modifying
    @Transactional
    @Query("UPDATE User u SET u.status = :status WHERE u.lastLoginAt < :date")
    int deactivateInactiveUsers(
        @Param("status") UserStatus status,
        @Param("date") LocalDateTime date
    );

    // ========== 更新单个字段 ==========
    @Modifying
    @Transactional
    @Query("UPDATE User u SET u.email = :email WHERE u.id = :id")
    int updateEmail(@Param("id") Long id, @Param("email") String email);

    // ========== 批量删除 ==========
    @Modifying
    @Transactional
    @Query("DELETE FROM User u WHERE u.status = :status")
    int deleteByStatus(@Param("status") UserStatus status);

    // ========== @Modifying 清空缓存 ==========
    @Modifying(clearAutomatically = true)          // 执行后清空一级缓存
    @Query("UPDATE User u SET u.password = :password WHERE u.id = :id")
    int updatePassword(@Param("id") Long id, @Param("password") String password);
}

// ========== 派生更新 ==========
// Spring Data JPA 3.0+ 支持派生 DELETE
public interface UserRepository extends JpaRepository<User, Long> {
    @Transactional
    void deleteByStatus(UserStatus status);

    @Transactional
    int deleteByUsername(String username);

    @Transactional
    int deleteByAgeLessThan(int age);
}

// ========== 使用示例 ==========
@Service
@RequiredArgsConstructor
public class UserAdminService {

    private final UserRepository userRepository;

    @Transactional
    public int deactivateOldUsers() {
        LocalDateTime cutoff = LocalDateTime.now().minusMonths(6);
        int count = userRepository.deactivateInactiveUsers(
            UserStatus.INACTIVE, cutoff);
        log.info("已停用 {} 个用户", count);
        return count;
    }
}
```


## Specification 动态查询


```
// ========== Specification ==========
// 动态组合查询条件 (类似 QueryDSL)
// 适用于: 多条件筛选, 动态 WHERE

// ========== 1. Repository 继承 ==========
public interface UserRepository extends JpaRepository<User, Long>,
                                        JpaSpecificationExecutor<User> {
    // 获得方法: findAll(Specification), count(Specification), findOne(Specification)
}

// ========== 2. 定义查询条件 ==========
public class UserSpecifications {

    // 用户名包含
    public static Specification<User> usernameContains(String keyword) {
        return (root, query, cb) -> {
            if (keyword == null || keyword.isBlank()) return null;
            return cb.like(root.get("username"), "%" + keyword + "%");
        };
    }

    // 年龄范围
    public static Specification<User> ageBetween(Integer min, Integer max) {
        return (root, query, cb) -> {
            if (min == null && max == null) return null;
            if (min == null) return cb.lessThanOrEqualTo(root.get("age"), max);
            if (max == null) return cb.greaterThanOrEqualTo(root.get("age"), min);
            return cb.between(root.get("age"), min, max);
        };
    }

    // 状态匹配
    public static Specification<User> statusEquals(UserStatus status) {
        return (root, query, cb) -> {
            if (status == null) return null;
            return cb.equal(root.get("status"), status);
        };
    }

    // 创建时间范围
    public static Specification<User> createdAtBetween(LocalDateTime start, LocalDateTime end) {
        return (root, query, cb) -> {
            if (start == null && end == null) return null;
            if (start == null) return cb.lessThanOrEqualTo(root.get("createdAt"), end);
            if (end == null) return cb.greaterThanOrEqualTo(root.get("createdAt"), start);
            return cb.between(root.get("createdAt"), start, end);
        };
    }
}

// ========== 3. 使用 ==========
@Service
@RequiredArgsConstructor
public class UserSearchService {

    private final UserRepository userRepository;

    public Page<User> searchUsers(UserSearchCriteria criteria, Pageable pageable) {
        Specification<User> spec = Specification
            .where(UserSpecifications.usernameContains(criteria.getKeyword()))
            .and(UserSpecifications.ageBetween(criteria.getMinAge(), criteria.getMaxAge()))
            .and(UserSpecifications.statusEquals(criteria.getStatus()))
            .and(UserSpecifications.createdAtBetween(criteria.getStartDate(), criteria.getEndDate()));

        return userRepository.findAll(spec, pageable);
    }
}

// ========== 查询条件对象 ==========
@Data
public class UserSearchCriteria {
    private String keyword;
    private Integer minAge;
    private Integer maxAge;
    private UserStatus status;
    private LocalDateTime startDate;
    private LocalDateTime endDate;
}

// ========== Controller ==========
@GetMapping("/users/search")
public PageResponse<User> searchUsers(
        @ModelAttribute UserSearchCriteria criteria,
        @RequestParam(defaultValue = "0") int page,
        @RequestParam(defaultValue = "20") int size) {
    Page<User> result = userSearchService.searchUsers(
        criteria, PageRequest.of(page, size, Sort.by("createdAt").descending()));
    return PageResponse.from(result);
}
```


## EntityGraph 与 N+1


```
// ========== N+1 问题 ==========
// 查询 N 条记录时, 额外发 N 条 SQL 查关联
// findAll() → 1 条 SQL
// 遍历时 user.getOrders() → N 条额外 SQL

// ========== 解决方案 1: @EntityGraph ==========
public interface UserRepository extends JpaRepository<User, Long> {

    // fetch 关联的 orders, 用 LEFT JOIN 一次性查出
    @EntityGraph(attributePaths = {"orders"})
    @Query("SELECT u FROM User u WHERE u.id = :id")
    Optional<User> findByIdWithOrders(@Param("id") Long id);

    // 多个关联
    @EntityGraph(attributePaths = {"orders", "roles", "profile"})
    List<User> findAllWithDetails();
}

// ========== 解决方案 2: @NamedEntityGraph ==========
@Entity
@NamedEntityGraph(
    name = "User.withRolesAndOrders",
    attributeNodes = {
        @NamedAttributeNode("roles"),
        @NamedAttributeNode(value = "orders", subgraph = "orderItems")
    }
)
public class User { ... }

// Repository 中使用:
@EntityGraph("User.withRolesAndOrders")
List<User> findAll();

// ========== 解决方案 3: JOIN FETCH ==========
@Query("SELECT u FROM User u LEFT JOIN FETCH u.orders WHERE u.id = :id")
Optional<User> findByIdWithOrders(@Param("id") Long id);

// ========== 解决方案 4: @BatchSize ==========
@Entity
public class User {
    @BatchSize(size = 20)                   // 批量加载 20 个关联
    @OneToMany(mappedBy = "user")
    private List<Order> orders = new ArrayList<>();
}

// ========== DTO 投影 ==========
// 另一种避免 N+1 的方式: 只查需要的字段
public interface UserSummary {
    Long getId();
    String getUsername();
    String getEmail();
}

// Repository:
List<UserSummary> findByStatus(UserStatus status);

// 或使用 @Query + 构造器:
@Query("SELECT new com.example.dto.UserDTO(u.id, u.username, u.email) FROM User u")
List<UserDTO> findAllDTO();
```


## 最佳实践


```
// ========== 查询最佳实践 ==========

// ========== 1. 优先派生查询 ==========
// 简单查询用方法名, 复杂用 @Query
// 派生查询简单直观, 但过长的方法名不如 @Query

// ========== 2. 使用命名参数 ==========
// @Param("name") 比 ?1 更可读
// 重构时不会因为参数顺序出错

// ========== 3. 分页永远传 Pageable ==========
// 避免查询大量数据
// 前端传入 page/size/sort

// ========== 4. 警惕 N+1 ==========
// 用 @EntityGraph / JOIN FETCH / @BatchSize
// 在循环中访问关联属性会触发额外查询

// ========== 5. @Modifying(clearAutomatically=true) ==========
// 更新操作后清空一级缓存
// 否则后续查询可能读到旧数据

// ========== 6. 原生 SQL 谨慎使用 ==========
// nativeQuery = true 时与数据库耦合
// 仅在 JPQL 无法表达时使用

// ========== 7. 只查询需要的字段 ==========
// 使用 DTO 投影而非整个实体
// 减少数据传输量

// ========== 8. 日志监控 SQL ==========
// show-sql=true + format_sql=true
// 生产用监控工具: p6spy / datasource-proxy
```


> **Note:** 💡 查询要点: @Query JPQL 查询实体, nativeQuery 原生 SQL; @Param 命名参数; @Modifying UPDATE/DELETE; Specification 动态条件组合; @EntityGraph / JOIN FETCH 解决 N+1; DTO 投影减少数据量; Pageable 分页; 命名参数优于位置参数; clearAutomatically 清空缓存。


## 练习


<!-- Converted from: 15_Spring Data JPA 查询.html -->
