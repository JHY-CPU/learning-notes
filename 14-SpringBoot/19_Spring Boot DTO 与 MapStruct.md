# Spring Boot DTO 与 MapStruct


## 🔄 DTO 模式与 MapStruct


DTO/VO 分层设计、Entity ↔ DTO 转换、MapStruct 编译时映射、BeanUtils 对比、分层架构最佳实践。


## DTO 分层设计


```
// ========== DTO (Data Transfer Object) ==========
// 在不同层之间传输数据
// 避免直接暴露 Entity 给外部

// ========== 为什么需要 DTO ==========
// 1. 安全: 不暴露敏感字段 (密码)
// 2. 解耦: Entity 变更不影响 API
// 3. 灵活: 自定义展示字段
// 4. 性能: 只传输需要的数据
// 5. 避免循环引用: Entity 关联关系复杂

// ========== 三层结构 ==========
// Controller 层   ↔   Service 层   ↔   Repository 层
//   DTO/VO              DTO/Entity         Entity

// ========== 请求 DTO (CreateUserRequest) ==========
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CreateUserRequest {
    @NotBlank
    private String username;

    @NotBlank
    @Size(min = 6)
    private String password;

    @Email
    private String email;

    @Min(0) @Max(150)
    private Integer age;
}

// ========== 响应 DTO (UserVO) ==========
@Data
@Builder
public class UserVO {
    private Long id;
    private String username;
    private String email;
    private Integer age;
    private String status;
    private LocalDateTime createdAt;
    // 注意: 不含 password!
}

// ========== 更新 DTO ==========
@Data
public class UpdateUserRequest {
    @Size(min = 3, max = 50)
    private String username;

    @Email
    private String email;

    @Min(0) @Max(150)
    private Integer age;
}

// ========== Entity (内部) ==========
@Entity
@Table(name = "users")
@Data
@NoArgsConstructor
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String username;
    private String password;        // 敏感: 不暴露
    private String email;
    private Integer age;
    @Enumerated(EnumType.STRING)
    private UserStatus status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
```


## 手动转换 vs BeanUtils


```
// ========== 方式 1: 手动转换 ==========
// 最直接, 但代码冗长

@Service
public class UserService {

    public UserVO toVO(User user) {
        return UserVO.builder()
            .id(user.getId())
            .username(user.getUsername())
            .email(user.getEmail())
            .age(user.getAge())
            .status(user.getStatus().name())
            .createdAt(user.getCreatedAt())
            .build();
    }

    public User toEntity(CreateUserRequest request) {
        User user = new User();
        user.setUsername(request.getUsername());
        user.setPassword(passwordEncoder.encode(request.getPassword()));
        user.setEmail(request.getEmail());
        user.setAge(request.getAge());
        user.setStatus(UserStatus.ACTIVE);
        return user;
    }

    public List<UserVO> toVOList(List<User> users) {
        return users.stream()
            .map(this::toVO)
            .collect(Collectors.toList());
    }
}

// ========== 方式 2: BeanUtils (反射, 不推荐) ==========
// 性能差, 无类型安全, 易出错

public UserVO toVO(User user) {
    UserVO vo = new UserVO();
    BeanUtils.copyProperties(user, vo);   // 同名属性复制
    // 问题: 字段名不同时不工作
    // 问题: 无编译期检查
    return vo;
}

// ========== BeanUtils 的问题 ==========
// 1. 反射调用 → 性能差 (比 MapStruct 慢 10-100 倍)
// 2. 无类型安全 → 运行时出错
// 3. 字段名需完全一致
// 4. null 复制问题
// 5. 内存溢出风险 (大对象)
```


## MapStruct


```
// ========== MapStruct ==========
// 编译时生成映射代码 (非反射)
// 性能最佳, 类型安全

// ========== 1. 添加依赖 ==========
// Maven:
<dependency>
    <groupId>org.mapstruct</groupId>
    <artifactId>mapstruct</artifactId>
    <version>1.5.5.Final</version>
</dependency>

// 注解处理器:
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-compiler-plugin</artifactId>
    <configuration>
        <annotationProcessorPaths>
            <path>
                <groupId>org.mapstruct</groupId>
                <artifactId>mapstruct-processor</artifactId>
                <version>1.5.5.Final</version>
            </path>
            <path>
                <groupId>org.projectlombok</groupId>
                <artifactId>lombok</artifactId>
            </path>
        </annotationProcessorPaths>
    </configuration>
</plugin>

// ========== 2. 定义 Mapper ==========
@Mapper(componentModel = "spring")         // 生成 Spring Bean
public interface UserMapper {

    // Entity → DTO
    UserVO toVO(User user);

    // DTO → Entity
    User toEntity(CreateUserRequest request);

    // List 映射
    List<UserVO> toVOList(List<User> users);

    // 多参数映射
    @Mapping(target = "id", ignore = true)
    @Mapping(target = "password", ignore = true)
    @Mapping(target = "status", constant = "ACTIVE")
    @Mapping(target = "createdAt", expression = "java(java.time.LocalDateTime.now())")
    User toEntity(CreateUserRequest request, @Context PasswordEncoder encoder);

    // 更新 Entity (部分字段)
    @Mapping(target = "id", ignore = true)
    @Mapping(target = "password", ignore = true)
    @Mapping(target = "status", ignore = true)
    @Mapping(target = "createdAt", ignore = true)
    void updateEntity(UpdateUserRequest request, @MappingTarget User user);
}

// ========== 3. 使用 ==========
@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;
    private final UserMapper userMapper;
    private final PasswordEncoder passwordEncoder;

    public UserVO createUser(CreateUserRequest request) {
        User user = userMapper.toEntity(request);
        user.setPassword(passwordEncoder.encode(request.getPassword()));
        User saved = userRepository.save(user);
        return userMapper.toVO(saved);
    }

    public UserVO getUser(Long id) {
        User user = userRepository.findById(id).orElseThrow();
        return userMapper.toVO(user);
    }

    public List<UserVO> getAllUsers() {
        return userMapper.toVOList(userRepository.findAll());
    }

    public UserVO updateUser(Long id, UpdateUserRequest request) {
        User user = userRepository.findById(id).orElseThrow();
        userMapper.updateEntity(request, user);           // 部分更新
        User saved = userRepository.save(user);
        return userMapper.toVO(saved);
    }
}

// ========== 生成的代码 (编译后) ==========
// MapStruct 编译时生成 UserMapperImpl
// 使用 getter/setter 直接赋值, 无反射
// 性能 ≈ 手动转换
```


## MapStruct 高级


```
// ========== MapStruct 高级映射 ==========

// ========== 自定义类型转换 ==========
public class DateMapper {
    public String asString(LocalDateTime date) {
        return date != null ? date.format(DateTimeFormatter.ISO_DATE) : null;
    }

    public LocalDateTime asLocalDateTime(String date) {
        return date != null ? LocalDateTime.parse(date) : null;
    }
}

@Mapper(componentModel = "spring", uses = DateMapper.class)
public interface OrderMapper {
    OrderVO toVO(Order order);
}

// ========== 默认值 ==========
@Mapper
public interface ProductMapper {
    @Mapping(target = "status", defaultValue = "ACTIVE")
    @Mapping(target = "version", defaultExpression = "java(1L)")
    @Mapping(target = "tags", defaultExpression = "java(new ArrayList<>())")
    Product toEntity(CreateProductRequest request);
}

// ========== 忽略字段 ==========
@Mapper
public interface UserMapper {
    @Mapping(target = "password", ignore = true)   // 忽略字段
    @Mapping(target = "roles", ignore = true)
    UserVO toVO(User user);
}

// ========== 嵌套对象映射 ==========
// User 有 Address address
// UserVO 有 AddressVO addressVO

@Mapper(componentModel = "spring")
public interface AddressMapper {
    AddressVO toVO(Address address);
}

@Mapper(componentModel = "spring", uses = AddressMapper.class)
public interface UserMapper {
    UserVO toVO(User user);   // 自动使用 AddressMapper 处理嵌套
}

// ========== 常量映射 ==========
@Mapping(target = "type", constant = "USER")
@Mapping(target = "source", constant = "API")

// ========== 多源映射 ==========
@Mapping(target = "fullName", source = "user.name")
@Mapping(target = "orderCount", source = "stats.count")
UserDetailVO toDetailVO(User user, OrderStats stats);

// ========== 反向映射 ==========
@Mapper
public interface UserMapper {
    UserVO toVO(User user);
    User toEntity(UserVO userVO);         // 反向
    // 或: @InheritInverseConfiguration
    @InheritInverseConfiguration
    User toEntity(UserVO userVO);
}
```


## 最佳实践


```
// ========== DTO 最佳实践 ==========

// ========== 1. 严格分层 ==========
// Controller: 接收/返回 DTO
// Service:    业务逻辑, DTO ↔ Entity
// Repository: 仅操作 Entity

// ❌ 错误: 在 Controller 暴露 Entity
@GetMapping("/{id}")
public User getUser(@PathVariable Long id) {  // 直接返回 Entity
    return userService.findById(id);
}

// ✅ 正确: 返回 DTO
@GetMapping("/{id}")
public UserVO getUser(@PathVariable Long id) {
    return userService.getUser(id);
}

// ========== 2. 不要循环引用 ==========
// DTO 中避免双向引用
// UserVO 不含 orders, OrderVO 含 userId

// ========== 3. MapStruct 优先 ==========
// 编译时生成, 无反射, 类型安全, 高性能
// 比 BeanUtils 快 10-100 倍

// ========== 4. 验证 ==========
// 请求 DTO 用 @Valid 校验
// 响应 DTO 不需要校验

// ========== 5. 不可变 DTO ==========
// 响应 DTO 用 @Value 或 @Builder
// 创建后不可修改

// ========== 6. 集合映射 ==========
// 用 MapStruct 的 List 映射方法
// 避免手动 stream().map()

// ========== 7. 使用 @MappingTarget ==========
// 更新已有 Entity 时避免 new 对象
// 保持 JPA 实体在持久化上下文中

// ========== 8. 关注点分离 ==========
// 一个 Entity 对应多个 DTO
// CreateUserRequest / UpdateUserRequest / UserVO
```


> **Note:** 💡 DTO 与 MapStruct 要点: DTO 分层避免暴露 Entity; MapStruct 编译时映射 (非反射); @Mapper(componentModel="spring") 注入; @Mapping 控制字段映射; ignore/defaultValue/constant/expression 多种策略; 避免 BeanUtils 反射; 请求 DTO 校验; 响应 DTO 不变; MapStruct 性能 ≈ 手动。


## 练习


<!-- Converted from: 19_Spring Boot DTO 与 MapStruct.html -->
