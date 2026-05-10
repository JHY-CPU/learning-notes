# 项目实战 Spring Boot 博客平台


## 📦 项目实战 4: Spring Boot 博客平台


Spring Initializr 项目生成、JPA 实体关联、REST 控制器分页、Spring Security + JWT、统一异常处理。


## 项目结构与依赖


```
// blog-api/
// ├── src/main/java/com/blog/
// │   ├── BlogApplication.java
// │   ├── config/
// │   │   ├── SecurityConfig.java
// │   │   └── SwaggerConfig.java
// │   ├── controller/
// │   │   ├── AuthController.java
// │   │   ├── PostController.java
// │   │   └── CommentController.java
// │   ├── model/
// │   │   ├── User.java
// │   │   ├── Post.java
// │   │   └── Comment.java
// │   ├── repository/
// │   │   ├── UserRepository.java
// │   │   ├── PostRepository.java
// │   │   └── CommentRepository.java
// │   ├── service/
// │   │   ├── AuthService.java
// │   │   ├── PostService.java
// │   │   └── CommentService.java
// │   ├── dto/
// │   │   ├── LoginRequest.java
// │   │   ├── PostRequest.java
// │   │   └── PostResponse.java
// │   └── exception/
// │       └── GlobalExceptionHandler.java
// ├── src/main/resources/
// │   ├── application.yml
// │   └── db/migration/
// └── pom.xml

// ========== Maven 依赖 (pom.xml) ==========
//
//
//         org.springframework.boot
//         spring-boot-starter-web
//
//
//         org.springframework.boot
//         spring-boot-starter-data-jpa
//
//
//         org.springframework.boot
//         spring-boot-starter-security
//
//
//         org.springframework.boot
//         spring-boot-starter-validation
//
//
//         org.postgresql
//         postgresql
//
//
//         io.jsonwebtoken
//         jjwt-api
//         0.12.3
//
//
//         org.projectlombok
//         lombok
//
//

// ========== 配置 ==========
// application.yml:
// spring:
//   datasource:
//     url: jdbc:postgresql://localhost:5432/blog
//     username: postgres
//     password: secret
//   jpa:
//     hibernate:
//       ddl-auto: validate
//     show-sql: false
//   flyway:
//     enabled: true
//
// jwt:
//   secret: my-secret-key-must-be-at-least-256-bits-long
//   expiration: 900000  # 15分钟
```


## JPA 实体


```
// ========== 用户实体 ==========
// model/User.java
@Entity
@Table(name = "users")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true, length = 50)
    private String username;

    @Column(nullable = false, unique = true)
    private String email;

    @Column(nullable = false)
    private String password;

    @Column(length = 200)
    private String bio;

    @CreatedDate
    private LocalDateTime createdAt;
}

// ========== 文章实体 ==========
// model/Post.java
@Entity
@Table(name = "posts")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Post {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 200)
    private String title;

    @Column(nullable = false, columnDefinition = "TEXT")
    private String content;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "author_id", nullable = false)
    private User author;

    @OneToMany(mappedBy = "post", cascade = CascadeType.ALL, orphanRemoval = true)
    private List comments = new ArrayList<>();

    @CreatedDate
    private LocalDateTime createdAt;

    @LastModifiedDate
    private LocalDateTime updatedAt;
}

// ========== 仓库 (Repository) ==========
// repository/PostRepository.java
public interface PostRepository extends JpaRepository {
    Page findByAuthorId(Long authorId, Pageable pageable);

    @Query("SELECT p FROM Post p LEFT JOIN FETCH p.author ORDER BY p.createdAt DESC")
    Page findAllWithAuthor(Pageable pageable);

    // 搜索标题和内容
    @Query("SELECT p FROM Post p WHERE " +
           "LOWER(p.title) LIKE LOWER(CONCAT('%', :keyword, '%')) OR " +
           "LOWER(p.content) LIKE LOWER(CONCAT('%', :keyword, '%'))")
    Page search(@Param("keyword") String keyword, Pageable pageable);
}
```


## 控制层与安全


```
// ========== 文章控制器 ==========
// controller/PostController.java
@RestController
@RequestMapping("/api/v1/posts")
@RequiredArgsConstructor
public class PostController {

    private final PostService postService;

    @GetMapping
    public ResponseEntity> listPosts(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        return ResponseEntity.ok(postService.findAll(page, size));
    }

    @GetMapping("/{id}")
    public ResponseEntity getPost(@PathVariable Long id) {
        return ResponseEntity.ok(postService.findById(id));
    }

    @PostMapping
    public ResponseEntity createPost(
            @Valid @RequestBody PostRequest request,
            @AuthenticationPrincipal UserDetails user) {
        PostResponse post = postService.create(request, user.getId());
        return ResponseEntity.status(201).body(post);
    }

    @PutMapping("/{id}")
    public ResponseEntity updatePost(
            @PathVariable Long id,
            @Valid @RequestBody PostRequest request,
            @AuthenticationPrincipal UserDetails user) {
        return ResponseEntity.ok(postService.update(id, request, user.getId()));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity deletePost(
            @PathVariable Long id,
            @AuthenticationPrincipal UserDetails user) {
        postService.delete(id, user.getId());
        return ResponseEntity.noContent().build();
    }
}

// ========== Spring Security + JWT ==========
// config/SecurityConfig.java
@Configuration
@EnableWebSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    private final JwtAuthFilter jwtAuthFilter;

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf(AbstractHttpConfigurer::disable)
            .sessionManagement(sm -> sm.sessionCreationPolicy(STATELESS))
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/v1/auth/**", "/health").permitAll()
                .requestMatchers(HttpMethod.GET, "/api/v1/posts/**").permitAll()
                .anyRequest().authenticated()
            )
            .addFilterBefore(jwtAuthFilter, UsernamePasswordAuthenticationFilter.class);
        return http.build();
    }
}

// ========== 全局异常处理 ==========
// exception/GlobalExceptionHandler.java
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity handleNotFound(ResourceNotFoundException ex) {
        return ResponseEntity.status(404).body(
            new ErrorResponse("NOT_FOUND", ex.getMessage()));
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity handleValidation(
            MethodArgumentNotValidException ex) {
        List errors = ex.getBindingResult().getFieldErrors().stream()
            .map(e -> new FieldError(e.getField(), e.getDefaultMessage()))
            .toList();
        return ResponseEntity.status(422).body(
            new ErrorResponse("VALIDATION_ERROR", "参数验证失败", errors));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity handleGeneral(Exception ex) {
        return ResponseEntity.status(500).body(
            new ErrorResponse("INTERNAL_ERROR", "服务器内部错误"));
    }
}
```


> **Note:** 💡 Spring Boot 博客要点: JPA 实体关联 @ManyToOne/@OneToMany; Repository 分页 Page; Spring Security JWT 无状态; @ControllerAdvice 统一异常; DTO 隔离实体; @Valid 参数验证; Flyway 数据库迁移; Lombok @Data/@Builder 简化代码。


## 练习


<!-- Converted from: 3_项目实战 Spring Boot 博客平台.html -->
