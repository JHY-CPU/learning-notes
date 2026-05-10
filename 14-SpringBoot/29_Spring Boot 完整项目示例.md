# Spring Boot 完整项目示例


## 🏗️ Spring Boot 完整项目示例 — 博客系统


综合应用 Spring Boot/Spring Security/JPA/Redis/RabbitMQ 等全套技术栈构建博客后端 API。


## 项目概览


```
// ========== 博客系统完整项目 ==========
// 功能: 用户注册登录、文章 CRUD、标签分类、评论、搜索、通知

// ========== 技术栈 ==========
// Spring Boot 3.2        — 框架基础
// Spring Data JPA         — ORM
// Spring Security + JWT   — 认证授权
// Spring Data Redis       — 缓存 + 分布式锁
// RabbitMQ                — 异步通知
// Flyway                  — 数据库迁移
// MapStruct               — DTO 映射
// SpringDoc OpenAPI       — API 文档
// JUnit 5 + Mockito       — 测试
// MySQL                   — 主数据库
// Docker                  — 容器化

// ========== 模块结构 ==========
// com.example.blog/
// ├── BlogApplication.java
// ├── config/          — Security/Redis/RabbitMQ/Swagger
// ├── common/          — ApiResponse/ErrorCode/BusinessException/BaseEntity
// ├── auth/            — JwtUtils/AuthController/AuthService
// ├── user/            — User/UserService/UserController/UserMapper
// ├── article/         — Article/Tag/Category/ArticleService
// ├── comment/         — Comment/CommentService
// └── notification/    — Notification/NotifyService/RabbitMQ

// ========== pom.xml 核心依赖 ==========
// <parent>
//     <groupId>org.springframework.boot</groupId>
//     <artifactId>spring-boot-starter-parent</artifactId>
//     <version>3.2.0</version>
// </parent>
//
// <dependencies>
//     <!-- Web -->
//     <dependency>spring-boot-starter-web</dependency>
//     <dependency>spring-boot-starter-validation</dependency>
//
//     <!-- Database -->
//     <dependency>spring-boot-starter-data-jpa</dependency>
//     <dependency>mysql-connector-j</dependency>
//     <dependency>flyway-core</dependency>
//     <dependency>flyway-mysql</dependency>
//
//     <!-- Security -->
//     <dependency>spring-boot-starter-security</dependency>
//     <dependency>jjwt-api, jjwt-impl, jjwt-jackson</dependency>
//
//     <!-- Cache & MQ -->
//     <dependency>spring-boot-starter-data-redis</dependency>
//     <dependency>spring-boot-starter-amqp</dependency>
//
//     <!-- Tools -->
//     <dependency>mapstruct</dependency>
//     <dependency>lombok</dependency>
//     <dependency>springdoc-openapi-starter-webmvc-ui</dependency>
//     <dependency>spring-boot-starter-actuator</dependency>
//
//     <!-- Test -->
//     <dependency>spring-boot-starter-test</dependency>
//     <dependency>spring-security-test</dependency>
//     <dependency>testcontainers</dependency>
// </dependencies>
```


## 用户与认证模块


```
// ========== 用户实体 ==========
@Entity
@Table(name = "users")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class User extends BaseEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false, length = 50)
    private String username;

    @Column(unique = true, nullable = false)
    private String email;

    @Column(nullable = false)
    private String password;           // BCrypt 哈希

    private String nickname;
    private String avatar;

    @Enumerated(EnumType.STRING)
    private UserStatus status;         // ACTIVE, INACTIVE, BANNED

    @ManyToMany(fetch = FetchType.EAGER)
    @JoinTable(name = "user_roles",
        joinColumns = @JoinColumn(name = "user_id"),
        inverseJoinColumns = @JoinColumn(name = "role_id"))
    private Set<Role> roles = new HashSet<>();

    @Column(columnDefinition = "int default 0")
    private Integer articleCount;

    private LocalDateTime lastLoginAt;
}

// ========== DTO ==========
@Data
public class RegisterRequest {
    @NotBlank @Size(min = 3, max = 50)
    private String username;
    @NotBlank @Email
    private String email;
    @NotBlank @Size(min = 6, max = 100)
    private String password;
}

@Data
public class LoginRequest {
    @NotBlank
    private String username;
    @NotBlank
    private String password;
}

@Data
@AllArgsConstructor
public class AuthResponse {
    private String accessToken;
    private String refreshToken;
    private String tokenType = "Bearer";
    private long expiresIn;
}

// ========== 认证服务 ==========
@Service
@RequiredArgsConstructor
@Transactional
public class AuthService {

    private final UserRepository userRepository;
    private final RoleRepository roleRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtils jwtUtils;
    private final AuthenticationManager authenticationManager;

    public AuthResponse register(RegisterRequest request) {
        if (userRepository.existsByUsername(request.getUsername())) {
            throw new BusinessException(ErrorCode.USER_EXISTS);
        }
        if (userRepository.existsByEmail(request.getEmail())) {
            throw new BusinessException(ErrorCode.CONFLICT, "邮箱已被注册");
        }

        User user = User.builder()
            .username(request.getUsername())
            .email(request.getEmail())
            .password(passwordEncoder.encode(request.getPassword()))
            .nickname(request.getUsername())
            .status(UserStatus.ACTIVE)
            .roles(Set.of(roleRepository.findByName("ROLE_USER")
                .orElseThrow(() -> new RuntimeException("默认角色不存在"))))
            .build();
        userRepository.save(user);

        String accessToken = jwtUtils.generateAccessToken(user);
        String refreshToken = jwtUtils.generateRefreshToken(user);
        return new AuthResponse(accessToken, refreshToken, "Bearer", jwtUtils.getExpirationMs());
    }

    public AuthResponse login(LoginRequest request) {
        Authentication authentication = authenticationManager.authenticate(
            new UsernamePasswordAuthenticationToken(request.getUsername(), request.getPassword()));

        UserDetails userDetails = (UserDetails) authentication.getPrincipal();
        User user = userRepository.findByUsername(userDetails.getUsername())
            .orElseThrow(() -> new BusinessException(ErrorCode.USER_NOT_FOUND));
        user.setLastLoginAt(LocalDateTime.now());
        userRepository.save(user);

        String accessToken = jwtUtils.generateAccessToken(user);
        String refreshToken = jwtUtils.generateRefreshToken(user);
        return new AuthResponse(accessToken, refreshToken, "Bearer", jwtUtils.getExpirationMs());
    }
}

// ========== UserDetailsService ==========
@Service
@RequiredArgsConstructor
public class BlogUserDetailsService implements UserDetailsService {
    private final UserRepository userRepository;

    @Override
    public UserDetails loadUserByUsername(String username) {
        User user = userRepository.findByUsername(username)
            .orElseThrow(() -> new UsernameNotFoundException("用户不存在"));
        return new org.springframework.security.core.userdetails.User(
            user.getUsername(), user.getPassword(), user.getStatus() == UserStatus.ACTIVE,
            true, true, true,
            user.getRoles().stream()
                .map(r -> new SimpleGrantedAuthority(r.getName()))
                .collect(Collectors.toSet()));
    }
}
```


## 文章模块


```
// ========== 文章实体 ==========
@Entity
@Table(name = "articles")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Article extends BaseEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 200)
    private String title;

    @Lob
    @Column(nullable = false, columnDefinition = "LONGTEXT")
    private String content;

    @Column(length = 500)
    private String summary;

    @Enumerated(EnumType.STRING)
    private ArticleStatus status;       // DRAFT, PUBLISHED, ARCHIVED

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "author_id", nullable = false)
    @ToString.Exclude
    private User author;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "category_id")
    @ToString.Exclude
    private Category category;

    @ManyToMany
    @JoinTable(name = "article_tags",
        joinColumns = @JoinColumn(name = "article_id"),
        inverseJoinColumns = @JoinColumn(name = "tag_id"))
    private Set<Tag> tags = new HashSet<>();

    @Column(columnDefinition = "int default 0")
    private Integer viewCount;

    @Column(columnDefinition = "int default 0")
    private Integer commentCount;

    @Column(columnDefinition = "boolean default false")
    private boolean featured;

    private LocalDateTime publishedAt;
}

// ========== 文章 DTO ==========
@Data
public class CreateArticleRequest {
    @NotBlank @Size(max = 200)
    private String title;
    @NotBlank
    private String content;
    @Size(max = 500)
    private String summary;
    private Long categoryId;
    private Set<Long> tagIds;
    private ArticleStatus status;
}

@Data
public class ArticleVO {
    private Long id;
    private String title;
    private String summary;
    private String content;
    private ArticleStatus status;
    private String authorName;
    private String authorAvatar;
    private String categoryName;
    private Set<String> tags;
    private int viewCount;
    private int commentCount;
    private boolean featured;
    private LocalDateTime publishedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}

// ========== 文章 Mapper ==========
@Mapper(componentModel = "spring", uses = {TagMapper.class, CategoryMapper.class})
public interface ArticleMapper {

    @Mapping(target = "id", ignore = true)
    @Mapping(target = "author", ignore = true)
    @Mapping(target = "category", ignore = true)
    @Mapping(target = "tags", ignore = true)
    @Mapping(target = "viewCount", constant = "0")
    @Mapping(target = "commentCount", constant = "0")
    @Mapping(target = "featured", constant = "false")
    @Mapping(target = "publishedAt", expression = "java(java.time.LocalDateTime.now())")
    Article toEntity(CreateArticleRequest request);

    @Mapping(source = "author.username", target = "authorName")
    @Mapping(source = "author.avatar", target = "authorAvatar")
    @Mapping(source = "category.name", target = "categoryName")
    @Mapping(target = "tags", expression = "java(article.getTags().stream().map(Tag::getName).collect(java.util.stream.Collectors.toSet()))")
    ArticleVO toVO(Article article);

    List<ArticleVO> toVOList(List<Article> articles);
}

// ========== 文章服务 ==========
@Service
@RequiredArgsConstructor
@Transactional
public class ArticleService {

    private final ArticleRepository articleRepository;
    private final ArticleMapper articleMapper;
    private final UserRepository userRepository;
    private final CategoryRepository categoryRepository;
    private final TagRepository tagRepository;

    @CacheEvict(cacheNames = "articles:list", allEntries = true)
    public ArticleVO createArticle(CreateArticleRequest request, String username) {
        User author = userRepository.findByUsername(username)
            .orElseThrow(() -> new BusinessException(ErrorCode.USER_NOT_FOUND));

        Article article = articleMapper.toEntity(request);
        article.setAuthor(author);

        if (request.getCategoryId() != null) {
            article.setCategory(categoryRepository.findById(request.getCategoryId())
                .orElseThrow(() -> new BusinessException(ErrorCode.NOT_FOUND, "分类不存在")));
        }
        if (request.getTagIds() != null) {
            article.setTags(new HashSet<>(tagRepository.findAllById(request.getTagIds())));
        }

        Article saved = articleRepository.save(article);

        // 异步发送通知
        // notificationService.notifyNewArticle(saved);
        return articleMapper.toVO(saved);
    }

    @Cacheable(cacheNames = "articles:list", key = "#pageable")
    @Transactional(readOnly = true)
    public Page<ArticleVO> getPublishedArticles(Pageable pageable) {
        return articleRepository.findByStatus(ArticleStatus.PUBLISHED, pageable)
            .map(articleMapper::toVO);
    }

    @Cacheable(cacheNames = "articles:detail", key = "#id")
    @Transactional(readOnly = true)
    public ArticleVO getArticleById(Long id) {
        Article article = articleRepository.findById(id)
            .orElseThrow(() -> new BusinessException(ErrorCode.NOT_FOUND, "文章不存在"));
        // 浏览量 +1 (不需要强一致性)
        articleRepository.incrementViewCount(id);
        return articleMapper.toVO(article);
    }

    @CacheEvict(cacheNames = {"articles:detail", "articles:list"}, allEntries = true)
    public ArticleVO updateArticle(Long id, CreateArticleRequest request, String username) {
        Article article = articleRepository.findById(id)
            .orElseThrow(() -> new BusinessException(ErrorCode.NOT_FOUND, "文章不存在"));
        if (!article.getAuthor().getUsername().equals(username)) {
            throw new BusinessException(ErrorCode.FORBIDDEN, "只能编辑自己的文章");
        }
        article.setTitle(request.getTitle());
        article.setContent(request.getContent());
        article.setSummary(request.getSummary());
        article.setStatus(request.getStatus());
        return articleMapper.toVO(articleRepository.save(article));
    }
}
```


## 评论与通知


```
// ========== 评论实体 ==========
@Entity
@Table(name = "comments")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Comment extends BaseEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Lob
    @Column(nullable = false)
    private String content;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "article_id", nullable = false)
    @ToString.Exclude
    private Article article;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "author_id", nullable = false)
    @ToString.Exclude
    private User author;

    // 回复评论 (自关联)
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "parent_id")
    @ToString.Exclude
    private Comment parent;

    @Column(columnDefinition = "boolean default false")
    private boolean approved;
}

// ========== 评论服务 ==========
@Service
@RequiredArgsConstructor
@Transactional
public class CommentService {

    private final CommentRepository commentRepository;
    private final ArticleRepository articleRepository;
    private final RabbitTemplate rabbitTemplate;

    public CommentVO createComment(CreateCommentRequest request, String username) {
        Article article = articleRepository.findById(request.getArticleId())
            .orElseThrow(() -> new BusinessException(ErrorCode.NOT_FOUND, "文章不存在"));
        User author = userRepository.findByUsername(username)
            .orElseThrow(() -> new BusinessException(ErrorCode.USER_NOT_FOUND));

        Comment comment = Comment.builder()
            .content(request.getContent())
            .article(article)
            .author(author)
            .approved(true)
            .build();

        if (request.getParentId() != null) {
            comment.setParent(commentRepository.findById(request.getParentId())
                .orElseThrow(() -> new BusinessException(ErrorCode.NOT_FOUND, "父评论不存在")));
        }

        Comment saved = commentRepository.save(comment);

        // 更新文章评论数
        articleRepository.incrementCommentCount(article.getId());

        // 发送异步通知 (RabbitMQ)
        CommentNotification notification = new CommentNotification(
            article.getAuthor().getId(), author.getUsername(),
            article.getId(), article.getTitle(), saved.getId());
        rabbitTemplate.convertAndSend("exchange.topic", "notification.comment", notification);

        return toVO(saved);
    }

    @Transactional(readOnly = true)
    public Page<CommentVO> getArticleComments(Long articleId, Pageable pageable) {
        return commentRepository.findByArticleIdAndParentIsNull(articleId, pageable)
            .map(this::toVO);
    }

    private CommentVO toVO(Comment comment) {
        CommentVO vo = new CommentVO();
        vo.setId(comment.getId());
        vo.setContent(comment.getContent());
        vo.setAuthorName(comment.getAuthor().getUsername());
        vo.setAuthorAvatar(comment.getAuthor().getAvatar());
        vo.setCreatedAt(comment.getCreatedAt());
        if (comment.getParent() != null) {
            vo.setParentId(comment.getParent().getId());
        }
        // 加载回复
        vo.setReplies(commentRepository.findByParentId(comment.getId())
            .stream().map(this::toVO).collect(Collectors.toList()));
        return vo;
    }
}

// ========== RabbitMQ 通知消费者 ==========
@Component
@Slf4j
public class NotificationConsumer {

    @RabbitListener(queues = "queue.notification")
    public void handleCommentNotification(CommentNotification notification) {
        log.info("评论通知: 用户{} 评论了文章 {}", notification.getCommenter(),
            notification.getArticleTitle());
        // 发送站内信/邮件/推送
    }

    @RabbitListener(queues = "queue.notification")
    public void handleSystemNotification(SystemNotification notification) {
        log.info("系统通知: {}", notification.getMessage());
    }
}

// ========== 搜索服务 ==========
@Service
@RequiredArgsConstructor
public class SearchService {

    private final ArticleRepository articleRepository;

    @Cacheable(cacheNames = "search", key = "#keyword + ':' + #pageable")
    @Transactional(readOnly = true)
    public Page<ArticleVO> searchArticles(String keyword, Pageable pageable) {
        return articleRepository
            .findByTitleContainingOrContentContaining(keyword, keyword, pageable)
            .map(articleMapper::toVO);
    }
}
```


## 安全配置与 API 文档


```
// ========== Security 配置 ==========
@Configuration
@EnableWebSecurity
@EnableMethodSecurity
@RequiredArgsConstructor
public class BlogSecurityConfig {

    private final JwtAuthenticationFilter jwtAuthFilter;
    private final BlogUserDetailsService userDetailsService;
    private final JwtAuthenticationEntryPoint authEntryPoint;

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf(AbstractHttpConfigurer::disable)
            .cors(cors -> cors.configurationSource(corsConfigurationSource()))
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/auth/**").permitAll()
                .requestMatchers("/api/public/**").permitAll()
                .requestMatchers(HttpMethod.GET, "/api/articles/**").permitAll()
                .requestMatchers("/swagger-ui/**", "/v3/api-docs/**").permitAll()
                .requestMatchers("/actuator/health").permitAll()
                .anyRequest().authenticated()
            )
            .sessionManagement(session ->
                session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .exceptionHandling(ex -> ex.authenticationEntryPoint(authEntryPoint))
            .authenticationProvider(authenticationProvider())
            .addFilterBefore(jwtAuthFilter, UsernamePasswordAuthenticationFilter.class);
        return http.build();
    }

    @Bean
    public AuthenticationProvider authenticationProvider() {
        DaoAuthenticationProvider provider = new DaoAuthenticationProvider();
        provider.setUserDetailsService(userDetailsService);
        provider.setPasswordEncoder(passwordEncoder());
        return provider;
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration config = new CorsConfiguration();
        config.setAllowedOriginPatterns(List.of("*"));
        config.setAllowedMethods(List.of("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"));
        config.setAllowedHeaders(List.of("*"));
        config.setAllowCredentials(true);
        config.setMaxAge(3600L);
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/api/**", config);
        return source;
    }
}

// ========== SpringDoc 配置 ==========
@Configuration
public class OpenApiConfig {

    @Bean
    public OpenAPI blogOpenAPI() {
        return new OpenAPI()
            .info(new Info()
                .title("博客系统 API")
                .description("Spring Boot 博客后端接口文档")
                .version("1.0.0")
                .contact(new Contact().name("开发者").email("dev@example.com")))
            .addSecurityItem(new SecurityRequirement().addList("BearerAuth"))
            .components(new Components()
                .addSecuritySchemes("BearerAuth",
                    new SecurityScheme()
                        .type(SecurityScheme.Type.HTTP)
                        .scheme("bearer")
                        .bearerFormat("JWT")));
    }
}

// ========== 缓存配置 ==========
@Configuration
@EnableCaching
public class BlogCacheConfig {

    @Bean
    public CacheManager cacheManager(RedisConnectionFactory factory) {
        RedisCacheConfiguration defaultConfig = RedisCacheConfiguration.defaultCacheConfig()
            .entryTtl(Duration.ofMinutes(10))
            .disableCachingNullValues()
            .serializeValuesWith(
                RedisSerializationContext.SerializationPair.fromSerializer(
                    new GenericJackson2JsonRedisSerializer()));

        Map<String, RedisCacheConfiguration> configs = new HashMap<>();
        configs.put("articles:list", defaultConfig.entryTtl(Duration.ofMinutes(5)));
        configs.put("articles:detail", defaultConfig.entryTtl(Duration.ofMinutes(30)));
        configs.put("search", defaultConfig.entryTtl(Duration.ofMinutes(2)));

        return RedisCacheManager.builder(factory)
            .cacheDefaults(defaultConfig)
            .withInitialCacheConfigurations(configs)
            .build();
    }
}

// ========== application.yml ==========
// spring:
//   datasource:
//     url: jdbc:mysql://localhost:3306/blog?useSSL=false&allowPublicKeyRetrieval=true
//     username: root
//     password: root
//     hikari:
//       maximum-pool-size: 20
//       minimum-idle: 5
//   jpa:
//     hibernate:
//       ddl-auto: validate
//     show-sql: false
//     properties:
//       hibernate:
//         format_sql: true
//   flyway:
//     enabled: true
//     locations: classpath:db/migration
//   data:
//     redis:
//       host: localhost
//       port: 6379
//       lettuce:
//         pool:
//           max-active: 16
//   rabbitmq:
//     host: localhost
//     port: 5672
//
// jwt:
//   secret: base64-encoded-256-bit-secret-key-here...
//   access-token-expiration: 1800000       # 30 分钟
//   refresh-token-expiration: 604800000    # 7 天
```


## API 端点总览


```
// ========== 完整 API 路由 ==========

// ========== 认证模块 ==========
// POST   /api/auth/register      — 注册
// POST   /api/auth/login         — 登录
// POST   /api/auth/refresh       — 刷新 Token
// POST   /api/auth/logout        — 登出

// ========== 用户模块 ==========
// GET    /api/users/me           — 获取当前用户信息
// PUT    /api/users/me           — 更新个人信息
// GET    /api/users/{id}         — 查看用户公开信息
// GET    /api/users/{id}/articles — 用户的文章列表

// ========== 文章模块 ==========
// GET    /api/articles           — 文章列表 (分页/排序/筛选)
// GET    /api/articles/{id}      — 文章详情
// POST   /api/articles           — 创建文章 [认证]
// PUT    /api/articles/{id}      — 更新文章 [作者]
// DELETE /api/articles/{id}      — 删除文章 [作者]
// PUT    /api/articles/{id}/publish  — 发布 [作者]

// ========== 分类与标签 ==========
// GET    /api/categories         — 分类列表
// GET    /api/tags               — 标签列表
// GET    /api/articles/search?keyword=xxx — 搜索

// ========== 评论模块 ==========
// GET    /api/articles/{id}/comments    — 文章评论
// POST   /api/articles/{id}/comments    — 发表评论 [认证]
// DELETE /api/comments/{id}             — 删除评论 [作者/管理员]

// ========== Admin 管理 ==========
// GET    /api/admin/users         — 用户管理 [ADMIN]
// PUT    /api/admin/users/{id}/status — 修改用户状态 [ADMIN]
// DELETE /api/admin/articles/{id} — 强制删除文章 [ADMIN]

// ========== 健康与监控 ==========
// GET    /actuator/health         — 健康检查
// GET    /actuator/info           — 应用信息
// GET    /actuator/metrics        — 指标
// GET    /swagger-ui.html         — API 文档

// ========== 完整请求/响应示例 ==========
// 请求: POST /api/articles
// Authorization: Bearer eyJhbGci...
// Content-Type: application/json
//
// {
//   "title": "Spring Boot 最佳实践",
//   "content": "# 介绍\n在本文中...",
//   "summary": "Spring Boot 开发中的最佳实践总结",
//   "categoryId": 1,
//   "tagIds": [1, 2, 3],
//   "status": "PUBLISHED"
// }
//
// 响应: 201 Created
// {
//   "code": 0,
//   "message": "success",
//   "data": {
//     "id": 42,
//     "title": "Spring Boot 最佳实践",
//     "summary": "Spring Boot 开发中的最佳实践总结",
//     "status": "PUBLISHED",
//     "authorName": "admin",
//     "categoryName": "后端开发",
//     "tags": ["Java", "Spring Boot", "最佳实践"],
//     "viewCount": 0,
//     "publishedAt": "2024-01-15T10:30:00"
//   },
//   "requestId": "req-abc123",
//   "timestamp": 1705300000000
// }
```


> **Note:** 💡 博客项目综合要点: 按业务分包 (user/article/comment/notification); JWT 认证 + 方法安全; Redis 缓存文章; RabbitMQ 异步通知; Flyway 数据库迁移; MapStruct DTO 映射; 统一异常处理; 分页/搜索/排序; Docker 容器化; 完整 API 文档。


## 练习


<!-- Converted from: 29_Spring Boot 完整项目示例.html -->
