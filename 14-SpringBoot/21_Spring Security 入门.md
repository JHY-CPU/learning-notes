# Spring Security 入门


## 🔐 Spring Security 入门


SecurityFilterChain 核心过滤链、认证与授权机制、密码编码器、UserDetailsService、内存/数据库登录、方法安全、CSRF 保护。


## 核心概念


```
// ========== Spring Security 核心 ==========
// 认证 (Authentication): 你是谁?
// 授权 (Authorization): 你能做什么?
// 过滤器链 (Filter Chain): 请求经过多个过滤器

// ========== 依赖 ==========
// <dependency>
//     <groupId>org.springframework.boot</groupId>
//     <artifactId>spring-boot-starter-security</artifactId>
// </dependency>

// ========== 默认行为 ==========
// 引入依赖后:
// 1. 所有端点需要认证
// 2. 生成默认密码 (启动日志: Using generated security password)
// 3. 表单登录页面: /login
// 4. 用户: user, 密码: 控制台随机 UUID

// ========== SecurityFilterChain ==========
// 核心: 配置过滤链, 定义安全规则
// 替代旧的 WebSecurityConfigurerAdapter (已弃用)

@Configuration
@EnableWebSecurity                        // 启用 Security
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/public/**").permitAll()     // 公开
                .requestMatchers("/admin/**").hasRole("ADMIN") // 需要 ADMIN
                .requestMatchers("/user/**").hasRole("USER")   // 需要 USER
                .anyRequest().authenticated()                   // 其他需认证
            )
            .formLogin(form -> form
                .loginPage("/login")                            // 自定义登录页
                .defaultSuccessUrl("/home")                     // 登录成功跳转
                .permitAll()
            )
            .logout(logout -> logout
                .logoutUrl("/logout")
                .logoutSuccessUrl("/login?logout")
                .invalidateHttpSession(true)
                .deleteCookies("JSESSIONID")
                .permitAll()
            )
            .csrf(csrf -> csrf.disable())                      // 开发时可禁用
            .sessionManagement(session -> session
                .sessionCreationPolicy(SessionCreationPolicy.IF_REQUIRED)
            );
        return http.build();
    }

    // ========== 密码编码器 ==========
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();  // 推荐: BCrypt
        // 其他: Pbkdf2PasswordEncoder, SCryptPasswordEncoder
    }

    // ========== 内存用户 (测试用) ==========
    @Bean
    public UserDetailsService userDetailsService() {
        UserDetails user = User.builder()
            .username("user")
            .password(passwordEncoder().encode("password"))
            .roles("USER")
            .build();

        UserDetails admin = User.builder()
            .username("admin")
            .password(passwordEncoder().encode("admin"))
            .roles("ADMIN", "USER")
            .build();

        return new InMemoryUserDetailsManager(user, admin);
    }
}

// ========== 请求匹配 ==========
// requestMatchers 用法:
.requestMatchers("/api/**").authenticated()
.requestMatchers("/api/admin/**").hasRole("ADMIN")
.requestMatchers(HttpMethod.GET, "/api/users/**").hasRole("USER")
.requestMatchers(HttpMethod.POST, "/api/users").hasRole("ADMIN")
.requestMatchers("/public/**", "/error", "/favicon.ico").permitAll()
.anyRequest().authenticated()

// ========== 授权表达式 ==========
// .access() 支持 SpEL 表达式:
.requestMatchers("/admin/**").access("hasRole('ADMIN') and isFullyAuthenticated()")
.requestMatchers("/special/**").access("@customAuth.check(request, authentication)")

// 常用方法:
// hasRole('ADMIN')          — 有 ADMIN 角色
// hasAuthority('WRITE')     — 有 WRITE 权限
// hasAnyRole('ADMIN','MOD') — 有任一角色
// isAuthenticated()         — 已认证
// isFullyAuthenticated()    — 非记住我认证
// permitAll()               — 全部允许
// denyAll()                 — 全部拒绝
```


## 数据库认证


```
// ========== 数据库用户认证 ==========
// 从数据库读取用户信息进行认证

// ========== 1. 用户实体 ==========
@Entity
@Table(name = "users")
@Data
@NoArgsConstructor
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true)
    private String username;

    private String password;            // BCrypt 哈希

    private String email;

    @Column(nullable = false)
    private boolean enabled = true;

    // 角色: "ROLE_USER", "ROLE_ADMIN"
    @ManyToMany(fetch = FetchType.EAGER)
    @JoinTable(name = "user_roles",
        joinColumns = @JoinColumn(name = "user_id"),
        inverseJoinColumns = @JoinColumn(name = "role_id"))
    private Set<Role> roles = new HashSet<>();
}

@Entity
@Table(name = "roles")
@Data
@NoArgsConstructor
public class Role {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true)
    private String name;                // "ROLE_USER", "ROLE_ADMIN"

    private String description;
}

// ========== 2. UserDetailsService 实现 ==========
@Service
@RequiredArgsConstructor
public class CustomUserDetailsService implements UserDetailsService {

    private final UserRepository userRepository;

    @Override
    public UserDetails loadUserByUsername(String username)
            throws UsernameNotFoundException {
        User user = userRepository.findByUsername(username)
            .orElseThrow(() -> new UsernameNotFoundException("用户不存在: " + username));

        return new org.springframework.security.core.userdetails.User(
            user.getUsername(),
            user.getPassword(),
            user.isEnabled(),
            true,                       // accountNonExpired
            true,                       // credentialsNonExpired
            true,                       // accountNonLocked
            user.getRoles().stream()
                .map(role -> new SimpleGrantedAuthority(role.getName()))
                .collect(Collectors.toSet())
        );
    }
}

// ========== 3. SecurityConfig 配置 ==========
@Configuration
@EnableWebSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    private final CustomUserDetailsService userDetailsService;

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/register", "/login", "/css/**", "/js/**").permitAll()
                .requestMatchers("/admin/**").hasRole("ADMIN")
                .anyRequest().authenticated()
            )
            .formLogin(form -> form
                .loginPage("/login")
                .defaultSuccessUrl("/dashboard")
                .permitAll()
            )
            .logout(logout -> logout
                .logoutSuccessUrl("/login?logout")
                .permitAll()
            )
            .rememberMe(remember -> remember
                .key("uniqueAndSecret")
                .tokenValiditySeconds(86400 * 7)  // 7 天
                .userDetailsService(userDetailsService)
            );
        return http.build();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public AuthenticationProvider authenticationProvider() {
        DaoAuthenticationProvider provider = new DaoAuthenticationProvider();
        provider.setUserDetailsService(userDetailsService);
        provider.setPasswordEncoder(passwordEncoder());
        return provider;
    }
}

// ========== 注册用户 ==========
@Service
public class AuthService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    @Transactional
    public User register(RegisterRequest request) {
        if (userRepository.findByUsername(request.getUsername()).isPresent()) {
            throw new RuntimeException("用户名已存在");
        }

        User user = new User();
        user.setUsername(request.getUsername());
        user.setPassword(passwordEncoder.encode(request.getPassword()));
        user.setEmail(request.getEmail());
        user.setEnabled(true);

        // 默认角色
        Role userRole = roleRepository.findByName("ROLE_USER")
            .orElseThrow(() -> new RuntimeException("角色不存在"));
        user.setRoles(Set.of(userRole));

        return userRepository.save(user);
    }
}
```


## 方法安全


```
// ========== 方法级别安全 ==========
// 在方法上直接加注解控制权限

// ========== 1. 启用方法安全 ==========
@Configuration
@EnableMethodSecurity                     // Spring Security 6+ (替代 @EnableGlobalMethodSecurity)
public class MethodSecurityConfig {
    // securedEnabled = true — @Secured
    // jsr250Enabled = true — @RolesAllowed
}

// 或:
@EnableGlobalMethodSecurity(prePostEnabled = true)  // Spring Security 5

// ========== 2. @PreAuthorize ==========
// 方法执行前检查权限

@RestController
@RequestMapping("/api/users")
public class UserController {

    // 需要 USER 角色
    @PreAuthorize("hasRole('USER')")
    @GetMapping("/{id}")
    public User getUser(@PathVariable Long id) { ... }

    // 需要 ADMIN 角色
    @PreAuthorize("hasRole('ADMIN')")
    @PostMapping
    public User createUser(@RequestBody CreateUserRequest request) { ... }

    // 检查当前用户 ID 是否匹配 (只能操作自己的数据)
    @PreAuthorize("#id == authentication.principal.id")
    @PutMapping("/{id}")
    public User updateUser(@PathVariable Long id, @RequestBody UpdateUserRequest req) {
        ...
    }

    // 复杂 SpEL 条件
    @PreAuthorize("hasRole('ADMIN') or (#username == authentication.principal.username)")
    @GetMapping("/by-username/{username}")
    public User getUserByUsername(@PathVariable String username) { ... }

    // 引用 Bean 方法
    @PreAuthorize("@userSecurity.canDelete(#id, authentication)")
    @DeleteMapping("/{id}")
    public void deleteUser(@PathVariable Long id) { ... }
}

// ========== 3. @PostAuthorize ==========
// 方法执行后检查 (可用于检查返回值)

@PostAuthorize("returnObject.owner == authentication.principal.username")
@GetMapping("/documents/{id}")
public Document getDocument(@PathVariable Long id) { ... }

// ========== 4. @Secured ==========
// 简单角色检查 (无需 ROLE_ 前缀)

@Secured("ROLE_ADMIN")
public void adminOnlyOperation() { ... }

@Secured({"ROLE_ADMIN", "ROLE_MANAGER"})
public void adminOrManagerOperation() { ... }

// ========== 5. 自定义权限检查器 ==========
@Component("userSecurity")
public class UserSecurity {

    public boolean canDelete(Long id, Authentication authentication) {
        UserDetails currentUser = (UserDetails) authentication.getPrincipal();
        // 管理员可以删除任何人
        if (currentUser.getAuthorities().contains(new SimpleGrantedAuthority("ROLE_ADMIN"))) {
            return true;
        }
        // 普通用户只能删除自己的账户
        return currentUser.getUsername().equals(userRepository.findById(id)
            .map(User::getUsername)
            .orElse(null));
    }
}
```


## Security 架构深入


```
// ========== Security 过滤器链 ==========
// 请求通过多个过滤器 (顺序重要):

// 1. SecurityContextPersistenceFilter    — 从 Session 恢复 SecurityContext
// 2. LogoutFilter                        — 处理登出
// 3. UsernamePasswordAuthenticationFilter — 处理表单登录
// 4. DefaultLoginPageGeneratingFilter    — 生成默认登录页
// 5. BasicAuthenticationFilter           — HTTP Basic 认证
// 6. RequestCacheAwareFilter             — 缓存请求 (登录后恢复)
// 7. SecurityContextHolderAwareRequestFilter — 包装 HttpServletRequest
// 8. AnonymousAuthenticationFilter       — 匿名用户
// 9. SessionManagementFilter             — 会话管理
// 10. ExceptionTranslationFilter         — 处理 AccessDeniedException / AuthenticationException
// 11. FilterSecurityInterceptor          — 最终授权决策

// ========== SecurityContextHolder ==========
// 存储当前登录用户信息

// 获取当前用户:
@Service
public class SecurityService {

    // 方式 1: SecurityContextHolder
    public String getCurrentUsername() {
        Authentication authentication = SecurityContextHolder.getContext()
            .getAuthentication();
        if (authentication == null || !authentication.isAuthenticated()) {
            return null;
        }
        return authentication.getName();
    }

    // 方式 2: 方法注入
    @GetMapping("/me")
    public UserDetails currentUser(
            @AuthenticationPrincipal UserDetails userDetails) {
        return userDetails;
    }

    // 方式 3: 自动注入
    @GetMapping("/me2")
    public Authentication currentAuth(Authentication authentication) {
        return authentication;
    }
}

// ========== 认证流程 ==========
// 1. 用户提交用户名/密码
// 2. UsernamePasswordAuthenticationFilter 创建 UsernamePasswordAuthenticationToken
// 3. AuthenticationManager 委托 AuthenticationProvider
// 4. DaoAuthenticationProvider 调用 UserDetailsService.loadUserByUsername()
// 5. PasswordEncoder.matches() 验证密码
// 6. 创建已认证的 Authentication (含 GrantedAuthority)
// 7. SecurityContextHolder.setContext(securityContext)
// 8. Session 保存认证状态

// ========== 自定义 AuthenticationProvider ==========
@Component
public class CustomAuthenticationProvider implements AuthenticationProvider {

    private final UserDetailsService userDetailsService;
    private final PasswordEncoder passwordEncoder;

    @Override
    public Authentication authenticate(Authentication authentication)
            throws AuthenticationException {
        String username = authentication.getName();
        String password = authentication.getCredentials().toString();

        UserDetails user = userDetailsService.loadUserByUsername(username);

        if (!passwordEncoder.matches(password, user.getPassword())) {
            throw new BadCredentialsException("密码错误");
        }

        // 可添加额外验证: 验证码、IP 限制、账号状态等
        if (!user.isEnabled()) {
            throw new DisabledException("账号已禁用");
        }

        return new UsernamePasswordAuthenticationToken(
            user, password, user.getAuthorities());
    }

    @Override
    public boolean supports(Class<?> authentication) {
        return UsernamePasswordAuthenticationToken.class.isAssignableFrom(authentication);
    }
}
```


## 最佳实践


```
// ========== Security 最佳实践 ==========

// ========== 1. 始终使用 BCryptPasswordEncoder ==========
// 加盐哈希, 自动处理 salt
// cost factor 默认为 10, 生产建议 12-13

@Bean
public PasswordEncoder passwordEncoder() {
    return new BCryptPasswordEncoder(12);  // 强度 12
}

// ========== 2. CSRF 保护 ==========
// 浏览器表单提交应启用 CSRF
// REST API (Bearer Token) 可禁用

// ========== 3. 会话管理 ==========
.sessionManagement(session -> session
    .sessionCreationPolicy(SessionCreationPolicy.STATELESS) // JWT 时无状态
    .maximumSessions(1)                                      // 单用户登录
    .maxSessionsPreventsLogin(true)                          // 拒绝新登录
)

// ========== 4. 安全头 ==========
http.headers(headers -> headers
    .frameOptions(frame -> frame.deny())
    .xssProtection(xss -> xss.enable())
    .contentSecurityPolicy(csp -> csp
        .policyDirectives("default-src 'self'"))
)

// ========== 5. 密码规则 ==========
// 最小 8 位, 含大小写字母+数字+特殊字符
// 不在日志中打印密码
// 不使用常见密码

// ========== 6. 权限最小化 ==========
// 默认拒绝, 按需开放
// 方法级别 @PreAuthorize 细粒度控制
// 避免粗粒度的角色控制

// ========== 7. 登录限制 ==========
// 失败次数限制 (防暴力破解)
// 验证码 (图形验证码/SMS)
// IP 白名单

// ========== 8. 日志审计 ==========
// 记录登录成功/失败事件
// 记录敏感操作
// 使用 Spring Security 事件机制

@Component
public class SecurityLogger {

    @EventListener
    public void handleAuthenticationSuccess(
            AuthenticationSuccessEvent event) {
        log.info("用户 {} 登录成功", event.getAuthentication().getName());
    }

    @EventListener
    public void handleAuthenticationFailure(
            AbstractAuthenticationFailureEvent event) {
        log.warn("用户 {} 登录失败: {}",
            event.getAuthentication().getName(),
            event.getException().getMessage());
    }
}

// ========== 9. 密码加密存储 ==========
// 绝不存明文
// BCrypt 自动加盐
// 即使密码相同, 存储的哈希也不同
```


> **Note:** 💡 Security 要点: SecurityFilterChain 替代 WebSecurityConfigurerAdapter; PasswordEncoder 用 BCrypt; @EnableMethodSecurity 方法安全; @PreAuthorize SpEL 表达式; UserDetailsService 从数据库加载用户; 表单登录/HTTP Basic/RememberMe; CSRF 浏览器保护; 会话管理/安全头; 事件审计日志。


## 练习


<!-- Converted from: 21_Spring Security 入门.html -->
