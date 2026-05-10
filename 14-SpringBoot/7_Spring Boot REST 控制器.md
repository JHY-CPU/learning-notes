# Spring Boot REST 控制器


## 🌐 Spring Boot REST 控制器


@RestController/@Controller、@RequestMapping/@GetMapping/@PostMapping、@RequestParam/@PathVariable/@RequestBody、ResponseEntity、统一响应格式。


## @RestController 基础


```
// ========== @RestController ==========
// @Controller + @ResponseBody 的组合
// 方法返回值直接序列化为 JSON/XML

// ========== 第一个 REST 控制器 ==========
@RestController                         // 声明为 REST 控制器
@RequestMapping("/api/users")           // 类级别 URL 前缀
public class UserController {

    // GET /api/users
    @GetMapping
    public List<User> getAllUsers() {
        return userService.findAll();
    }

    // GET /api/users/123
    @GetMapping("/{id}")
    public User getUserById(@PathVariable Long id) {
        return userService.findById(id);
    }

    // POST /api/users
    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)  // 201 Created
    public User createUser(@RequestBody User user) {
        return userService.create(user);
    }

    // PUT /api/users/123
    @PutMapping("/{id}")
    public User updateUser(@PathVariable Long id, @RequestBody User user) {
        return userService.update(id, user);
    }

    // DELETE /api/users/123
    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)  // 204 No Content
    public void deleteUser(@PathVariable Long id) {
        userService.delete(id);
    }
}

// ========== @RequestMapping 简写 ==========
// @GetMapping      = @RequestMapping(method = RequestMethod.GET)
// @PostMapping     = @RequestMapping(method = RequestMethod.POST)
// @PutMapping      = @RequestMapping(method = RequestMethod.PUT)
// @DeleteMapping   = @RequestMapping(method = RequestMethod.DELETE)
// @PatchMapping    = @RequestMapping(method = RequestMethod.PATCH)

// ========== @ResponseStatus ==========
// 200 OK        — @GetMapping (默认)
// 201 Created   — @PostMapping 创建成功
// 204 No Content — @DeleteMapping 删除成功 (无返回体)
// 400 Bad Request  — 参数错误
// 404 Not Found    — 资源不存在
// 500 Internal Server Error — 服务器错误
```


## 请求参数绑定


```
// ========== 请求参数绑定 ==========

@RestController
@RequestMapping("/api/products")
public class ProductController {

    // ========== @PathVariable: 路径变量 ==========
    // GET /api/products/123
    @GetMapping("/{id}")
    public Product getById(@PathVariable Long id) { ... }

    // 多个路径变量
    // GET /api/categories/5/products/10
    @GetMapping("/categories/{catId}/products/{prodId}")
    public Product getProduct(
            @PathVariable Long catId,
            @PathVariable Long prodId) { ... }

    // 指定名称 (当参数名与路径变量名不同时)
    @GetMapping("/{productId}")
    public Product getProduct(@PathVariable("productId") Long id) { ... }

    // ========== @RequestParam: 查询参数 ==========
    // GET /api/products?page=1&size=20
    @GetMapping
    public List<Product> search(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(required = false) String keyword) { ... }

    // 多个值
    // GET /api/products?tags=phone&tags=apple
    @GetMapping("/search")
    public List<Product> search(@RequestParam List<String> tags) { ... }

    // ========== @RequestBody: 请求体 ==========
    @PostMapping
    public Product create(@Valid @RequestBody ProductCreateRequest request) { ... }

    // ========== @RequestHeader: 请求头 ==========
    @GetMapping("/header")
    public String getHeader(
            @RequestHeader("User-Agent") String userAgent,
            @RequestHeader("Authorization") String auth) { ... }

    // ========== @CookieValue: Cookie ==========
    @GetMapping("/me")
    public User getMe(@CookieValue("session_id") String sessionId) { ... }

    // ========== @ModelAttribute: 对象绑定 ==========
    // 将查询参数绑定到对象
    @GetMapping("/filter")
    public List<Product> filter(@ModelAttribute ProductFilter filter) { ... }
}

// 查询参数对象
@Data
public class ProductFilter {
    private String category;
    private BigDecimal minPrice;
    private BigDecimal maxPrice;
    private String sortBy = "id";
    private String sortDir = "asc";
}

// ========== @MatrixVariable (较冷门) ==========
// GET /api/cars;color=red;year=2024
@GetMapping("/{path}")
public List<Car> getCars(
        @MatrixVariable("color") String color,
        @MatrixVariable("year") int year) { ... }
```


## ResponseEntity


```
// ========== ResponseEntity ==========
// 完全控制 HTTP 响应: 状态码 + 响应头 + 响应体

@RestController
@RequestMapping("/api/orders")
public class OrderController {

    // ========== 基本用法 ==========
    @GetMapping("/{id}")
    public ResponseEntity<Order> getOrder(@PathVariable Long id) {
        return orderService.findById(id)
            .map(order -> ResponseEntity.ok(order))          // 200
            .orElse(ResponseEntity.notFound().build());      // 404
    }

    // ========== 创建 ==========
    @PostMapping
    public ResponseEntity<Order> createOrder(@RequestBody Order order) {
        Order created = orderService.create(order);
        URI location = URI.create("/api/orders/" + created.getId());
        return ResponseEntity
            .created(location)                   // 201 + Location 头
            .body(created);
    }

    // ========== 自定义状态 + 响应头 ==========
    @PostMapping("/{id}/cancel")
    public ResponseEntity<Void> cancelOrder(@PathVariable Long id) {
        orderService.cancel(id);
        return ResponseEntity
            .status(HttpStatus.ACCEPTED)         // 202 Accepted
            .header("X-Cancel-Reason", "用户申请")
            .build();
    }

    // ========== 条件响应 ==========
    @GetMapping("/check")
    public ResponseEntity<Void> checkAvailability() {
        if (!orderService.isAvailable()) {
            return ResponseEntity
                .status(HttpStatus.SERVICE_UNAVAILABLE)  // 503
                .header("Retry-After", "120")
                .build();
        }
        return ResponseEntity.ok().build();
    }

    // ========== 缓存控制 ==========
    @GetMapping("/cache")
    public ResponseEntity<Product> getCached(@PathVariable Long id) {
        return ResponseEntity.ok()
            .cacheControl(CacheControl.maxAge(1, TimeUnit.HOURS))
            .eTag("\"v1\"")
            .body(product);
    }
}

// ========== ResponseEntity 静态方法 ==========
ResponseEntity.ok(body)                  // 200
ResponseEntity.created(uri)              // 201
ResponseEntity.accepted()                // 202
ResponseEntity.noContent()               // 204
ResponseEntity.badRequest()              // 400
ResponseEntity.notFound()                // 404
ResponseEntity.status(HttpStatus.N)      // 自定义
ResponseEntity.of(Optional<T>)           // 200 或 404 (Java 9+)
```


## 统一响应格式


```
// ========== 统一响应格式 ==========
// 项目规范: 所有 API 返回统一结构

// ========== 通用响应类 ==========
@Data
@Builder
public class ApiResponse<T> {
    private int code;           // 业务状态码
    private String message;     // 提示信息
    private T data;             // 数据
    private long timestamp;     // 时间戳

    public static <T> ApiResponse<T> success(T data) {
        return ApiResponse.<T>builder()
            .code(200)
            .message("success")
            .data(data)
            .timestamp(System.currentTimeMillis())
            .build();
    }

    public static <T> ApiResponse<T> error(int code, String message) {
        return ApiResponse.<T>builder()
            .code(code)
            .message(message)
            .timestamp(System.currentTimeMillis())
            .build();
    }

    public static <T> ApiResponse<T> created(T data) {
        return ApiResponse.<T>builder()
            .code(201)
            .message("created")
            .data(data)
            .timestamp(System.currentTimeMillis())
            .build();
    }
}

// ========== 分页响应 ==========
@Data
@Builder
public class PageResponse<T> {
    private List<T> content;
    private int page;
    private int size;
    private long total;
    private int totalPages;
    private boolean first;
    private boolean last;
}

// ========== 控制器使用 ==========
@RestController
@RequestMapping("/api/users")
public class UserController {

    @GetMapping
    public ApiResponse<List<User>> getAllUsers() {
        return ApiResponse.success(userService.findAll());
    }

    @GetMapping("/{id}")
    public ApiResponse<User> getUserById(@PathVariable Long id) {
        return userService.findById(id)
            .map(ApiResponse::success)
            .orElse(ApiResponse.error(404, "用户不存在"));
    }

    @PostMapping
    public ApiResponse<User> createUser(@Valid @RequestBody User user) {
        return ApiResponse.created(userService.create(user));
    }

    @GetMapping("/page")
    public PageResponse<User> getUsersByPage(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        Page<User> userPage = userService.findPage(page, size);
        return PageResponse.<User>builder()
            .content(userPage.getContent())
            .page(userPage.getNumber())
            .size(userPage.getSize())
            .total(userPage.getTotalElements())
            .totalPages(userPage.getTotalPages())
            .first(userPage.isFirst())
            .last(userPage.isLast())
            .build();
    }
}
```


## 参数校验


```
// ========== Bean Validation ==========
// spring-boot-starter-validation 自动集成

// ========== DTO 校验 ==========
@Data
public class CreateUserRequest {

    @NotBlank(message = "用户名不能为空")
    @Size(min = 3, max = 50, message = "用户名长度 3-50")
    private String username;

    @NotBlank(message = "密码不能为空")
    @Size(min = 6, max = 100, message = "密码长度 6-100")
    private String password;

    @Email(message = "邮箱格式不正确")
    private String email;

    @NotNull(message = "年龄不能为空")
    @Min(value = 0, message = "年龄不能小于 0")
    @Max(value = 150, message = "年龄不能超过 150")
    private Integer age;

    @Pattern(regexp = "^1[3-9]\\d{9}$", message = "手机号格式不正确")
    private String phone;

    @Future(message = "过期时间必须是将来的时间")
    private LocalDateTime expireTime;
}

// ========== Controller 启用校验 ==========
@RestController
@RequestMapping("/api/users")
@Validated                                    // 类级别 (分组校验)
public class UserController {

    @PostMapping
    public ApiResponse<User> createUser(
            @Valid @RequestBody CreateUserRequest request) {  // @Valid 触发校验
        User user = userService.create(request);
        return ApiResponse.success(user);
    }

    @PutMapping("/{id}")
    public ApiResponse<User> updateUser(
            @PathVariable Long id,
            @Valid @RequestBody UpdateUserRequest request) {
        return ApiResponse.success(userService.update(id, request));
    }

    // 路径变量校验
    @GetMapping("/{id}")
    public ApiResponse<User> getUser(
            @PathVariable @Positive(message = "ID 必须为正数") Long id) {
        return userService.findById(id)
            .map(ApiResponse::success)
            .orElse(ApiResponse.error(404, "用户不存在"));
    }
}

// ========== 分组校验 ==========
public interface CreateGroup {}
public interface UpdateGroup {}

@Data
public class UserRequest {
    @Null(groups = CreateGroup.class)           // 创建时 ID 必须为空
    @NotNull(groups = UpdateGroup.class)        // 更新时 ID 不能为空
    private Long id;

    @NotBlank(groups = CreateGroup.class)
    private String username;
}

// 使用:
@Validated(CreateGroup.class) @RequestBody UserRequest req  // 创建
@Validated(UpdateGroup.class) @RequestBody UserRequest req  // 更新

// ========== 校验注解汇总 ==========
// @NotBlank   — String 不为 null 且不为空
// @NotEmpty   — 集合/数组不为空
// @NotNull    — 任何类型不为 null
// @Size       — 字符串/集合长度范围
// @Min/@Max   — 数字最小值/最大值
// @Positive   — 正数
// @Email      — 邮箱格式
// @Pattern    — 正则表达式
// @Future     — 将来时间
// @Past       — 过去时间
// @AssertTrue — 必须为 true
```


> **Note:** 💡 REST 控制器要点: @RestController = @Controller + @ResponseBody; @GetMapping/@PostMapping/@PutMapping/@DeleteMapping 映射; @PathVariable 路径变量; @RequestParam 查询参数; @RequestBody 请求体; ResponseEntity 完全控制 HTTP; @Valid + @NotBlank/@Email/@Size 参数校验; 最好统一响应格式 ApiResponse 包裹。


## 练习


<!-- Converted from: 7_Spring Boot REST 控制器.html -->
