# Go Gin 框架深入


## 🚀 Go Gin 框架深入


Gin 路由与参数绑定、中间件机制、请求验证、错误处理、文件上传、分组路由与项目结构。


## Gin 基础与路由


```
// ========== Gin 路由引擎 ==========
// import "github.com/gin-gonic/gin"

func main() {
    r := gin.Default()  // 默认集成 Logger + Recovery 中间件

    // 基本路由
    r.GET("/ping", func(c *gin.Context) {
        c.JSON(200, gin.H{"message": "pong"})
    })

    r.POST("/users", createUser)
    r.PUT("/users/:id", updateUser)
    r.DELETE("/users/:id", deleteUser)

    r.Run(":8080")  // 默认 :8080
}

// ========== 路径参数 ==========
func getUser(c *gin.Context) {
    id := c.Param("id")          // /users/:id → "123"
    name := c.Query("name")      // /users/123?name=john
    page := c.DefaultQuery("page", "1")

    c.JSON(200, gin.H{
        "id":   id,
        "name": name,
        "page": page,
    })
}

// ========== 查询参数绑定 ==========
type Pagination struct {
    Page  int    `form:"page" binding:"required,min=1"`
    Limit int    `form:"limit" binding:"required,min=1,max=100"`
    Sort  string `form:"sort" binding:"omitempty,oneof=asc desc"`
}

func listItems(c *gin.Context) {
    var p Pagination
    if err := c.ShouldBindQuery(&p); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }
    c.JSON(200, p)
}

// ========== 请求体绑定 ==========
type CreateUserRequest struct {
    Name  string `json:"name"  binding:"required,min=2,max=50"`
    Email string `json:"email" binding:"required,email"`
    Age   int    `json:"age"   binding:"required,min=1,max=150"`
}

func createUser(c *gin.Context) {
    var req CreateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }
    c.JSON(201, gin.H{"id": 1, "name": req.Name})
}
```


## Gin 中间件


```
// ========== 自定义中间件 ==========
// Gin 中间件: gin.HandlerFunc
// func(c *gin.Context) { c.Next() / c.Abort() }

// 1. 日志中间件
func Logger() gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        path := c.Request.URL.Path

        c.Next()  // 处理请求

        latency := time.Since(start)
        status := c.Writer.Status()
        log.Printf("%s %s %d %v", c.Request.Method, path, status, latency)
    }
}

// 2. 认证中间件
func AuthMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        token := c.GetHeader("Authorization")
        if token == "" || !strings.HasPrefix(token, "Bearer ") {
            c.AbortWithStatusJSON(401, gin.H{"error": "unauthorized"})
            return
        }

        claims, err := validateToken(token[7:])
        if err != nil {
            c.AbortWithStatusJSON(401, gin.H{"error": "invalid token"})
            return
        }

        // 设置到上下文 (类型安全)
        c.Set("user_id", claims.UserID)
        c.Set("role", claims.Role)
        c.Next()
    }
}

// 3. CORS 中间件
func CORSMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Header("Access-Control-Allow-Origin", "*")
        c.Header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        c.Header("Access-Control-Allow-Headers", "Content-Type,Authorization")

        if c.Request.Method == "OPTIONS" {
            c.AbortWithStatus(204)
            return
        }
        c.Next()
    }
}

// 4. 限流中间件
func RateLimitMiddleware(rps int, burst int) gin.HandlerFunc {
    limiter := rate.NewLimiter(rate.Limit(rps), burst)
    return func(c *gin.Context) {
        if !limiter.Allow() {
            c.AbortWithStatusJSON(429, gin.H{"error": "too many requests"})
            return
        }
        c.Next()
    }
}

// ========== 中间件使用 ==========
func setupRouter() *gin.Engine {
    r := gin.New()  // 无默认中间件

    // 全局中间件
    r.Use(gin.Logger())
    r.Use(gin.Recovery())
    r.Use(CORSMiddleware())
    r.Use(RateLimitMiddleware(100, 200))

    // 路由组 + 中间件
    api := r.Group("/api")
    api.Use(AuthMiddleware())
    {
        api.GET("/users", listUsers)
        api.GET("/users/:id", getUser)
        api.POST("/users", createUser)
    }

    // 公共路由 (无认证)
    r.POST("/login", login)
    r.GET("/health", healthCheck)

    return r
}

// ========== 在中间件中获取值 ==========
func getCurrentUser(c *gin.Context) *User {
    userID, _ := c.Get("user_id")
    role, _ := c.Get("role")
    return &User{ID: userID.(int), Role: role.(string)}
}
```


## 请求验证与错误处理


```
// ========== 自定义验证器 ==========
import "github.com/go-playground/validator/v10"

// 注册自定义验证
func init() {
    if v, ok := binding.Validator.Engine().(*validator.Validate); ok {
        v.RegisterValidation("phone", func(fl validator.FieldLevel) bool {
            phone := fl.Field().String()
            matched, _ := regexp.MatchString(`^1[3-9]\d{9}$`, phone)
            return matched
        })
    }
}

type RegisterRequest struct {
    Username string `json:"username" binding:"required,min=4,max=20"`
    Phone    string `json:"phone"    binding:"required,phone"`
    Password string `json:"password" binding:"required,min=8,max=32"`
    Email    string `json:"email"    binding:"required,email"`
}

// ========== 统一错误处理 ==========
// 错误结构
type APIError struct {
    Code    int         `json:"code"`
    Message string      `json:"message"`
    Details interface{} `json:"details,omitempty"`
}

// 错误处理辅助
func ErrorResponse(c *gin.Context, status int, msg string) {
    c.AbortWithStatusJSON(status, APIError{
        Code:    status,
        Message: msg,
    })
}

func ValidationErrorResponse(c *gin.Context, err error) {
    var ve validator.ValidationErrors
    if errors.As(err, &ve) {
        details := make([]map[string]interface{}, len(ve))
        for i, fe := range ve {
            details[i] = map[string]interface{}{
                "field":   fe.Field(),
                "tag":     fe.Tag(),
                "value":   fe.Value(),
                "message": fmt.Sprintf("%s 字段%s检查不通过", fe.Field(), fe.Tag()),
            }
        }
        c.AbortWithStatusJSON(422, APIError{
            Code:    422,
            Message: "请求参数验证失败",
            Details: details,
        })
        return
    }
    ErrorResponse(c, 400, err.Error())
}

// ========== 全局错误中间件 ==========
func ErrorHandler() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Next()

        // 检查是否有错误
        if len(c.Errors) > 0 {
            err := c.Errors.Last()
            log.Printf("请求错误: %v, path: %s", err, c.Request.URL.Path)
            ErrorResponse(c, 500, "internal server error")
        }
    }
}

// 使用:
// r.Use(ErrorHandler())

// ========== 业务错误码 ==========
type BizCode int

const (
    Success      BizCode = 0
    InvalidParam BizCode = 40001
    NotFound     BizCode = 40004
    AuthFailed   BizCode = 40100
    Forbidden    BizCode = 40300
    ServerError  BizCode = 50000
)

type BizResponse struct {
    Code    BizCode     `json:"code"`
    Message string      `json:"message"`
    Data    interface{} `json:"data,omitempty"`
}

func SuccessData(c *gin.Context, data interface{}) {
    c.JSON(200, BizResponse{Code: Success, Message: "ok", Data: data})
}

func BizError(c *gin.Context, httpStatus int, code BizCode, msg string) {
    c.AbortWithStatusJSON(httpStatus, BizResponse{
        Code:    code,
        Message: msg,
    })
}
```


## 文件上传与静态文件


```
// ========== 单文件上传 ==========
func uploadSingle(c *gin.Context) {
    // 表单字段名 "file"
    file, err := c.FormFile("file")
    if err != nil {
        ErrorResponse(c, 400, "请选择文件")
        return
    }

    // 验证大小 (10MB)
    if file.Size > 10<<20 {
        ErrorResponse(c, 400, "文件大小不能超过 10MB")
        return
    }

    // 验证类型
    allowed := map[string]bool{
        "image/jpeg": true,
        "image/png":  true,
        "image/gif":  true,
    }
    if !allowed[file.Header.Get("Content-Type")] {
        ErrorResponse(c, 400, "只支持 JPEG/PNG/GIF 图片")
        return
    }

    // 保存文件
    dst := fmt.Sprintf("./uploads/%d_%s", time.Now().UnixNano(), file.Filename)
    if err := c.SaveUploadedFile(file, dst); err != nil {
        ErrorResponse(c, 500, "文件保存失败")
        return
    }

    c.JSON(200, gin.H{
        "url":      "/uploads/" + filepath.Base(dst),
        "filename": file.Filename,
        "size":     file.Size,
    })
}

// ========== 多文件上传 ==========
func uploadMultiple(c *gin.Context) {
    form, err := c.MultipartForm()
    if err != nil {
        ErrorResponse(c, 400, "解析表单失败")
        return
    }

    files := form.File["files"]  // 表单字段名 "files"
    var urls []string

    for _, file := range files {
        dst := fmt.Sprintf("./uploads/%d_%s", time.Now().UnixNano(), file.Filename)
        c.SaveUploadedFile(file, dst)
        urls = append(urls, "/uploads/"+filepath.Base(dst))
    }

    c.JSON(200, gin.H{"urls": urls, "count": len(urls)})
}

// ========== 静态文件服务 ==========
func staticFileServer() *gin.Engine {
    r := gin.Default()

    // 静态文件目录
    r.Static("/uploads", "./uploads")
    r.Static("/static", "./static")

    // 单个静态文件
    r.StaticFile("/favicon.ico", "./static/favicon.ico")

    // 自定义文件服务器 (带缓存)
    r.StaticFS("/assets", http.Dir("./assets"))

    return r
}

// ========== 路由分组与项目结构 ==========
func projectStructure() {
    // 推荐项目结构:
    //
    // project/
    // ├── main.go
    // ├── config/
    // │   └── config.go       // 配置
    // ├── handler/
    // │   ├── user.go          // 用户 handler
    // │   └── product.go       // 产品 handler
    // ├── middleware/
    // │   ├── auth.go          // 认证中间件
    // │   └── cors.go          // CORS 中间件
    // ├── model/
    // │   ├── user.go          // 数据模型
    // │   └── product.go
    // ├── repository/
    // │   ├── user_repo.go     // 数据访问
    // │   └── product_repo.go
    // ├── service/
    // │   ├── user_service.go  // 业务逻辑
    // │   └── product_service.go
    // ├── dto/
    // │   ├── request.go       // 请求/响应 DTO
    // │   └── response.go
    // ├── router/
    // │   └── router.go        // 路由注册
    // └── pkg/
    //     ├── response.go      // 统一响应
    //     └── validator.go     // 自定义验证
    _ = nil
}

// ========== 路由组高级用法 ==========
func advancedRouter(r *gin.Engine) {
    // 嵌套路由组
    api := r.Group("/api/v1")
    {
        users := api.Group("/users")
        {
            users.GET("", listUsers)
            users.GET("/:id", getUser)
            users.POST("", createUser)
            users.PUT("/:id", updateUser)
            users.DELETE("/:id", deleteUser)
        }

        products := api.Group("/products")
        products.Use(AuthMiddleware())
        {
            products.GET("", listProducts)
            products.GET("/:id", getProduct)
            products.POST("", createProduct)
        }
    }

    // 路由参数
    r.GET("/welcome", func(c *gin.Context) {
        firstname := c.DefaultQuery("firstname", "Guest")
        lastname := c.Query("lastname")

        c.String(200, "Hello %s %s", firstname, lastname)
    })
}

// ========== 路由注册模式 ==========
// 分离路由注册
func RegisterUserRoutes(r *gin.RouterGroup, h *UserHandler) {
    r.GET("/users", h.List)
    r.GET("/users/:id", h.Get)
    r.POST("/users", h.Create)
    r.PUT("/users/:id", h.Update)
    r.DELETE("/users/:id", h.Delete)
}

func RegisterProductRoutes(r *gin.RouterGroup, h *ProductHandler) {
    r.GET("/products", h.List)
    r.GET("/products/:id", h.Get)
    r.POST("/products", h.Create)
}

// ========== 完整项目入口 ==========
func main() {
    r := gin.Default()

    // 全局中间件
    r.Use(CORSMiddleware())
    r.Use(RateLimitMiddleware(100, 200))

    // 静态文件
    r.Static("/uploads", "./uploads")

    // 健康检查
    r.GET("/health", func(c *gin.Context) {
        c.JSON(200, gin.H{"status": "ok"})
    })

    // API 路由
    v1 := r.Group("/api/v1")
    {
        RegisterUserRoutes(v1, &UserHandler{})
        RegisterProductRoutes(v1, &ProductHandler{})
    }

    r.Run(":8080")
}
```


> **Note:** 💡 Gin 框架要点: gin.Default() = Logger + Recovery; 路由 GET/POST/PUT/DELETE + :param + ?query; ShouldBindJSON/ShouldBindQuery 结构体绑定; 中间件 c.Next()/c.Abort()/c.Set()/c.Get(); 自定义验证器 validator.RegisterValidation; 统一错误处理 APIError + BizCode; 文件上传 FormFile/SaveUploadedFile/MultipartForm; 静态文件 Static/StaticFS; 路由组 Group 嵌套 + 分离注册; 项目结构 handler/service/repository/model 分层。


## 练习


<!-- Converted from: 37_Go Gin 框架深入.html -->
