# Go Web 开发


## 🌐 Go Web 开发


net/http 标准库、ServeMux 路由、中间件模式、Gin 框架、REST API 构建、JSON 响应、请求验证、数据库集成。


## net/http 标准库


```
// ========== Go Web 基础 ==========
// Go 标准库 net/http 提供完整 HTTP 功能

package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
)

// ========== Handler 接口 ==========
// type Handler interface {
//     ServeHTTP(w http.ResponseWriter, r *http.Request)
// }

// 方式 1: 函数 handler
func helloHandler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello, World!")
}

// 方式 2: 结构体实现 Handler
type UserHandler struct {
    db *DB
}

func (h *UserHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    switch r.Method {
    case http.MethodGet:
        h.listUsers(w, r)
    case http.MethodPost:
        h.createUser(w, r)
    default:
        http.Error(w, "Method not allowed", 405)
    }
}

// ========== ServeMux 路由 ==========
// Go 1.22 之前没有路径参数
// Go 1.22+ 支持模式匹配: GET /users/{id}

func main() {
    mux := http.NewServeMux()

    // 精确路径
    mux.HandleFunc("/", homePage)
    mux.HandleFunc("/hello", helloHandler)

    // 路径前缀 (以 / 结尾匹配所有子路径)
    mux.Handle("/api/", apiMiddleware(http.HandlerFunc(apiHandler)))

    // Go 1.22+: 方法 + 路径参数
    mux.HandleFunc("GET /users/{id}", getUser)
    mux.HandleFunc("POST /users", createUser)
    mux.HandleFunc("PUT /users/{id}", updateUser)
    mux.HandleFunc("DELETE /users/{id}", deleteUser)

    // 静态文件服务
    fs := http.FileServer(http.Dir("./static"))
    mux.Handle("/static/", http.StripPrefix("/static/", fs))

    // 启动服务器
    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", mux))
}

// ========== 路径参数 (Go 1.22+) ==========
func getUser(w http.ResponseWriter, r *http.Request) {
    id := r.PathValue("id")
    fmt.Fprintf(w, "User ID: %s", id)
}

// ========== 请求处理 ==========
func homePage(w http.ResponseWriter, r *http.Request) {
    if r.URL.Path != "/" {
        http.NotFound(w, r)
        return
    }
    fmt.Fprint(w, "Home Page")
}

func apiHandler(w http.ResponseWriter, r *http.Request) {
    // 读取查询参数
    page := r.URL.Query().Get("page")
    limit := r.URL.Query().Get("limit")

    // 读取请求头
    token := r.Header.Get("Authorization")

    // 读取请求体
    // body, _ := io.ReadAll(r.Body)
    // defer r.Body.Close()

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "page":  page,
        "limit": limit,
    })
}

// ========== JSON 响应 ==========
func jsonResponse(w http.ResponseWriter, status int, data interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(data)
}

func errorResponse(w http.ResponseWriter, status int, message string) {
    jsonResponse(w, status, map[string]string{"error": message})
}
```


## 中间件


```
// ========== 中间件 ==========

// 中间件类型
type Middleware func(http.Handler) http.Handler

// 链式组合
func Chain(handler http.Handler, middlewares ...Middleware) http.Handler {
    for i := len(middlewares) - 1; i >= 0; i-- {
        handler = middlewares[i](handler)
    }
    return handler
}

// ========== 1. 日志中间件 ==========
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()

        next.ServeHTTP(w, r)

        log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
    })
}

// ========== 2. 恢复中间件 ==========
func RecoveryMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                log.Printf("panic recovered: %v", err)
                http.Error(w, "Internal Server Error", 500)
            }
        }()
        next.ServeHTTP(w, r)
    })
}

// ========== 3. CORS 中间件 ==========
func CORSMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

        if r.Method == "OPTIONS" {
            w.WriteHeader(204)
            return
        }

        next.ServeHTTP(w, r)
    })
}

// ========== 4. 认证中间件 ==========
func AuthMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        if token == "" || !validateToken(token) {
            http.Error(w, `{"error":"unauthorized"}`, 401)
            return
        }
        next.ServeHTTP(w, r)
    })
}

// ========== 5. 带状态的中间件 ==========

// 返回中间件的工厂函数
func RateLimitMiddleware(rpm int) Middleware {
    var mu sync.Mutex
    counts := make(map[string]int)

    go func() {
        for range time.Tick(time.Minute) {
            mu.Lock()
            counts = make(map[string]int)
            mu.Unlock()
        }
    }()

    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            ip := r.RemoteAddr

            mu.Lock()
            count := counts[ip]
            if count >= rpm {
                mu.Unlock()
                http.Error(w, "rate limit exceeded", 429)
                return
            }
            counts[ip]++
            mu.Unlock()

            next.ServeHTTP(w, r)
        })
    }
}

// ========== 使用: ==========
func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("/api/users", handleUsers)

    // 链式应用中间件
    handler := Chain(mux,
        RecoveryMiddleware,
        LoggingMiddleware,
        CORSMiddleware,
        AuthMiddleware,
    )

    http.ListenAndServe(":8080", handler)
}

// ========== 请求上下文 ==========
// 在中间件中传递值到 handler
type contextKey string

const UserKey contextKey = "user"

func AuthMiddlewareWithContext(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        user := validateAndGetUser(token)

        ctx := context.WithValue(r.Context(), UserKey, user)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

// handler 中获取:
// func handler(w http.ResponseWriter, r *http.Request) {
//     user := r.Context().Value(UserKey).(User)
//     jsonResponse(w, 200, user)
// }
```


## Gin 框架


```
// ========== Gin ==========
// 最流行的 Go Web 框架
// go get github.com/gin-gonic/gin

package main

import (
    "net/http"
    "github.com/gin-gonic/gin"
)

func main() {
    // 创建路由
    r := gin.Default()
    // gin.Default 自带 Logger 和 Recovery 中间件

    // ========== 路由 ==========
    r.GET("/ping", func(c *gin.Context) {
        c.JSON(200, gin.H{"message": "pong"})
    })

    // 路径参数
    r.GET("/users/:id", getUser)
    r.GET("/users/:id/orders/:oid", getUserOrder)

    // 查询参数
    r.GET("/search", func(c *gin.Context) {
        q := c.Query("q")           // ?q=keyword
        page := c.DefaultQuery("page", "1")
        c.JSON(200, gin.H{"q": q, "page": page})
    })

    // 分组路由
    api := r.Group("/api")
    {
        api.GET("/users", listUsers)
        api.POST("/users", createUser)

        v2 := api.Group("/v2")
        v2.GET("/users", listUsersV2)
    }

    // 静态文件
    r.Static("/static", "./static")
    r.StaticFile("/favicon.ico", "./resources/favicon.ico")

    // 启动
    r.Run(":8080")
}

// ========== 请求处理 ==========
type CreateUserRequest struct {
    Username string `json:"username" binding:"required,min=3,max=50"`
    Email    string `json:"email" binding:"required,email"`
    Age      int    `json:"age" binding:"gte=0,lte=150"`
}

func getUser(c *gin.Context) {
    id := c.Param("id")
    c.JSON(200, gin.H{"id": id})
}

func createUser(c *gin.Context) {
    var req CreateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }

    // 处理业务...
    user := User{Username: req.Username, Email: req.Email}
    c.JSON(201, gin.H{"data": user})
}

// ========== 中间件 ==========
// 自定义中间件
func AuthRequired() gin.HandlerFunc {
    return func(c *gin.Context) {
        token := c.GetHeader("Authorization")
        if token == "" {
            c.AbortWithStatusJSON(401, gin.H{"error": "unauthorized"})
            return
        }
        // 验证 token...
        c.Set("user_id", "123")
        c.Next()  // 继续处理
    }
}

// 使用:
// r.GET("/protected", AuthRequired(), func(c *gin.Context) {
//     userID := c.GetString("user_id")
//     c.JSON(200, gin.H{"user_id": userID})
// })

// ========== 完整 REST API ==========
func setupRouter() *gin.Engine {
    r := gin.Default()

    // 全局中间件
    r.Use(gin.Recovery())
    r.Use(CORSMiddleware())

    api := r.Group("/api/v1")
    {
        users := api.Group("/users")
        users.GET("", listUsers)
        users.GET("/:id", getUser)
        users.POST("", createUser)
        users.PUT("/:id", updateUser)
        users.DELETE("/:id", deleteUser)
    }

    return r
}

// 运行:
// go run main.go
// 默认: http://localhost:8080
```


## 数据库集成


```
// ========== database/sql ==========
// Go 标准数据库接口

import (
    "database/sql"
    _ "github.com/go-sql-driver/mysql"  // MySQL 驱动
    // _ "github.com/lib/pq"              // PostgreSQL
    // _ "github.com/mattn/go-sqlite3"    // SQLite
)

// ========== 连接 ==========
func initDB() (*sql.DB, error) {
    dsn := "user:password@tcp(localhost:3306)/dbname?charset=utf8mb4&parseTime=true"
    db, err := sql.Open("mysql", dsn)
    if err != nil {
        return nil, err
    }

    db.SetMaxOpenConns(25)       // 最大连接数
    db.SetMaxIdleConns(5)        // 最大空闲连接
    db.SetConnMaxLifetime(5 * time.Minute)  // 连接最大存活时间

    if err = db.Ping(); err != nil {
        return nil, err
    }
    return db, nil
}

// ========== CRUD ==========
type User struct {
    ID       int64     `json:"id"`
    Username string    `json:"username"`
    Email    string    `json:"email"`
    CreatedAt time.Time `json:"created_at"`
}

// 查询单条
func GetUser(db *sql.DB, id int64) (*User, error) {
    user := &User{}
    err := db.QueryRow(
        "SELECT id, username, email, created_at FROM users WHERE id = ?", id,
    ).Scan(&user.ID, &user.Username, &user.Email, &user.CreatedAt)
    if err != nil {
        return nil, err
    }
    return user, nil
}

// 查询列表
func ListUsers(db *sql.DB) ([]User, error) {
    rows, err := db.Query("SELECT id, username, email FROM users ORDER BY id")
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    var users []User
    for rows.Next() {
        var u User
        if err := rows.Scan(&u.ID, &u.Username, &u.Email); err != nil {
            return nil, err
        }
        users = append(users, u)
    }
    return users, rows.Err()
}

// 插入
func CreateUser(db *sql.DB, user *User) error {
    result, err := db.Exec(
        "INSERT INTO users (username, email) VALUES (?, ?)",
        user.Username, user.Email)
    if err != nil {
        return err
    }
    user.ID, _ = result.LastInsertId()
    return nil
}

// ========== 事务 ==========
func TransferMoney(db *sql.DB, fromID, toID int64, amount float64) error {
    tx, err := db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()  // 出错时回滚

    _, err = tx.Exec("UPDATE accounts SET balance = balance - ? WHERE id = ?", amount, fromID)
    if err != nil { return err }

    _, err = tx.Exec("UPDATE accounts SET balance = balance + ? WHERE id = ?", amount, toID)
    if err != nil { return err }

    return tx.Commit()
}

// ========== Go ORM: GORM ==========
// go get gorm.io/gorm
// go get gorm.io/driver/mysql

// type Product struct {
//     gorm.Model
//     Code  string
//     Price uint
// }
//
// db, _ := gorm.Open(mysql.Open(dsn), &gorm.Config{})
// db.AutoMigrate(∏{})
// db.Create(∏{Code: "D42", Price: 100})
// var product Product
// db.First(&product, 1)
// db.Where("code = ?", "D42").First(&product)
```


> **Note:** 💡 Web 开发要点: net/http ServeMux 基础; Handler 接口; 中间件链模式; Gin 框架路由/中间件/参数绑定; database/sql 标准接口; GORM ORM; JSON 响应; 路径参数; 请求上下文传递值; 事务管理; RESTful API 设计。


## 练习


<!-- Converted from: 10_Go Web 开发.html -->
