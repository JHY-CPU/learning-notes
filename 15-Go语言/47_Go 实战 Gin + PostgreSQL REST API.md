# Go 实战: Gin + PostgreSQL REST API


## 🛠️ Go 实战: Gin + PostgreSQL REST API


完整项目实战: 项目结构、模型定义、数据库迁移、CRUD 路由、认证中间件、统一错误处理、测试。


## 项目结构


```
// ========== 项目结构 ==========
// todo-api/
// ├── main.go
// ├── go.mod
// ├── go.sum
// ├── config/
// │   └── config.go          // 配置管理
// ├── model/
// │   ├── todo.go             // 数据模型
// │   └── user.go
// ├── repository/
// │   ├── todo_repo.go        // 数据访问
// │   └── user_repo.go
// ├── service/
// │   ├── todo_service.go     // 业务逻辑
// │   └── user_service.go
// ├── handler/
// │   ├── todo_handler.go     // HTTP handler
// │   ├── user_handler.go
// │   └── auth_handler.go
// ├── middleware/
// │   ├── auth.go             // 认证中间件
// │   ├── cors.go
// │   └── logger.go
// ├── router/
// │   └── router.go           // 路由注册
// └── pkg/
//     ├── response.go         // 统一响应
//     └── errors.go           // 错误定义

// ========== go.mod ==========
// module todo-api
//
// go 1.22
//
// require (
//     github.com/gin-gonic/gin v1.9.1
//     github.com/golang-jwt/jwt/v5 v5.2.0
//     github.com/lib/pq v1.10.9
//     golang.org/x/crypto v0.17.0
// )

// ========== 配置 ==========
// config/config.go

type Config struct {
    Server   ServerConfig
    Database DBConfig
    JWT      JWTConfig
}

type ServerConfig struct {
    Port string `default:":8080"`
}

type DBConfig struct {
    DSN string `default:"host=localhost port=5432 user=postgres password=postgres dbname=todo sslmode=disable"`
}

type JWTConfig struct {
    Secret string `default:"my-secret-key"`
    Expire int    `default:"24"` // hours
}

func LoadConfig() *Config {
    return &Config{
        Server:   ServerConfig{Port: getEnv("SERVER_PORT", ":8080")},
        Database: DBConfig{DSN: getEnv("DATABASE_DSN", "host=localhost port=5432 user=postgres password=postgres dbname=todo sslmode=disable")},
        JWT:      JWTConfig{Secret: getEnv("JWT_SECRET", "my-secret-key"), Expire: 24},
    }
}

func getEnv(key, fallback string) string {
    if v := os.Getenv(key); v != "" {
        return v
    }
    return fallback
}
```


## 数据模型与迁移


```
// ========== 数据模型 ==========
// model/todo.go

type Todo struct {
    ID        int       `json:"id"`
    Title     string    `json:"title"`
    Completed bool      `json:"completed"`
    UserID    int       `json:"user_id"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
}

type CreateTodoRequest struct {
    Title string `json:"title" binding:"required,min=1,max=200"`
}

type UpdateTodoRequest struct {
    Title     *string `json:"title" binding:"omitempty,min=1,max=200"`
    Completed *bool   `json:"completed"`
}

// model/user.go

type User struct {
    ID        int       `json:"id"`
    Username  string    `json:"username"`
    Email     string    `json:"email"`
    Password  string    `json:"-"`  // 不序列化
    CreatedAt time.Time `json:"created_at"`
}

type RegisterRequest struct {
    Username string `json:"username" binding:"required,min=3,max=50"`
    Email    string `json:"email"    binding:"required,email"`
    Password string `json:"password" binding:"required,min=6,max=100"`
}

type LoginRequest struct {
    Email    string `json:"email"    binding:"required,email"`
    Password string `json:"password" binding:"required"`
}

// ========== 数据库初始化与迁移 ==========
// repository/db.go

func InitDB(dsn string) (*sql.DB, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, fmt.Errorf("open db: %w", err)
    }

    db.SetMaxOpenConns(25)
    db.SetMaxIdleConns(10)
    db.SetConnMaxLifetime(5 * time.Minute)

    if err := db.Ping(); err != nil {
        return nil, fmt.Errorf("ping db: %w", err)
    }

    return db, nil
}

func RunMigrations(db *sql.DB) error {
    schema := `
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS todos (
        id SERIAL PRIMARY KEY,
        title VARCHAR(200) NOT NULL,
        completed BOOLEAN DEFAULT FALSE,
        user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_todos_user_id ON todos(user_id);
    CREATE INDEX IF NOT EXISTS idx_todos_completed ON todos(completed);
    `

    _, err := db.Exec(schema)
    return err
}
```


## 数据访问层


```
// ========== Todo Repository ==========
// repository/todo_repo.go

type TodoRepository struct {
    db *sql.DB
}

func NewTodoRepository(db *sql.DB) *TodoRepository {
    return &TodoRepository{db: db}
}

func (r *TodoRepository) Create(todo *Todo) error {
    query := `INSERT INTO todos (title, user_id) VALUES ($1, $2) RETURNING id, created_at, updated_at`
    return r.db.QueryRow(query, todo.Title, todo.UserID).Scan(&todo.ID, &todo.CreatedAt, &todo.UpdatedAt)
}

func (r *TodoRepository) GetByID(id, userID int) (*Todo, error) {
    query := `SELECT id, title, completed, user_id, created_at, updated_at FROM todos WHERE id = $1 AND user_id = $2`
    t := &Todo{}
    err := r.db.QueryRow(query, id, userID).Scan(&t.ID, &t.Title, &t.Completed, &t.UserID, &t.CreatedAt, &t.UpdatedAt)
    if err == sql.ErrNoRows {
        return nil, fmt.Errorf("todo not found")
    }
    return t, err
}

func (r *TodoRepository) List(userID int, completed *bool, limit, offset int) ([]Todo, error) {
    query := `SELECT id, title, completed, user_id, created_at, updated_at FROM todos WHERE user_id = $1`
    args := []interface{}{userID}
    argIdx := 2

    if completed != nil {
        query += fmt.Sprintf(" AND completed = $%d", argIdx)
        args = append(args, *completed)
        argIdx++
    }

    query += ` ORDER BY created_at DESC`
    query += fmt.Sprintf(" LIMIT $%d OFFSET $%d", argIdx, argIdx+1)
    args = append(args, limit, offset)

    rows, err := r.db.Query(query, args...)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    var todos []Todo
    for rows.Next() {
        var t Todo
        if err := rows.Scan(&t.ID, &t.Title, &t.Completed, &t.UserID, &t.CreatedAt, &t.UpdatedAt); err != nil {
            return nil, err
        }
        todos = append(todos, t)
    }
    return todos, rows.Err()
}

func (r *TodoRepository) Update(todo *Todo) error {
    query := `UPDATE todos SET title=$1, completed=$2, updated_at=NOW() WHERE id=$3 AND user_id=$4`
    result, err := r.db.Exec(query, todo.Title, todo.Completed, todo.ID, todo.UserID)
    if err != nil {
        return err
    }
    n, _ := result.RowsAffected()
    if n == 0 {
        return fmt.Errorf("todo not found")
    }
    return nil
}

func (r *TodoRepository) Delete(id, userID int) error {
    result, err := r.db.Exec(`DELETE FROM todos WHERE id=$1 AND user_id=$2`, id, userID)
    if err != nil {
        return err
    }
    n, _ := result.RowsAffected()
    if n == 0 {
        return fmt.Errorf("todo not found")
    }
    return nil
}
```


## Handler 与路由


```
// ========== 统一响应 ==========
// pkg/response.go

type Response struct {
    Code    int         `json:"code"`
    Message string      `json:"message"`
    Data    interface{} `json:"data,omitempty"`
}

func Success(c *gin.Context, data interface{}) {
    c.JSON(200, Response{Code: 0, Message: "success", Data: data})
}

func Created(c *gin.Context, data interface{}) {
    c.JSON(201, Response{Code: 0, Message: "created", Data: data})
}

func Error(c *gin.Context, httpStatus int, msg string) {
    c.AbortWithStatusJSON(httpStatus, Response{Code: httpStatus, Message: msg})
}

// ========== Todo Handler ==========
// handler/todo_handler.go

type TodoHandler struct {
    service *TodoService
}

func NewTodoHandler(service *TodoService) *TodoHandler {
    return &TodoHandler{service: service}
}

func (h *TodoHandler) List(c *gin.Context) {
    userID := c.GetInt("user_id")
    limit := getQueryInt(c, "limit", 20)
    offset := getQueryInt(c, "offset", 0)

    var completed *bool
    if c.Query("completed") != "" {
        v := c.Query("completed") == "true"
        completed = &v
    }

    todos, err := h.service.List(userID, completed, limit, offset)
    if err != nil {
        Error(c, 500, "获取列表失败")
        return
    }
    Success(c, todos)
}

func (h *TodoHandler) Get(c *gin.Context) {
    id, _ := strconv.Atoi(c.Param("id"))
    userID := c.GetInt("user_id")

    todo, err := h.service.Get(id, userID)
    if err != nil {
        Error(c, 404, "待办事项不存在")
        return
    }
    Success(c, todo)
}

func (h *TodoHandler) Create(c *gin.Context) {
    var req CreateTodoRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        Error(c, 400, "请求参数错误: "+err.Error())
        return
    }

    userID := c.GetInt("user_id")
    todo, err := h.service.Create(req.Title, userID)
    if err != nil {
        Error(c, 500, "创建失败")
        return
    }
    Created(c, todo)
}

func (h *TodoHandler) Update(c *gin.Context) {
    id, _ := strconv.Atoi(c.Param("id"))
    var req UpdateTodoRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        Error(c, 400, "请求参数错误")
        return
    }

    userID := c.GetInt("user_id")
    todo, err := h.service.Update(id, userID, req)
    if err != nil {
        Error(c, 404, "更新失败")
        return
    }
    Success(c, todo)
}

func (h *TodoHandler) Delete(c *gin.Context) {
    id, _ := strconv.Atoi(c.Param("id"))
    userID := c.GetInt("user_id")

    if err := h.service.Delete(id, userID); err != nil {
        Error(c, 404, "删除失败")
        return
    }
    Success(c, nil)
}

func getQueryInt(c *gin.Context, key string, defaultVal int) int {
    v := c.Query(key)
    if v == "" {
        return defaultVal
    }
    n, err := strconv.Atoi(v)
    if err != nil || n < 0 {
        return defaultVal
    }
    return n
}

// ========== 路由注册 ==========
// router/router.go

func SetupRouter(db *sql.DB) *gin.Engine {
    r := gin.Default()

    // 依赖注入
    userRepo := repository.NewUserRepository(db)
    todoRepo := repository.NewTodoRepository(db)
    userService := service.NewUserService(userRepo)
    todoService := service.NewTodoService(todoRepo)
    authHandler := handler.NewAuthHandler(userService)
    todoHandler := handler.NewTodoHandler(todoService)

    // 公共路由
    r.POST("/api/register", authHandler.Register)
    r.POST("/api/login", authHandler.Login)

    // 需要认证的路由
    api := r.Group("/api")
    api.Use(middleware.AuthJWT())
    {
        api.GET("/todos", todoHandler.List)
        api.GET("/todos/:id", todoHandler.Get)
        api.POST("/todos", todoHandler.Create)
        api.PUT("/todos/:id", todoHandler.Update)
        api.DELETE("/todos/:id", todoHandler.Delete)
    }

    return r
}

// main.go

func main() {
    cfg := config.LoadConfig()

    db, err := InitDB(cfg.Database.DSN)
    if err != nil {
        log.Fatalf("数据库连接失败: %v", err)
    }
    defer db.Close()

    if err := RunMigrations(db); err != nil {
        log.Fatalf("数据库迁移失败: %v", err)
    }

    r := router.SetupRouter(db)
    r.Run(cfg.Server.Port)
}
```


> **Note:** 💡 实战要点: 项目分层 (handler/service/repository); 依赖注入通过构造函数; 数据库迁移 DDL 自动建表; JWT 认证中间件获取 user_id; 统一响应格式 {code,message,data}; CRUD 参数验证 binding; 分页 limit/offset/filter; 请求上下文传递用户信息; 错误处理返回友好消息。


## 练习


<!-- Converted from: 47_Go 实战 Gin + PostgreSQL REST API.html -->
