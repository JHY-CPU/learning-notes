# 项目实战 Go + Redis 短链接


## 📦 项目实战 5: Go + Redis 短链接


Gin 路由框架、Base62 编码算法、Redis 键值存储、301/302 重定向、点击统计与 Dockerfile。


## 项目结构


```
// shorturl/
// ├── main.go              # 入口
// ├── config/
// │   └── config.go        # 配置
// ├── handler/
// │   └── url.go           # HTTP 处理器
// ├── service/
// │   └── url.go           # 业务逻辑
// ├── store/
// │   └── redis.go         # Redis 存储
// ├── Dockerfile
// └── docker-compose.yml

// ========== 配置 ==========
// config/config.go
type Config struct {
    ServerPort string
    RedisAddr  string
    RedisPass  string
    BaseURL    string
}

func Load() *Config {
    return &Config{
        ServerPort: getEnv("PORT", "8080"),
        RedisAddr:  getEnv("REDIS_ADDR", "localhost:6379"),
        RedisPass:  getEnv("REDIS_PASS", ""),
        BaseURL:    getEnv("BASE_URL", "http://localhost:8080"),
    }
}

func getEnv(key, def string) string {
    if v := os.Getenv(key); v != "" {
        return v
    }
    return def
}

// ========== Redis 存储 ==========
// store/redis.go
type URLStore struct {
    client *redis.Client
}

func NewURLStore(addr, pass string) *URLStore {
    client := redis.NewClient(&redis.Options{
        Addr:         addr,
        Password:     pass,
        DB:           0,
        PoolSize:     10,
        MinIdleConns: 3,
    })
    return &URLStore{client: client}
}

// 存储短链接 → 长链接
func (s *URLStore) Save(shortCode, longURL string, expiry time.Duration) error {
    return s.client.Set(ctx, "url:"+shortCode, longURL, expiry).Err()
}

// 获取长链接
func (s *URLStore) Get(shortCode string) (string, error) {
    return s.client.Get(ctx, "url:"+shortCode).Result()
}

// 增加点击计数
func (s *URLStore) IncrClick(shortCode string) error {
    return s.client.Incr(ctx, "click:"+shortCode).Err()
}

// 获取点击统计
func (s *URLStore) GetClicks(shortCode string) (int64, error) {
    return s.client.Get(ctx, "click:"+shortCode).Int64()
}

// 检查短码是否存在
func (s *URLStore) Exists(shortCode string) (bool, error) {
    return s.client.Exists(ctx, "url:"+shortCode).Result()
}
```


## Base62 与业务逻辑


```
// ========== Base62 编码 ==========
// service/url.go
const base62Chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

// ID → Base62 (短码)
func toBase62(id uint64) string {
    if id == 0 {
        return string(base62Chars[0])
    }

    var result []byte
    for id > 0 {
        result = append(result, base62Chars[id%62])
        id /= 62
    }

    // 反转
    for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
        result[i], result[j] = result[j], result[i]
    }

    return string(result)
}

// Base62 → ID (可选, 用于分析)
func fromBase62(s string) uint64 {
    var id uint64
    for _, c := range s {
        id = id * 62
        switch {
        case c >= '0' && c <= '9':
            id += uint64(c - '0')
        case c >= 'A' && c <= 'Z':
            id += uint64(c - 'A' + 10)
        case c >= 'a' && c <= 'z':
            id += uint64(c - 'a' + 36)
        }
    }
    return id
}

// ========== 业务服务 ==========
type URLService struct {
    store   *URLStore
    counter *Counter // 发号器
    baseURL string
}

func NewURLService(store *URLStore, baseURL string) *URLService {
    return &URLService{
        store:   store,
        counter: NewCounter(store.client),
        baseURL: baseURL,
    }
}

// 创建短链接
func (s *URLService) Create(longURL string, expiry time.Duration) (string, error) {
    // 1. 获取自增 ID
    id, err := s.counter.Next()
    if err != nil {
        return "", err
    }

    // 2. 编码为短码
    code := toBase62(id)

    // 3. 存储到 Redis
    if err := s.store.Save(code, longURL, expiry); err != nil {
        return "", err
    }

    // 4. 返回完整短链接
    return fmt.Sprintf("%s/%s", s.baseURL, code), nil
}

// 访问短链接
func (s *URLService) Visit(code string) (string, error) {
    longURL, err := s.store.Get(code)
    if err != nil {
        return "", err
    }

    // 异步增加点击数
    go s.store.IncrClick(code)

    return longURL, nil
}

// ========== 发号器 (Redis INCR) ==========
type Counter struct {
    client *redis.Client
    key    string
}

func NewCounter(client *redis.Client) *Counter {
    return &Counter{client: client, key: "url:counter"}
}

func (c *Counter) Next() (uint64, error) {
    return c.client.Incr(ctx, c.key).Uint64()
}
```


## HTTP 处理器与 Docker


```
// ========== HTTP 处理器 ==========
// handler/url.go
type URLHandler struct {
    service *URLService
}

// POST /shorten — 创建短链接
// Body: {"url": "https://example.com/long/url", "expiry_days": 30}
func (h *URLHandler) Create(c *gin.Context) {
    var req struct {
        URL        string `json:"url" binding:"required,url"`
        ExpiryDays int    `json:"expiry_days"`
    }

    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": "无效的 URL"})
        return
    }

    expiry := 0 * time.Second
    if req.ExpiryDays > 0 {
        expiry = time.Duration(req.ExpiryDays) * 24 * time.Hour
    }

    shortURL, err := h.service.Create(req.URL, expiry)
    if err != nil {
        c.JSON(500, gin.H{"error": "创建失败"})
        return
    }

    c.JSON(201, gin.H{
        "short_url":  shortURL,
        "long_url":   req.URL,
        "expiry_days": req.ExpiryDays,
    })
}

// GET /:code — 重定向
func (h *URLHandler) Redirect(c *gin.Context) {
    code := c.Param("code")

    longURL, err := h.service.Visit(code)
    if err != nil {
        c.JSON(404, gin.H{"error": "链接不存在或已过期"})
        return
    }

    c.Redirect(http.StatusMovedPermanently, longURL) // 301
}

// GET /:code/stats — 统计
func (h *URLHandler) Stats(c *gin.Context) {
    code := c.Param("code")
    clicks, _ := h.store.GetClicks(code)
    c.JSON(200, gin.H{"code": code, "clicks": clicks})
}

// ========== Main 入口 ==========
// main.go
func main() {
    cfg := config.Load()
    store := store.NewURLStore(cfg.RedisAddr, cfg.RedisPass)
    svc := service.NewURLService(store, cfg.BaseURL)
    h := &handler.URLHandler{Service: svc, Store: store}

    r := gin.Default()
    r.POST("/shorten", h.Create)
    r.GET("/:code", h.Redirect)
    r.GET("/:code/stats", h.Stats)
    r.GET("/health", func(c *gin.Context) { c.JSON(200, gin.H{"status": "ok"}) })

    r.Run(":" + cfg.ServerPort)
}

// ========== Dockerfile ==========
// FROM golang:1.22-alpine AS builder
// WORKDIR /app
// COPY go.mod go.sum ./
// RUN go mod download
// COPY . .
// RUN CGO_ENABLED=0 go build -o shorturl .
//
// FROM alpine:3.19
// RUN apk add --no-cache ca-certificates
// COPY --from=builder /app/shorturl /usr/local/bin/
// EXPOSE 8080
// CMD ["shorturl"]
```


> **Note:** 💡 Go + Redis 短链接要点: Base62 编码可逆; Redis INCR 发号器; 301 Moved Permanently 浏览器缓存; 302 Found 用于统计 (不缓存); 点击统计异步 INCR; 可选过期 TTL; Docker 多阶段构建; 短码 7 位支持 62^7 ≈ 3.5万亿 组合。


## 练习


<!-- Converted from: 4_项目实战 Go  Redis 短链接.html -->
