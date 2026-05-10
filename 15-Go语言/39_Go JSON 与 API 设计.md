# Go JSON 与 API 设计


## 📦 Go JSON 与 API 设计


JSON 编码解码、自定义 Marshal/Unmarshal、流式 JSON、API 响应设计、版本化、统一错误格式、分页与过滤。


## JSON 编码与解码


```
// ========== 基础 JSON 操作 ==========
import "encoding/json"

type User struct {
    ID        int       `json:"id"`
    Name      string    `json:"name"`
    Email     string    `json:"email,omitempty"`   // 空值跳过
    Password  string    `json:"-"`                 // 忽略
    CreatedAt time.Time `json:"created_at"`
    Role     string    `json:"role,omitempty"`
}

// Marshal (编码)
func marshalExample() {
    u := User{ID: 1, Name: "John", Email: "john@example.com", CreatedAt: time.Now()}
    data, err := json.Marshal(u)
    // data, err := json.MarshalIndent(u, "", "  ")  // 格式化

    _ = err
    os.Stdout.Write(data)
    // {"id":1,"name":"John","email":"john@example.com","created_at":"2025-01-01T12:00:00Z"}
}

// Unmarshal (解码)
func unmarshalExample() {
    data := []byte(`{"id":1,"name":"John","email":"john@example.com"}`)
    var u User
    if err := json.Unmarshal(data, &u); err != nil {
        log.Fatal(err)
    }
    fmt.Println(u.Name)  // John
}

// ========== 泛型 JSON 处理 ==========
// 使用 map[string]interface{} 或 any
func genericJSON() {
    data := []byte(`{"name":"John","age":30,"active":true,"tags":["go","json"]}`)
    var result map[string]interface{}
    json.Unmarshal(data, &result)
    fmt.Println(result["name"])  // John (float64)
    fmt.Println(result["age"])   // float64 !!

    // 类型断言
    name := result["name"].(string)
    age := result["age"].(float64)
    _ = name
    _ = age
}

// ========== JSON 到结构体 (灵活解码) ==========
// 使用 json.RawMessage 延迟解码
type FlexibleResponse struct {
    Status  string           `json:"status"`
    Data    json.RawMessage `json:"data"`    // 原始 JSON, 稍后解码
}

func handleAPIResponse(body []byte) {
    var resp FlexibleResponse
    json.Unmarshal(body, &resp)

    switch resp.Status {
    case "user":
        var user User
        json.Unmarshal(resp.Data, &user)
    case "list":
        var users []User
        json.Unmarshal(resp.Data, &users)
    }
}
```


## 自定义序列化


```
// ========== 自定义 MarshalJSON ==========
type CustomTime time.Time

func (ct CustomTime) MarshalJSON() ([]byte, error) {
    t := time.Time(ct)
    if t.IsZero() {
        return []byte(`""`), nil
    }
    return []byte(`"` + t.Format("2006-01-02 15:04:05") + `"`), nil
}

func (ct *CustomTime) UnmarshalJSON(data []byte) error {
    s := strings.Trim(string(data), `"`)
    if s == "" || s == "null" {
        return nil
    }
    t, err := time.Parse("2006-01-02 15:04:05", s)
    if err != nil {
        return err
    }
    *ct = CustomTime(t)
    return nil
}

// ========== 枚举类型序列化 ==========
type Status int

const (
    StatusActive   Status = iota + 1  // 1
    StatusInactive                    // 2
    StatusBanned                      // 3
)

var statusNames = map[Status]string{
    StatusActive:   "active",
    StatusInactive: "inactive",
    StatusBanned:   "banned",
}

var statusValues = map[string]Status{
    "active":   StatusActive,
    "inactive": StatusInactive,
    "banned":   StatusBanned,
}

func (s Status) MarshalJSON() ([]byte, error) {
    name, ok := statusNames[s]
    if !ok {
        return nil, fmt.Errorf("unknown status: %d", s)
    }
    return json.Marshal(name)
}

func (s *Status) UnmarshalJSON(data []byte) error {
    var name string
    if err := json.Unmarshal(data, &name); err != nil {
        return err
    }
    val, ok := statusValues[name]
    if !ok {
        return fmt.Errorf("unknown status: %s", name)
    }
    *s = val
    return nil
}

// ========== 空数组 vs null ==========
type Response struct {
    Items []string `json:"items"`
}

// nil → null, 空切片 → []
func fixNilSlice() {
    // 确保返回 [] 而不是 null
    resp := Response{Items: []string{}}  // 空切片
    data, _ := json.Marshal(resp)
    fmt.Println(string(data))  // {"items":[]}

    // 或者在 MarshalJSON 中处理
}

// ========== 时间格式化 ==========
type Event struct {
    Name string    `json:"name"`
    Time time.Time `json:"time"`
}

// 默认: RFC3339
// 自定义格式: 用自定义类型包装 time.Time
```


## 流式 JSON


```
// ========== 流式解码 (Decoder) ==========
// 适用于大文件或 HTTP 流

func streamDecode(reader io.Reader) ([]User, error) {
    dec := json.NewDecoder(reader)
    var users []User

    for {
        var u User
        if err := dec.Decode(&u); err == io.EOF {
            break
        } else if err != nil {
            return nil, err
        }
        users = append(users, u)
    }
    return users, nil
}

// ========== 流式编码 (Encoder) ==========
func streamEncode(writer io.Writer, users []User) error {
    enc := json.NewEncoder(writer)
    for _, u := range users {
        if err := enc.Encode(u); err != nil {
            return err
        }
    }
    return nil
}

// ========== 大 JSON 数组流式处理 ==========
// 逐行处理而不是加载全部到内存
func streamLargeJSON(r io.Reader) error {
    dec := json.NewDecoder(r)

    // 读开头的 [
    if _, err := dec.Token(); err != nil {
        return err
    }

    for dec.More() {
        var u User
        if err := dec.Decode(&u); err != nil {
            return err
        }
        processUser(u)
    }

    // 读结尾的 ]
    if _, err := dec.Token(); err != nil {
        return err
    }
    return nil
}

// ========== JSON 行 (JSON Lines) ==========
// 每行一个 JSON 对象, 适合日志
// {"level":"info","msg":"start"}
// {"level":"error","msg":"failed"}

func readJSONLines(r io.Reader) error {
    scanner := bufio.NewScanner(r)
    for scanner.Scan() {
        var entry map[string]interface{}
        if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
            return err
        }
        fmt.Println(entry["level"], entry["msg"])
    }
    return scanner.Err()
}

func writeJSONLines(w io.Writer, entries []map[string]interface{}) error {
    for _, entry := range entries {
        data, err := json.Marshal(entry)
        if err != nil {
            return err
        }
        if _, err := fmt.Fprintln(w, string(data)); err != nil {
            return err
        }
    }
    return nil
}
```


## API 响应设计


```
// ========== 统一响应格式 ==========
type APIResponse struct {
    Success bool        `json:"success"`
    Code    int         `json:"code"`
    Message string      `json:"message"`
    Data    interface{} `json:"data,omitempty"`
    Error   *APIError   `json:"error,omitempty"`
    Meta    *Meta       `json:"meta,omitempty"`
}

type APIError struct {
    Code    string      `json:"code"`
    Message string      `json:"message"`
    Details interface{} `json:"details,omitempty"`
}

type Meta struct {
    Page       int   `json:"page"`
    PerPage    int   `json:"per_page"`
    Total      int64 `json:"total"`
    TotalPages int   `json:"total_pages"`
}

// ========== 响应辅助函数 ==========
func Success(w http.ResponseWriter, data interface{}) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(APIResponse{
        Success: true,
        Code:    200,
        Message: "success",
        Data:    data,
    })
}

func Created(w http.ResponseWriter, data interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(APIResponse{
        Success: true,
        Code:    201,
        Message: "created",
        Data:    data,
    })
}

func Error(w http.ResponseWriter, status int, code, message string) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(APIResponse{
        Success: false,
        Code:    status,
        Message: message,
        Error: &APIError{
            Code:    code,
            Message: message,
        },
    })
}

func Paginated(w http.ResponseWriter, data interface{}, page, perPage int, total int64) {
    totalPages := int(math.Ceil(float64(total) / float64(perPage)))
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(APIResponse{
        Success: true,
        Code:    200,
        Message: "success",
        Data:    data,
        Meta: &Meta{
            Page:       page,
            PerPage:    perPage,
            Total:      total,
            TotalPages: totalPages,
        },
    })
}

// ========== 分页请求 ==========
type PaginationReq struct {
    Page    int    `json:"page"    form:"page"    binding:"required,min=1"`
    PerPage int    `json:"per_page" form:"per_page" binding:"required,min=1,max=100"`
    Sort    string `json:"sort"    form:"sort"    binding:"omitempty"`
    Order   string `json:"order"   form:"order"   binding:"omitempty,oneof=asc desc"`
}

func (p *PaginationReq) Offset() int {
    return (p.Page - 1) * p.PerPage
}

func (p *PaginationReq) Limit() int {
    return p.PerPage
}

// ========== API 版本化 ==========
// URL 前缀版本
func versionedRouter() {
    r := gin.New()

    v1 := r.Group("/api/v1")
    {
        v1.GET("/users", listUsersV1)
    }

    v2 := r.Group("/api/v2")
    {
        v2.GET("/users", listUsersV2)
    }
}

// ========== 响应封装中间件 ==========
func ResponseMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Set("response", &APIResponse{})
        c.Next()
    }
}

// ========== 错误码体系 ==========
type ErrCode int

const (
    ErrSuccess      ErrCode = 0
    ErrBadRequest   ErrCode = 40000
    ErrValidation   ErrCode = 40001
    ErrUnauthorized ErrCode = 40100
    ErrForbidden    ErrCode = 40300
    ErrNotFound     ErrCode = 40400
    ErrRateLimit    ErrCode = 42900
    ErrInternal     ErrCode = 50000
    ErrServiceUnavail ErrCode = 50300
)

var errMessages = map[ErrCode]string{
    ErrSuccess:      "success",
    ErrBadRequest:   "请求参数错误",
    ErrValidation:   "参数验证失败",
    ErrUnauthorized: "未授权",
    ErrForbidden:    "无权限",
    ErrNotFound:     "资源不存在",
    ErrRateLimit:    "请求太频繁",
    ErrInternal:     "服务器内部错误",
    ErrServiceUnavail: "服务暂不可用",
}
```


> **Note:** 💡 JSON 与 API 要点: json.Marshal/Unmarshal 结构体标签 (json:"name,omitempty/-"); 自定义 MarshalJSON/UnmarshalJSON (时间格式/枚举名称); json.NewDecoder/Encoder 流式处理大 JSON; json.RawMessage 延迟解码; 统一响应格式 {success,code,message,data,meta,error}; 分页 Meta {page,per_page,total,total_pages}; API 版本化 /v1/ /v2/; JSON Lines 逐行读写适合日志; nil vs 空数组: 空切片 emit empty。


## 练习


<!-- Converted from: 39_Go JSON 与 API 设计.html -->
