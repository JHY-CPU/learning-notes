# 38_Cookie与Session

## 核心概念

- **Cookie**：HTTP服务器发送给浏览器的小型数据片段
  - 浏览器保存Cookie，每次请求时自动携带
  - 用于在**无状态**的HTTP协议中维护用户状态
  - Cookie存储在**客户端**
- **Session**：服务器端存储的用户会话信息
  - Session ID通过Cookie传递给客户端
  - Session数据存储在**服务器端**
- **408考试重点**：Cookie的作用、工作机制、与HTTP无状态的关系

## 原理分析

### Cookie工作机制

1. **首次访问**：
   - 客户端发送HTTP请求（无Cookie）
   - 服务器处理请求，在响应中设置Cookie
   - 响应头部：`Set-Cookie: sessionId=abc123`

2. **后续访问**：
   - 客户端在请求中携带Cookie
   - 请求头部：`Cookie: sessionId=abc123`
   - 服务器根据Cookie识别用户

### Cookie类型

| 类型 | 生命周期 | 存储位置 |
|------|---------|---------|
| 会话Cookie | 浏览器关闭时删除 | 内存 |
| 持久Cookie | 到期时间到达时删除 | 硬盘 |

### Cookie属性

- `name=value`：Cookie名称和值
- `expires`：过期时间
- `path`：Cookie适用的路径
- `domain`：Cookie适用的域名
- `secure`：仅通过HTTPS传输
- `HttpOnly`：禁止JavaScript访问

### Session机制

1. **创建Session**：
   - 用户首次访问时，服务器创建Session
   - 生成唯一的Session ID
   - 存储Session数据在服务器

2. **传递Session ID**：
   - 服务器通过Cookie将Session ID发送给客户端
   - `Set-Cookie: JSESSIONID=abc123`

3. **识别用户**：
   - 后续请求携带Session ID
   - 服务器根据Session ID找到对应的Session数据

### Cookie vs Session对比

| 特性 | Cookie | Session |
|------|--------|---------|
| 存储位置 | 客户端 | 服务器端 |
| 安全性 | 较低 | 较高 |
| 大小限制 | 4KB左右 | 无限制 |
| 性能影响 | 小 | 大（服务器存储） |
| 依赖关系 | 无 | 依赖Cookie传递ID |

## 直观理解

**Cookie就像会员卡**：
- 商店（服务器）给你一张会员卡（Cookie）
- 你（客户端）保存这张卡
- 下次来商店，出示会员卡，商店就知道你是谁
- 会员卡信息有限（4KB），但足够识别身份

**Session就像商店的档案**：
- 商店给每个会员建立档案（Session）
- 档案存在商店（服务器端）
- 会员卡（Cookie）上只有档案编号（Session ID）
- 档案信息丰富，但需要商店维护

**记忆技巧**：
- Cookie = "客户端小票"
- Session = "服务器端档案"
- Cookie解决HTTP无状态问题
- Session通过Cookie传递ID

## 协议关联

- **Cookie与HTTP**：Cookie通过HTTP头部（Set-Cookie/Cookie）传递
- **Cookie与HTTP无状态**：Cookie使HTTP可以维护用户状态
- **Cookie与安全**：HttpOnly和Secure属性提高安全性
- **408考点**：
  - Cookie的作用：解决HTTP无状态问题
  - Cookie存储在客户端
  - Session存储在服务器端
  - Cookie通过HTTP头部传递
- **陷阱**：Cookie使HTTP"有状态"，但HTTP本身仍是无状态协议
