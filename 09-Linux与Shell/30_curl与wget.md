# curl与wget


## 🌐 curl 与 wget


curl HTTP 请求调试、wget 文件下载、常见场景、API 测试。


## curl 基础


```
// ========== curl 概述 ==========
// curl = Client URL, 强大的 HTTP 请求工具
// 支持 HTTP/HTTPS/FTP/SFTP/IMAP 等 20+ 协议
// 常用于 API 调试、下载文件、测试响应

// ========== GET 请求 ==========
curl https://api.example.com/users        # 基本 GET
curl -v https://api.example.com           # 详细输出 (含请求/响应头)
curl -i https://api.example.com           # 包含响应头
curl -s https://api.example.com           # 静默模式 (无进度条)
curl -sS https://api.example.com          # 静默但显示错误
curl -o output.json https://api.example.com  # 保存到文件
curl -O https://example.com/file.zip      # 保存为文件名
curl -L https://httpbin.org/redirect      # 跟随重定向 (30x)

// ========== POST 请求 ==========
// JSON 数据:
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"name":"Alice","email":"alice@example.com"}'

// 表单数据:
curl -X POST https://api.example.com/login \
  -d "username=alice&password=secret"

// URL 编码:
curl -X POST https://api.example.com/login \
  --data-urlencode "username=alice" \
  --data-urlencode "password=secret"

// 文件上传:
curl -X POST https://api.example.com/upload \
  -F "file=@photo.jpg" \
  -F "description=Vacation photo"

// ========== PUT / PATCH / DELETE ==========
curl -X PUT https://api.example.com/users/1 \
  -H "Content-Type: application/json" \
  -d '{"name":"Alice Updated"}'

curl -X PATCH https://api.example.com/users/1 \
  -H "Content-Type: application/json" \
  -d '{"email":"new@example.com"}'

curl -X DELETE https://api.example.com/users/1
curl -X DELETE https://api.example.com/users/1 -v  # 看响应状态码
```


## curl 进阶


```
// ========== 请求头 ==========
curl -H "Authorization: Bearer token123" https://api.example.com/protected
curl -H "Accept: application/json" https://api.example.com
curl -H "User-Agent: MyApp/1.0" https://api.example.com

// 多个头:
curl -H "Content-Type: application/json" \
  -H "X-API-Key: abc123" \
  -H "X-Request-ID: uuid-1234" \
  https://api.example.com/data

// ========== 认证 ==========
// Basic Auth:
curl -u username:password https://api.example.com/protected
curl -u username:password -X POST https://api.example.com/login

// Bearer Token:
curl -H "Authorization: Bearer eyJhbGciOi..." https://api.example.com/me

// ========== Cookie ==========
curl -c cookies.txt https://example.com/login  # 保存 Cookie
curl -b cookies.txt https://example.com/dashboard  # 使用 Cookie
curl -b "session=abc123" https://example.com  # 直接传 Cookie

// ========== SSL/TLS ==========
curl -k https://self-signed.example.com  # 忽略证书验证
curl --cacert ca.pem https://example.com  # 指定 CA 证书
curl --cert client.pem --key client.key https://example.com  # 客户端证书

// ========== 超时与重试 ==========
curl --connect-timeout 5 https://api.example.com  # 连接超时 5s
curl --max-time 30 https://api.example.com         # 总超时 30s
curl --retry 3 https://api.example.com              # 失败重试 3 次
curl --retry-delay 5 https://api.example.com        # 重试间隔 5s

// ========== 代理 ==========
curl -x http://proxy.example.com:8080 https://api.example.com
curl --socks5 localhost:1080 https://api.example.com
curl -x "" https://api.example.com  # 不使用代理 (覆盖环境变量)

// ========== 调试 ==========
curl -v https://api.example.com        # 详细模式
curl -vvv https://api.example.com       # 极其详细
curl --trace trace.txt https://api.example.com  # 完整跟踪
curl -w "状态码: %{http_code}\n" https://api.example.com  # 仅状态码

// 计时:
curl -w "\n时间: %{time_total}s\n" https://api.example.com
// time_namelookup  time_connect  time_appconnect  time_starttransfer  time_total
```


## wget — 文件下载


```
// ========== wget 基础 ==========
// wget = web get,专注于文件下载
// 优势: 递归下载、断点续传、批量下载

wget https://example.com/file.zip        # 下载文件
wget -O output.zip https://example.com/file.zip  # 指定文件名
wget -P /downloads https://example.com/file.zip  # 指定目录

// ========== 断点续传 ==========
wget -c https://example.com/large-file.zip  # 继续未完成的下载

// ========== 批量下载 ==========
wget -i urls.txt                        # 从文件读取 URL 列表
wget -A pdf,doc -r https://example.com/docs/  # 仅下载 PDF/DOC

// ========== 递归下载 ==========
wget -r -l 2 https://example.com        # 递归,2 层深度
wget -r --no-parent https://example.com/docs/  # 不下载父目录
wget -r -np -nH --cut-dirs=1 https://example.com/docs/  # 镜像网站

// ========== 限速 ==========
wget --limit-rate=200k https://example.com/large-file.zip  # 限速 200KB/s

// ========== 认证 ==========
wget --user=alice --password=secret https://example.com/protected
wget --ask-password https://example.com/protected  # 交互式输入密码

// ========== 后台下载 ==========
wget -b https://example.com/large-file.zip  # 后台运行
tail -f wget-log                           # 查看进度

// ========== curl vs wget ==========
// curl:                          wget:
// 更多协议支持                   擅长递归下载
// 调试 API 更方便                断点续传更好
// 单文件下载                    批量下载更强
// 几乎所有系统预装              预装但不一定
// curl 是 libcurl 的前端        纯命令行工具

// 选择指南:
// 调试 API → curl
// 下载文件 → wget (断点续传/递归)
// 快速测试 → curl -v
// 批量下载 → wget -i
```


## 实战场景


```
// ========== 场景: API 健康检查 ==========
// 脚本中检查服务状态
if curl -f -s http://localhost:8080/health; then
    echo "服务正常"
else
    echo "服务异常"
    exit 1
fi

// ========== 场景: 发送通知 ==========
curl -X POST https://api.telegram.org/bot/sendMessage \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "123456", "text": "部署完成!"}'

// ========== 场景: 下载 GitHub Release ==========
wget -c https://github.com/owner/repo/releases/download/v1.0/app-linux.tar.gz

// ========== 场景: 测试接口响应时间 ==========
curl -w "DNS: %{time_namelookup}s\nTCP: %{time_connect}s\nTLS: %{time_appconnect}s\nTTFB: %{time_starttransfer}s\n总时间: %{time_total}s\n" \
  -o /dev/null -s https://api.example.com

// ========== 场景: 用 curl 做负载测试 ==========
for i in {1..10}; do
    curl -s -o /dev/null -w "%{http_code} %{time_total}\n" http://localhost:3000 &
done
wait
```


> **Note:** 💡 curl 是后端开发调试 API 的必备工具。熟练使用 -v/-X POST/-H/-d 可以完全替代 Postman。alias curl='curl -sS' 可以省去总是加 -s 的麻烦。更推荐用 curlie 或 httpie 做交互式 API 调试。


## 练习


<!-- Converted from: 30_curl与wget.html -->
