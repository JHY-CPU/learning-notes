# Nginx 架构与配置


## Nginx 架构与配置


Nginx事件驱动配置


Nginx 是高性能的 HTTP 和反向代理服务器，采用 Master/Worker 多进程架构和事件驱动模型，能处理数万并发连接。


## Master/Worker 进程模型


```
进程架构：
  ┌─────────────────────────────────────────────┐
  │  Master Process（管理进程，root运行）        │
  │  ├── 读取和验证配置                         │
  │  ├── 创建/绑定套接字                        │
  │  ├── 启动/管理 Worker 进程                  │
  │  ├── 热重载配置（不中断服务）               │
  │  ├── 平滑升级二进制                         │
  │  └── 管理日志文件                           │
  │                                             │
  │  Worker Process × CPU核心数（通常1:1）      │
  │  ├── 处理客户端请求                         │
  │  ├── 事件循环（epoll/kqueue）               │
  │  ├── 读取请求、处理、返回响应               │
  │  └── 进程间通过共享内存通信                 │
  └─────────────────────────────────────────────┘

多 Worker 的优势：
  - 每个 Worker 独立运行，互不影响
  - 一个 Worker 崩溃不影响其他 Worker
  - 利用多核 CPU 并行处理
  - 无锁设计（每个 Worker 独立接受连接）
  - 惊群锁（accept_mutex）防止多个 Worker 争抢连接

配置 Worker 数量：
  # 通常设置为 CPU 核心数
  worker_processes auto;

  # 绑定到特定 CPU 核心（减少上下文切换）
  worker_cpu_affinity 0001 0010 0100 1000;

  # Worker 进程最大文件描述符数
  worker_rlimit_nofile 65535;

  # Worker 进程优先级
  worker_priority -10;

事件驱动模型：
  Nginx 使用异步非阻塞事件驱动
  每个 Worker 可处理数千并发连接

  事件循环：
  while (true) {
    events = epoll_wait(epfd, timeout);  // 等待事件
    for event in events {
      if (event.type == READ)  handle_read(event);
      if (event.type == WRITE) handle_write(event);
    }
  }

  // 与传统模型对比
  Apache prefork: 每连接一进程 → 内存开销大
  Apache worker:  每连接一线程 → 上下文切换开销
  Nginx:          单进程多连接 → 最小资源开销

进程间通信：
  - 共享内存：缓存、限流计数器、会话存储
  - 信号：Master 通过信号管理 Worker
  - 套接字对：用于热重载时传递监听套接字
```


## nginx.conf 配置结构


```
配置文件层次：
  ┌─────────────────────────────────────────────┐
  │  main（全局配置）                           │
  │  ├── events { ... }                         │
  │  ├── http { ... }                           │
  │  │   ├── upstream { ... }                   │
  │  │   ├── server { ... }                     │
  │  │   │   ├── location { ... }               │
  │  │   │   │   ├── if { ... }                 │
  │  │   │   │   └── limit_except { ... }       │
  │  │   │   └── location { ... }               │
  │  │   └── server { ... }                     │
  │  └── stream { ... }（四层代理）              │
  └─────────────────────────────────────────────┘

完整配置示例：
  # 全局配置
  user nginx;
  worker_processes auto;
  error_log /var/log/nginx/error.log warn;
  pid /var/run/nginx.pid;

  events {
      worker_connections 10240;  # 每个Worker最大连接数
      use epoll;                 # Linux使用epoll
      multi_accept on;           # 同时接受多个连接
  }

  http {
      # MIME 类型
      include       /etc/nginx/mime.types;
      default_type  application/octet-stream;

      # 日志格式
      log_format main '$remote_addr - $remote_user [$time_local] '
                      '"$request" $status $body_bytes_sent '
                      '"$http_referer" "$http_user_agent" '
                      'rt=$request_time';

      # 基本设置
      sendfile on;
      tcp_nopush on;
      tcp_nodelay on;
      keepalive_timeout 65;
      types_hash_max_size 2048;

      # Gzip 压缩
      gzip on;
      gzip_types text/plain application/json application/javascript text/css;

      # 虚拟主机
      server {
          listen 80;
          server_name www.example.com;

          # 重定向到 HTTPS
          return 301 https://$host$request_uri;
      }

      server {
          listen 443 ssl http2;
          server_name www.example.com;

          ssl_certificate     /etc/nginx/ssl/cert.pem;
          ssl_certificate_key /etc/nginx/ssl/key.pem;

          location / {
              root /var/www/html;
              index index.html;
          }

          location /api {
              proxy_pass http://backend;
          }
      }

      # 引入其他配置
      include /etc/nginx/conf.d/*.conf;
  }

指令上下文：
  main:    worker_processes, user, error_log, pid
  events:  worker_connections, use, multi_accept
  http:    server, upstream, include, log_format
  server:  listen, server_name, location, ssl
  location: proxy_pass, root, index, try_files
```


## location 匹配规则


```
匹配语法与优先级（从高到低）：
  1. =  精确匹配
  2. ^~ 前缀匹配（匹配后不检查正则）
  3. ~  正则匹配（区分大小写）
  4. ~* 正则匹配（不区分大小写）
  5. /  通用前缀匹配（最长匹配）

示例：
  # 精确匹配，优先级最高
  location = / {
      return 200 "首页";
  }

  # 匹配 /api 开头的请求，不继续检查正则
  location ^~ /api {
      proxy_pass http://api_backend;
  }

  # 正则匹配，区分大小写
  location ~ \.php$ {
      fastcgi_pass php_backend;
  }

  # 正则匹配，不区分大小写
  location ~* \.(jpg|jpeg|png|gif|ico)$ {
      expires 30d;
  }

  # 通用前缀匹配（最长前缀）
  location / {
      root /var/www/html;
      try_files $uri $uri/ /index.html;
  }

匹配过程：
  1. 先检查 = 精确匹配，匹配则直接使用
  2. 再检查所有前缀匹配，记录最长的
  3. 如果最长前缀是 ^~，直接使用
  4. 否则检查正则匹配（按配置顺序）
  5. 正则匹配成功则使用
  6. 正则都没匹配到则用最长前缀

try_files 指令：
  # 按顺序尝试文件，都不存在则返回最后一个
  try_files $uri $uri/ /index.html;
  # 1. 尝试 $uri 文件
  # 2. 尝试 $uri/ 目录
  # 3. 都不存在则返回 /index.html

  # 用于 SPA 单页应用
  location / {
      try_files $uri $uri/ /index.html;
  }

  # 用于 API 代理
  location /api {
      try_files $uri @backend;
  }
  location @backend {
      proxy_pass http://api_server;
  }

正则捕获：
  # 用 () 捕获，通过 $1, $2 引用
  location ~ ^/user/(\d+)$ {
      proxy_pass http://backend/user?id=$1;
  }

  location ~ ^/product/([a-z]+)/(\d+)$ {
      # $1 = category, $2 = id
      proxy_pass http://backend/product?category=$1&id=$2;
  }
```


## Nginx 内置变量


```
请求相关变量：
  $request_method    - 请求方法 (GET, POST, ...)
  $request_uri       - 完整请求URI（含参数）
  $uri               - 请求URI（不含参数）
  $args              - 查询参数字符串
  $query_string      - 同 $args
  $is_args           - 有参数为 ?，否则为空
  $request_filename  - 请求文件的完整路径
  $document_root     - 当前请求的根目录
  $request_time      - 请求处理时间（秒）

客户端相关变量：
  $remote_addr       - 客户端IP地址
  $remote_port       - 客户端端口
  $http_user_agent   - User-Agent 头部
  $http_referer      - Referer 头部
  $http_x_forwarded_for - X-Forwarded-For 头部
  $http_cookie       - Cookie 头部
  $cookie_name       - 指定Cookie值

服务端相关变量：
  $server_name       - 匹配的server_name
  $server_port       - 服务端端口
  $server_protocol   - 协议版本 (HTTP/1.1)
  $scheme            - 协议 (http/https)
  $host              - 请求的Host头部
  $hostname          - 服务器主机名

连接相关变量：
  $connection        - 连接序号
  $connection_requests - 当前连接的请求数
  $ssl_protocol      - SSL协议版本
  $ssl_cipher        - SSL加密算法

响应相关变量：
  $status            - 响应状态码
  $body_bytes_sent   - 发送的body字节数
  $bytes_sent        - 发送的总字节数
  $request_length    - 请求长度
  $request_body      - 请求体内容

自定义变量：
  # 用 set 或 map 定义
  set $my_var "default";

  # map 可以根据条件赋值
  map $http_user_agent $is_mobile {
      default         0;
      "~*iPhone"      1;
      "~*Android"     1;
  }

  # 使用变量
  if ($is_mobile) {
      rewrite ^ /mobile$request_uri;
  }

  # 在日志中使用
  log_format main '$remote_addr - $request_uri - $is_mobile';
```


> **Note:** Nginx 采用 Master/Worker 多进程 + 事件驱动架构，每个 Worker 通过 epoll 处理数千并发连接。配置文件分层嵌套（main→events/http→server→location）。location 匹配优先级：= 精确 > ^~ 前缀 > ~/~* 正则 > 通用前缀。合理使用内置变量可以实现灵活的请求路由和日志记录。


<!-- Converted from: 01_Nginx架构与配置.html -->
