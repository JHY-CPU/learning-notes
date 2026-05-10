# OpenResty 与 Lua 扩展


## OpenResty 与 Lua 扩展


OpenRestyLuangx_lua


OpenResty 是基于 Nginx 和 LuaJIT 的高性能 Web 平台，通过 ngx_lua 模块在 Nginx 请求处理的各个阶段嵌入 Lua 脚本，实现强大的可编程网关能力。


## OpenResty 架构


```
OpenResty 组成：
  ┌─────────────────────────────────────────────┐
  │  OpenResty                                  │
  │  ├── Nginx 核心                             │
  │  ├── ngx_lua 模块（Lua 嵌入 Nginx）         │
  │  ├── LuaJIT（高性能 Lua 运行时）            │
  │  ├── lua-resty-* 库集合                     │
  │  │   ├── lua-resty-core                     │
  │  │   ├── lua-resty-lrucache                 │
  │  │   ├── lua-resty-redis                    │
  │  │   ├── lua-resty-mysql                    │
  │  │   ├── lua-resty-http                     │
  │  │   ├── lua-resty-string（加密）           │
  │  │   └── lua-resty-websocket                │
  │  └── 其他第三方 Nginx 模块                  │
  └─────────────────────────────────────────────┘

安装 OpenResty：
  # Ubuntu/Debian
  apt-get -y install software-properties-common
  add-apt-repository -y ppa:openresty/ppa
  apt-get update
  apt-get install openresty

  # CentOS
  yum install yum-utils
  yum-config-manager --add-repo https://openresty.org/package/centos/openresty.repo
  yum install openresty

  # 目录结构
  /usr/local/openresty/
  ├── nginx/              # Nginx 主程序
  ├── luajit/             # LuaJIT 运行时
  ├── lualib/             # Lua 库
  │   ├── resty/          # lua-resty-* 库
  │   └── ngx/            # ngx.* API
  └── bin/
      └── resty           # 命令行工具

基本配置：
  http {
      # Lua 包路径
      lua_package_path "/usr/local/openresty/lualib/?.lua;;";
      lua_package_cpath "/usr/local/openresty/lualib/?.so;;";

      # 共享内存字典
      lua_shared_dict my_cache 10m;

      server {
          listen 80;
          server_name localhost;

          # 内联 Lua 代码
          location /hello {
              content_by_lua_block {
                  ngx.say("Hello, OpenResty!")
              }
          }

          # 外部 Lua 文件
          location /api {
              content_by_lua_file /etc/openresty/lua/api.lua;
          }
      }
  }
```


## Nginx 请求处理阶段与 Lua 指令


```
Nginx 请求处理阶段：
  ┌─────────────────────────────────────────────┐
  │  1. POST_READ          读取请求后           │
  │  2. SERVER_REWRITE     server 块重写        │
  │  3. FIND_CONFIG        查找配置             │
  │  4. REWRITE            location 块重写      │
  │  5. POST_REWRITE       重写后               │
  │  6. PREACCESS          访问前（限流）       │
  │  7. ACCESS             访问控制             │
  │  8. POST_ACCESS        访问后               │
  │  9. PRECONTENT         内容前               │
  │  10. CONTENT           内容生成             │
  │  11. LOG               日志记录             │
  └─────────────────────────────────────────────┘

对应 Lua 指令：
  init_by_lua_file          # Master 进程启动时
  init_worker_by_lua_file   # Worker 进程启动时

  set_by_lua_file           # 设置变量时
  rewrite_by_lua_file       # REWRITE 阶段
  access_by_lua_file        # ACCESS 阶段
  content_by_lua_file       # CONTENT 阶段
  header_filter_by_lua_file # 响应头过滤
  body_filter_by_lua_file   # 响应体过滤
  log_by_lua_file           # LOG 阶段
  balancer_by_lua_file      # 负载均衡阶段

阶段使用示例：
  # init_worker：定时任务、预热缓存
  init_worker_by_lua_file /etc/openresty/lua/init_worker.lua;

  # rewrite：URL 重写、请求改写
  rewrite_by_lua_file /etc/openresty/lua/rewrite.lua;

  # access：认证、限流、IP 黑名单
  access_by_lua_file /etc/openresty/lua/access.lua;

  # content：业务逻辑处理
  content_by_lua_file /etc/openresty/lua/content.lua;

  # header_filter：添加/修改响应头
  header_filter_by_lua_file /etc/openresty/lua/header_filter.lua;

  # body_filter：修改响应体
  body_filter_by_lua_file /etc/openresty/lua/body_filter.lua;

  # log：异步日志记录
  log_by_lua_file /etc/openresty/lua/log.lua;
```


## Lua 脚本实战示例


```
access.lua — JWT 认证：
  local jwt = require "resty.jwt"
  local auth_header = ngx.var.http_authorization

  if not auth_header then
      ngx.status = 401
      ngx.say('{"error": "Missing token"}')
      return ngx.exit(401)
  end

  local token = string.match(auth_header, "Bearer%s+(.+)")
  local jwt_obj = jwt:verify("your-secret-key", token)

  if not jwt_obj.verified then
      ngx.status = 401
      ngx.say('{"error": "Invalid token"}')
      return ngx.exit(401)
  end

  -- 将用户信息传递给后端
  ngx.req.set_header("X-User-ID", jwt_obj.payload.sub)
  ngx.req.set_header("X-User-Role", jwt_obj.payload.role)

access.lua — IP 黑名单：
  local ip_blacklist = {
      ["1.2.3.4"] = true,
      ["5.6.7.8"] = true,
  }

  local client_ip = ngx.var.remote_addr
  if ip_blacklist[client_ip] then
      ngx.status = 403
      ngx.say('{"error": "Forbidden"}')
      return ngx.exit(403)
  end

content.lua — API 网关：
  local http = require "resty.http"
  local cjson = require "cjson"

  local httpc = http.new()
  local res, err = httpc:request_uri("http://backend:8080/api/users", {
      method = "GET",
      headers = {
          ["Content-Type"] = "application/json",
      }
  })

  if not res then
      ngx.status = 502
      ngx.say(cjson.encode({error = err}))
      return
  end

  ngx.status = res.status
  ngx.header["Content-Type"] = "application/json"
  ngx.say(res.body)

balancer.lua — 动态负载均衡：
  local balancer = require "ngx.balancer"
  local upstreams = {
      "192.168.1.10:8080",
      "192.168.1.11:8080",
      "192.168.1.12:8080",
  }

  local idx = math.random(#upstreams)
  local host, port = string.match(upstreams[idx], "(.+):(%d+)")

  local ok, err = balancer.set_current_peer(host, tonumber(port))
  if not ok then
      ngx.status = 502
      ngx.say('{"error": "Balancer failed"}')
  end

log.lua — 异步日志：
  local cjson = require "cjson"
  local log_data = {
      timestamp = ngx.now(),
      client_ip = ngx.var.remote_addr,
      method = ngx.var.request_method,
      uri = ngx.var.uri,
      status = ngx.status,
      response_time = ngx.now() - ngx.req.start_time(),
  }

  -- 异步发送到日志系统
  local ok, err = ngx.timer.at(0, function(premature)
      if premature then return end
      -- 发送到 Kafka/Redis/文件
      local file = io.open("/var/log/openresty/access.json", "a")
      if file then
          file:write(cjson.encode(log_data) .. "\n")
          file:close()
      end
  end)
```


## Shared Dict 与动态 Upstream


```
lua_shared_dict 共享内存：
  # nginx.conf
  http {
      lua_shared_dict my_cache 100m;    # 100MB 共享内存
      lua_shared_dict rate_limit 10m;   # 限流计数器
      lua_shared_dict config 1m;        # 配置存储
  }

共享内存操作：
  local cache = ngx.shared.my_cache

  -- 设置值
  cache:set("key", "value", exptime)   -- 带过期时间
  cache:set("key", "value")            -- 永不过期
  cache:add("key", "value", exptime)   -- 不存在才设置

  -- 获取值
  local value, flags = cache:get("key")

  -- 自增（原子操作）
  cache:incr("counter", 1, 0)  -- 不存在则初始化为0

  -- 删除
  cache:delete("key")

  -- 获取所有键
  local keys = cache:get_keys()

  -- 容量信息
  local capacity = cache:capacity()   -- 总容量
  local free = cache:free_space()     -- 剩余空间

限流实现（基于 shared_dict）：
  local limit = require "resty.limit.req"
  local lim, err = limit.new("rate_limit", 10, 10)  -- 10r/s, burst=10
  if not lim then
      ngx.log(ngx.ERR, "failed to create limiter: ", err)
      return ngx.exit(500)
  end

  local key = ngx.var.binary_remote_addr
  local delay, err = lim:incoming(key, true)

  if not delay then
      if err == "rejected" then
          return ngx.exit(429)
      end
      return ngx.exit(500)
  end

  if delay > 0 then
      ngx.sleep(delay)
  end

动态 Upstream（balancer_by_lua）：
  http {
      # 动态 upstream
      upstream backend {
          server 0.0.0.1;  # 占位服务器
          balancer_by_lua_file /etc/openresty/lua/balancer.lua;
          keepalive 32;
      }

      server {
          location /api {
              proxy_pass http://backend;
          }
      }
  }

  -- balancer.lua
  local balancer = require "ngx.balancer"
  local cjson = require "cjson"

  -- 从 Redis/Consul 获取后端列表
  local redis = require "resty.redis"
  local red = redis:new()
  red:connect("127.0.0.1", 6379)

  local backends = red:get("upstream:backend")
  if backends then
      local servers = cjson.decode(backends)
      local idx = math.random(#servers)
      local host, port = string.match(servers[idx], "(.+):(%d+)")
      balancer.set_current_peer(host, tonumber(port))
  end

  -- 连接池
  red:set_keepalive(10000, 100)

场景应用：
  ┌──────────────────┬────────────────────────────┐
  │ 场景             │ OpenResty 实现             │
  ├──────────────────┼────────────────────────────┤
  │ API 网关         │ access + content + balancer│
  │ 限流             │ shared_dict + limit.req    │
  │ 缓存             │ shared_dict + redis        │
  │ 灰度发布         │ rewrite + balancer         │
  │ A/B 测试         │ rewrite + split_clients    │
  │ 动态配置         │ init_worker + redis订阅    │
  │ 实时日志         │ log + kafka/redis          │
  │ 金丝雀发布       │ balancer + shared_dict     │
  └──────────────────┴────────────────────────────┘
```


> **Note:** OpenResty 通过 ngx_lua 将 Lua 脚本嵌入 Nginx 各处理阶段，实现高性能可编程网关。核心能力包括：shared_dict 共享内存（缓存/限流计数器）、lua-resty-* 库（Redis/MySQL/HTTP客户端）、动态 Upstream（balancer_by_lua）。典型应用：API 网关认证、动态限流、灰度发布、实时日志收集。LuaJIT 的 JIT 编译确保了接近原生的执行性能。


<!-- Converted from: 03_OpenResty与Lua扩展.html -->
