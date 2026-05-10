# Docker 入门与核心概念


## 🐳 Docker 入门与核心概念


Docker 是什么、镜像 vs 容器 vs 仓库、架构 (daemon/client/registry)、基本命令、交互式 vs 后台运行。


## Docker 核心概念


```
// ========== 什么是 Docker ==========
// Docker 是一个容器化平台, 将应用及其依赖打包到
// 轻量级、可移植的容器中运行

// ========== 三大核心概念 ==========
// 1. 镜像 (Image)
//    - 只读模板, 包含运行应用所需的代码/运行时/库/环境变量
//    - 由多层文件系统组成 (Layer)
//    - 不可变, 运行时创建可写层

// 2. 容器 (Container)
//    - 镜像的运行实例
//    - 轻量级 (共享宿主机内核)
//    - 有自己的文件系统/网络/进程空间
//    - 启动/停止/删除/暂停

// 3. 仓库 (Registry)
//    - 存储和分发镜像
//    - Docker Hub (官方), GitHub Container Registry, 私有仓库
//    - docker pull / docker push

// ========== Docker 架构 ==========
// ┌──────────┐     ┌──────────────┐     ┌──────────┐
// │  Client  │────▶│    Docker    │────▶│ Registry │
// │ (docker) │     │    Daemon    │     │ (Hub)    │
// └──────────┘     │  (dockerd)   │     └──────────┘
//                  │              │
//                  │ ┌──┐ ┌──┐   │
//                  │ │C1│ │C2│   │  (Containers)
//                  │ └──┘ └──┘   │
//                  │ ┌──┐ ┌──┐   │
//                  │ │I1│ │I2│   │  (Images)
//                  │ └──┘ └──┘   │
//                  └──────────────┘

// Client: 命令行工具 docker
// Daemon: dockerd, 管理容器/镜像/网络/卷
// Registry: 镜像仓库 (默认 Docker Hub)

// ========== 镜像 vs 容器 ==========
// 镜像 = 类 (Class)
// 容器 = 实例 (Instance)

// 一个镜像可以启动多个容器
// 容器可以启动/停止/删除
// 镜像一旦构建不可变
```


## 基本命令


```
// ========== 镜像管理 ==========
// docker pull nginx:alpine        // 拉取镜像
// docker images                   // 查看本地镜像
// docker image ls                 // 同上
// docker rmi nginx                // 删除镜像
// docker image prune              // 删除悬空镜像
// docker build -t myapp .         // 构建镜像

// ========== 容器生命周期 ==========
// docker run nginx                // 创建并启动容器
// docker run -d nginx             // 后台运行
// docker run --name my-nginx nginx // 指定名称
// docker run -p 8080:80 nginx     // 端口映射

// docker ps                       // 运行中的容器
// docker ps -a                    // 所有容器 (含停止)
// docker stop          // 停止
// docker start         // 启动已停止的
// docker restart       // 重启
// docker rm            // 删除容器
// docker rm -f         // 强制删除 (运行中)

// ========== 交互式运行 ==========
// docker run -it ubuntu bash       // 交互式终端
// docker run -it --rm ubuntu bash  // 退出时自动删除
// docker exec -it  bash // 在运行容器中执行命令

// 常用选项:
//   -i: 交互式 (保持 STDIN 打开)
//   -t: 分配伪终端 (TTY)
//   -d: 后台运行 (detach)
//   --rm: 退出时自动删除容器
//   --name: 指定容器名
//   -p host:container: 端口映射
//   -e KEY=VALUE: 环境变量

// ========== 日志与信息 ==========
// docker logs           // 查看日志
// docker logs -f        // 实时跟踪日志
// docker logs --tail 100  // 最后 100 行

// docker inspect        // 查看详细信息
// docker stats                     // 实时资源占用
// docker top            // 查看进程

// ========== 清理 ==========
// docker container prune          // 删除所有已停止的
// docker image prune              // 删除悬空镜像
// docker system prune             // 清理所有未使用
// docker system prune -a          // 清理所有 (含未使用镜像)
// docker system df                // 查看磁盘使用
```


## 端口映射与数据卷


```
// ========== 端口映射 ==========
// -p :

// docker run -p 8080:80 nginx
// 访问 http://localhost:8080 → 容器内 80 端口

// docker run -p 3000:3000 -p 9229:9229 node-app
// 多个端口映射

// docker run -p 80:80/tcp -p 53:53/udp dns-server
// 指定协议

// 不指定 host 端口 (随机分配):
// docker run -P nginx
// docker port   // 查看映射

// ========== 数据卷 ==========
// Volume: Docker 管理的数据持久化

// 创建卷:
// docker volume create mydata
// docker volume ls
// docker volume inspect mydata

// 使用卷:
// docker run -v mydata:/data nginx
// docker run --mount source=mydata,target=/data nginx

// ========== 绑定挂载 ==========
// 将宿主机目录挂载到容器

// docker run -v /host/path:/container/path:ro nginx
// docker run --mount type=bind,source=/host,target=/container nginx

// :ro = 只读
// 适合开发 (代码热重载)

// ========== tmpfs 挂载 ==========
// 仅存内存, 容器停止后删除
// docker run --tmpfs /tmp nginx
// docker run --mount type=tmpfs,target=/tmp nginx

// ========== 卷类型对比 ==========
// Volume        : Docker 管理 /var/lib/docker/volumes/
// Bind Mount    : 宿主机任意目录
// tmpfs         : 仅内存

// Volume 推荐用于生产 (可备份、可迁移)
// Bind Mount 推荐用于开发
```


## 网络基础


```
// ========== Docker 网络模式 ==========
// docker network ls

// 五种网络模式:
// 1. bridge (默认) — 内部 NAT, 可互相通信
// 2. host — 共享宿主机网络 (无隔离)
// 3. none — 无网络
// 4. overlay — 跨宿主机网络 (Swarm)
// 5. macvlan — 分配 MAC 地址

// ========== Bridge 网络 ==========
// 默认网络, 容器通过 IP 通信

// docker network create mynet       // 自定义 bridge
// docker run --network mynet nginx
// docker network connect mynet container
// docker network disconnect mynet container

// 自定义 bridge 支持 DNS 解析
// (容器名可直接作为主机名)

// docker run --network mynet --name app1 nginx
// docker run --network mynet --name app2 busybox
// app2 中可以直接 ping app1

// ========== Host 网络 ==========
// 容器直接使用宿主机网络
// 无端口映射, 直接访问容器端口

// docker run --network host nginx
// 访问 http://localhost:80 直接到容器

// ========== 网络命令 ==========
// docker network ls                 // 列出网络
// docker network create mynet       // 创建网络
// docker network inspect mynet      // 查看详情
// docker network rm mynet           // 删除
// docker network prune              // 清理未使用的

// 端口映射:
// docker port            // 查看端口映射
```


> **Note:** 💡 Docker 入门要点: 镜像 (只读模板) → 容器 (运行实例) → 仓库 (存储分发); Client-Daemon-Registry 架构; docker run/create/start/stop/rm; -d 后台, -it 交互, -p 端口映射, -v 卷挂载, -e 环境变量, --name 命名; docker ps/logs/inspect/stats; 网络 bridge/host/none; 自定义 bridge 支持 DNS; Volume 持久化数据, Bind Mount 开发热重载。


## 练习


<!-- Converted from: 0_Docker 入门与核心概念.html -->
