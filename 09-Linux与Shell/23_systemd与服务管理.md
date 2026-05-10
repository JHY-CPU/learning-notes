# systemd与服务管理


## ⚙️ systemd 与服务管理


systemd 基础、Unit 文件、systemctl 命令、journalctl 日志、定时器。


## systemd 概述


```
// ========== systemd ==========
// systemd 是现代 Linux 的标准 init 系统 (PID 1)
// 替代了 SysV init 和 Upstart
// 几乎所有主流发行版都在使用

// ========== Unit 类型 ==========
// systemd 管理的资源称为 Unit
// .service    — 服务 (最常用)
// .socket     — 网络或 IPC 套接字
// .timer      — 定时任务 (替代 cron)
// .target     — 逻辑分组 (类似 runlevel)
// .mount      — 挂载点
// .path       — 文件或目录变化监控
// .device     — 内核设备
// .automount  — 自动挂载
// .slice      — 资源控制 (cgroups)
// .scope      — 外部创建进程

// ========== Unit 文件位置 ==========
// /etc/systemd/system/      # 系统管理员配置 (优先级最高)
// /run/systemd/system/      # 运行时配置
// /usr/lib/systemd/system/  # 软件包安装的配置

// 用户服务:
// ~/.config/systemd/user/   # 用户级服务 (无需 root)
```


## systemctl 命令


```
// ========== 服务生命周期 ==========
systemctl start nginx          # 启动
systemctl stop nginx           # 停止
systemctl restart nginx        # 重启
systemctl reload nginx         # 重载配置 (不中断)
systemctl reload-or-restart nginx  # 优先 reload
systemctl enable nginx         # 开机自启
systemctl disable nginx        # 取消开机自启
systemctl enable --now nginx   # 启用并立即启动
systemctl mask nginx           # 禁止启动 (link to /dev/null)
systemctl unmask nginx         # 取消禁止

// ========== 状态查看 ==========
systemctl status nginx         # 详细状态 (含最近日志)
systemctl status --no-pager nginx  # 不分页
systemctl is-active nginx      # active / inactive
systemctl is-enabled nginx     # enabled / disabled
systemctl is-failed nginx      # failed / active

// systemctl status 输出:
// → nginx.service - A high performance web server
//    Loaded: loaded (/usr/lib/systemd/system/nginx.service; enabled)
//    Active: active (running) since Mon 2024-01-15 10:00:00 CST
//    Process: 1234 ExecStart=/usr/sbin/nginx (code=exited, status=0/SUCCESS)
//    Main PID: 1235 (nginx)
//    Tasks: 3 (limit: 4915)
//    Memory: 12.3M
//    CGroup: /system.slice/nginx.service
//            ├─1235 nginx: master process /usr/sbin/nginx
//            └─1236 nginx: worker process

// ========== 管理命令 ==========
systemctl list-units                     # 所有活跃 unit
systemctl list-units --all               # 所有 unit
systemctl list-units --type=service      # 只显示服务
systemctl list-units --state=running     # 运行中的
systemctl list-unit-files                # 所有 unit 文件及状态
systemctl daemon-reload                  # 重新加载 unit 文件
systemctl reset-failed                   # 重置 failed 状态
systemctl show nginx                     # 查看所有属性

// ========== 系统命令 ==========
systemctl reboot              # 重启
systemctl poweroff            # 关机
systemctl suspend             # 暂停 (睡眠)
systemctl hibernate           # 休眠
systemctl hybrid-sleep        # 混合睡眠
systemctl emergency           # 紧急模式
systemctl default             # 回到默认 target
```


## Service Unit 文件


```
// ========== 自定义 service 示例 ==========
// 文件: /etc/systemd/system/myapp.service

[Unit]
Description=My Node.js Application
Documentation=https://example.com/docs
After=network.target postgresql.service
Wants=postgresql.service
Requires=network.target

[Service]
Type=simple
User=alice
Group=alice
WorkingDirectory=/opt/myapp

# 启动命令
ExecStart=/usr/bin/node /opt/myapp/server.js
ExecStartPre=/bin/mkdir -p /var/log/myapp
ExecReload=/bin/kill -HUP $MAINPID
ExecStop=/bin/kill -15 $MAINPID
ExecStopPost=/bin/rm -f /var/run/myapp.pid

# 重启策略
Restart=on-failure
RestartSec=5
StartLimitIntervalSec=60
StartLimitBurst=3

# 环境变量
Environment=NODE_ENV=production
EnvironmentFile=/etc/myapp/env.conf

# 资源限制
LimitNOFILE=65536
LimitNPROC=4096
MemoryMax=500M
CPUQuota=50%

# 安全
ProtectSystem=strict
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target

// ========== Type 类型 ==========
// simple:    ExecStart 启动后即为 up (默认)
// exec:      ExecStart 进程执行后即为 up (systemd v240+)
// forking:   启动后 fork,父进程退出 (传统 daemon)
// oneshot:   执行一次就结束,配合 RemainAfterExit=yes
// dbus:      等待 D-Bus 名称注册
// notify:    进程发送 sd_notify() 通知就绪
// idle:      等待其他任务完成再启动

// ========== Restart 策略 ==========
// no:            不自动重启 (默认)
// always:        总是重启
// on-success:    退出码 0 时重启
// on-failure:    非 0 退出码时重启
// on-abnormal:   被信号终止时重启
// on-watchdog:   看门狗超时重启
// on-abort:      被非 SIGHUP 信号终止时重启

// ========== 创建服务步骤 ==========
// 1. 创建 /etc/systemd/system/myapp.service
// 2. sudo systemctl daemon-reload
// 3. sudo systemctl enable --now myapp
// 4. sudo systemctl status myapp
```


## journalctl — 日志查看


```
// ========== journalctl 基础 ==========
// systemd 统一日志收集器 (二进制)

journalctl                     # 查看所有日志
journalctl -u nginx            # 查看特定服务日志
journalctl -u nginx -u sshd   # 多个服务
journalctl -k                  # 内核日志
journalctl -f                  # 实时跟踪 (类似 tail -f)
journalctl -n 50               # 最后 50 行
journalctl -r                  # 反向 (最新在前)

// ========== 按时间过滤 ==========
journalctl --since "1 hour ago"
journalctl --since "2024-01-15 10:00:00" --until "2024-01-15 12:00:00"
journalctl --since yesterday
journalctl --since "2024-01-01" --until "2024-01-31"

// ========== 按优先级过滤 ==========
journalctl -p err               # 错误及以上
journalctl -p warning           # 警告及以上
// 优先级: emerg > alert > crit > err > warning > notice > info > debug

// ========== 输出控制 ==========
journalctl -o verbose          # 详细格式 (含所有字段)
journalctl -o json             # JSON 格式
journalctl -o json-pretty      # 格式化 JSON
journalctl --no-pager          # 不分页

// ========== 维护 ==========
journalctl --disk-usage        # 查看日志占用
journalctl --vacuum-size=500M  # 限制日志大小
journalctl --vacuum-time=7d    # 保留 7 天
journalctl --rotate            # 主动轮转

// ========== systemd-timer — 定时任务 ==========
// 替代 cron 的 systemd 方式
// 需要两个文件: .timer + .service

// /etc/systemd/system/backup.timer
[Unit]
Description=Daily backup timer

[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=1800

[Install]
WantedBy=timers.target

// /etc/systemd/system/backup.service
[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup.sh

// 启用:
systemctl enable --now backup.timer
systemctl list-timers          # 查看所有定时器
```


> **Note:** 💡 systemd 是现代 Linux 服务管理的核心。编写 service unit 时要特别注意 Type 选择、Restart 策略、EnvironmentFile 分离配置。journalctl -u
> -f --since "10 min ago" 是排查服务问题的标准命令。


## 练习


<!-- Converted from: 23_systemd与服务管理.html -->
