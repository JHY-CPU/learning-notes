# sudo与权限提升


## 🔐 sudo 与权限提升


sudo 原理、/etc/sudoers 配置、visudo、sudo 最佳实践、安全注意事项。


## sudo 基础


```
// ========== sudo 概述 ==========
// sudo = superuser do
// 以其他用户身份执行命令 (默认以 root)
// 无需知道 root 密码
// 权限由 /etc/sudoers 控制

// ========== 基本用法 ==========
sudo command              # 以 root 执行命令
sudo -u alice command     # 以 alice 执行
sudo -i                   # 交互式 root Shell (登录)
sudo -s                   # root Shell (非登录)
sudo -l                   # 查看自己的 sudo 权限
sudo -v                   # 刷新密码缓存 (不执行命令)
sudo -k                   # 清除密码缓存
sudo -E command           # 保留当前环境变量
sudo -u alice -H command  # 以 alice 执行,设置 HOME

// ========== 执行流程 ==========
// 1. 用户执行 sudo command
// 2. sudo 检查 /etc/sudoers 是否授权
// 3. 如果需密码,提示用户输入自己的密码
// 4. 验证通过后,以目标用户执行命令
// 5. 日志记录到 /var/log/auth.log 或 /var/log/secure

// ========== sudo 日志 ==========
// Debian/Ubuntu: /var/log/auth.log
// Red Hat/CentOS: /var/log/secure
// 查看失败的 sudo 尝试:
grep "sudo" /var/log/auth.log | grep "FAILED"

// ========== sudo 常见问题 ==========
// "user is not in the sudoers file"
//   → 需要 root 将用户加入 sudo 组或编辑 sudoers

// "sudo: command not found"
//   → 命令在 /sbin 下, PATH 未包含
//   → 用 sudo /usr/sbin/command 全路径

// "sorry, you must have a tty to run sudo"
//   → 在脚本或 cron 中需要设置 requiretty
```


## /etc/sudoers 配置


```
// ========== sudoers 语法 ==========
// user  host=(runas)  TAG: command
//
// 字段:
//   user:   用户名或 %组名
//   host:   主机名匹配
//   runas:  可以切换到的用户
//   TAG:    NOPASSWD, PASSWD, NOEXEC, SETENV 等标签
//   command: 允许的命令路径

// ========== 常用配置示例 ==========
// 允许 alice 执行任何命令 (需密码)
alice ALL=(ALL:ALL) ALL

// 允许 sudo 组成员执行任何命令
%sudo ALL=(ALL:ALL) ALL

// 允许 wheel 组成员无需密码运行任何命令
%wheel ALL=(ALL:ALL) NOPASSWD:ALL

// 允许 alice 无需密码重启服务
alice ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart *, /usr/bin/systemctl start *, /usr/bin/systemctl stop *

// 允许 webadmin 组仅管理 Web 服务
%webadmin ALL=(ALL) /usr/bin/systemctl * nginx, /usr/bin/systemctl * php*

// 允许 backup 用户只运行备份相关命令
backup ALL=(ALL) NOPASSWD: /usr/bin/rsync, /usr/bin/tar, /usr/local/bin/backup.sh

// 允许 deploy 用户以 webapp 身份部署
deploy ALL=(webapp) NOPASSWD: /usr/bin/git pull, /usr/local/bin/deploy.sh

// ========== 别名 ==========
// 用户别名
User_Alias ADMINS = alice, bob, charlie
User_Alias DEPLOY = deploy-bot, ci-server

// 命令别名
Cmnd_Alias SERVICES = /usr/bin/systemctl restart *, /usr/bin/systemctl start *, /usr/bin/systemctl stop *
Cmnd_Alias NETWORKING = /sbin/ip, /sbin/ifconfig, /usr/bin/nmcli
Cmnd_Alias PACKAGES = /usr/bin/apt update, /usr/bin/apt install *, /usr/bin/apt remove *
Cmnd_Alias SHUTDOWN = /usr/sbin/shutdown, /usr/sbin/reboot, /usr/sbin/halt

// 使用别名:
ADMINS ALL=(ALL) NOPASSWD: SERVICES
DEPLOY ALL=(webapp) NOPASSWD: /usr/local/bin/deploy.sh

// ========== 默认配置 ==========
Defaults        env_reset      # 重置环境变量 (安全)
Defaults        mail_badpass   # 密码错误发邮件
Defaults        secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Defaults        timestamp_timeout=15  # 密码缓存 15 分钟
Defaults        logfile=/var/log/sudo.log  # 日志文件

// 安全相关:
Defaults        passwd_tries=3    # 最多尝试 3 次
Defaults        badpass_message="密码错误!"
Defaults        requiretty        # 需要终端 (cron 中可能有问题)
```


## visudo — 安全编辑


```
// ========== visudo ==========
// visudo 专门用于编辑 /etc/sudoers
// 提供语法检查,防止配置错误把自己锁在外面

visudo                    # 编辑主配置
visudo -f /etc/sudoers.d/webadmin  # 编辑分文件

// ========== 分文件配置 ==========
// 推荐在 /etc/sudoers.d/ 下创建分文件
// 避免直接修改 /etc/sudoers
// 文件名不能含 . 和 ~

cat > /etc/sudoers.d/webadmin << 'EOF'
%webadmin ALL=(ALL) /usr/bin/systemctl * nginx
EOF

// 设置正确权限:
chmod 440 /etc/sudoers.d/webadmin

// ========== 语法检查 ==========
visudo -c                 # 检查所有 sudoers 文件语法
// → /etc/sudoers: parsed OK

// ========== 安全提示 ==========
// 1. 始终用 visudo 编辑 (语法验证)
// 2. 分文件比直接改 sudoers 好
// 3. 使用 NOPASSWD 要谨慎
// 4. 指定命令全路径,避免通配符滥用
// 5. 注意命令覆盖 (如 cp 可覆盖系统文件)

// 危险配置示例:
// alice ALL=(ALL) NOPASSWD: ALL  # 完全 sudo 权限
// alice ALL=(ALL) /usr/bin/vi   # vi 中可 :!bash 逃逸!
// alice ALL=(ALL) /usr/bin/cp   # cp 可覆盖任意文件
```


> **Note:** 💡 用 visudo 分文件配置是最佳实践:将不同职责的 sudo 权限分离到 /etc/sudoers.d/ 下不同文件,便于管理和审计。危险的 sudo 授权:vi/tar/cp/nmap/gdb 等工具可以被用于权限提升,需要小心。


## sudo 最佳实践


```
// ========== 日常原则 ==========
// 1. 最小权限 — 只给需要的命令
// 2. 使用非 root 用户日常工作
// 3. sudo 不要用于日常命令
// 4. 审计 sudo 日志

// ========== 安全配置示例 ==========
# /etc/sudoers.d/90-cloud-init-users
# 为开发团队配置

# 开发 - 查看日志
%dev ALL=(ALL) /usr/bin/tail, /usr/bin/less, /usr/bin/grep

# 运维 - 管理服务
%ops ALL=(ALL) NOPASSWD: /usr/bin/systemctl
%ops ALL=(ALL) /usr/bin/journalctl

# 管理员 - 完全权限
%admin ALL=(ALL) ALL

# 部署机器人 - 特定命令,无需密码
deploy ALL=(webapp) NOPASSWD: /usr/local/bin/deploy.sh

// ========== sudo 替代方案 ==========
// doas — OpenBSD 的 sudo 替代 (更简洁)
//   安装: apt install doas
//   配置: /etc/doas.conf
//   permit alice cmd nginx

// polkit — 更细粒度的权限控制
//   pkexec command
```


## 练习


<!-- Converted from: 27_sudo与权限提升.html -->
