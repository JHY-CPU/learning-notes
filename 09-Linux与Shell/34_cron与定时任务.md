# cron与定时任务


## ⏰ cron 与定时任务


cron/crontab 语法、at 一次性任务、systemd timer、日志与调试、最佳实践。


## cron 基础


```
// ========== cron 概述 ==========
// cron = 定时任务执行器 (Linux 标配)
// crond 守护进程常驻后台,每分钟检查任务
// crontab 是编辑任务列表的工具

// ========== crontab 命令 ==========
crontab -l                 # 列出当前用户的定时任务
crontab -e                 # 编辑当前用户的定时任务
crontab -r                 # 删除所有定时任务
crontab -u alice -l        # 查看 alice 的任务 (root)
crontab -u alice -e        # 编辑 alice 的任务 (root)

// ========== crontab 格式 ==========
// ┌──────── 分钟 (0-59)
// │ ┌────── 小时 (0-23)
// │ │ ┌──── 日 (1-31)
// │ │ │ ┌── 月 (1-12)
// │ │ │ │ ┌ 星期 (0-7, 0/7=周日)
// │ │ │ │ │
// * * * * *

// 示例:
# 每分钟执行
* * * * * /usr/bin/check_health.sh

# 每天凌晨 2:30 执行
30 2 * * * /usr/bin/backup.sh

# 每周一 9:00 执行
0 9 * * 1 /usr/bin/weekly_report.sh

# 每月 1 号 0:00 执行
0 0 1 * * /usr/bin/monthly_cleanup.sh

# 每 15 分钟
*/15 * * * * /usr/bin/poll.sh

# 工作日 9-17 点每 30 分钟
30 9-17 * * 1-5 /usr/bin/check.sh

// ========== 特殊字符串 ==========
@reboot     # 重启后执行一次
@yearly     # 每年 (0 0 1 1 *)
@monthly    # 每月 (0 0 1 * *)
@weekly     # 每周 (0 0 * * 0)
@daily      # 每天 (0 0 * * *)
@hourly     # 每小时 (0 * * * *)

@reboot /usr/local/bin/start_myapp.sh
@daily /usr/local/bin/daily_cleanup.sh
```


## cron 实战


```
// ========== 常用定时任务 ==========
# 系统维护
0 3 * * * /usr/bin/apt update && /usr/bin/apt upgrade -y      # 每天 3 点更新软件
0 4 * * 0 /usr/sbin/reboot                                    # 每周日凌晨 4 点重启
*/5 * * * * /usr/local/bin/check_disk.sh                      # 每 5 分钟检查磁盘

# 备份
30 2 * * * /usr/local/bin/backup_db.sh                        # 每天 2:30 备份数据库
0 0 * * 0 /usr/local/bin/full_backup.sh                       # 每周日全量备份

# 日志清理
0 0 * * * find /var/log/myapp -mtime +30 -delete              # 删除 30 天前的日志

# 监控
*/5 * * * * /usr/local/bin/health_check.sh                    # 每 5 分钟健康检查
0 9 * * * /usr/local/bin/daily_report.sh | mail -s "Daily Report" admin@example.com  # 日报

// ========== 备份脚本示例 ==========
#!/bin/bash
# /usr/local/bin/backup_db.sh
set -euo pipefail

BACKUP_DIR="/backup/db"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="myapp"

mkdir -p "$BACKUP_DIR"
pg_dump "$DB_NAME" | gzip > "$BACKUP_DIR/${DB_NAME}_${TIMESTAMP}.sql.gz"

# 保留 7 天
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +7 -delete

echo "Backup completed: ${DB_NAME}_${TIMESTAMP}.sql.gz"

// ========== 环境变量注意 ==========
// cron 默认使用受限环境 (PATH=/usr/bin:/bin)
// 需要全路径或设置 PATH

# 错误的写法:
# python /opt/script.py    # 可能找不到 python

# 正确写法:
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
0 2 * * * /usr/bin/python3 /opt/script.py

// ========== 输出处理 ==========
# 将输出和错误都记录到文件
30 2 * * * /usr/local/bin/backup.sh >> /var/log/backup.log 2>&1

# 静默运行 (不产生邮件)
30 2 * * * /usr/local/bin/backup.sh > /dev/null 2>&1

# 只在出错时发邮件
MAILTO=admin@example.com
30 2 * * * /usr/local/bin/backup.sh
```


## at — 一次性定时任务


```
// ========== at 命令 ==========
// at = 在指定时间执行一次命令
// 适合"5 分钟后执行"的场景

// ========== 基本用法 ==========
at now + 5 minutes            # 5 分钟后执行
at now + 1 hour               # 1 小时后
at now + 2 days               # 2 天后
at 14:30                      # 今天 14:30
at 09:00 tomorrow             # 明天 9:00
at 10:00 next week            # 下周 10:00
at 23:59 2024-12-31           # 指定日期

// 输入命令后按 Ctrl+D 提交:
$ at 14:30
warning: commands will be executed using /bin/sh
at> /usr/local/bin/deploy.sh
at>
job 5 at Wed Jan 15 14:30:00 2024

// ========== 管理 at 任务 ==========
atq                           # 查看排队任务
atrm 5                        # 删除任务 5
at -c 5                       # 查看任务 5 的具体命令

// ========== 与 cron 对比 ==========
// cron:    重复执行,永久有效
// at:      一次性执行,用完即删

// atd 服务:
systemctl status atd          # 确保 atd 在运行
systemctl enable --now atd
```


## 日志与调试


```
// ========== cron 日志 ==========
// 查看 cron 执行记录:
grep CRON /var/log/syslog           # Debian/Ubuntu
grep cron /var/log/cron             # CentOS/RHEL

// 实时查看:
tail -f /var/log/syslog | grep CRON

// 查看特定任务:
grep "backup.sh" /var/log/syslog

// ========== 调试技巧 ==========
// 1. 测试命令是否能手动运行
/usr/local/bin/backup.sh

// 2. 模拟 cron 环境运行
env -i HOME=$HOME PATH=/usr/bin:/bin SHELL=/bin/sh /bin/sh -c "/usr/local/bin/backup.sh"

// 3. 检查 crontab 语法
crontab -l | grep -v "^#" | while read line; do
    echo "$line" | grep -E "^[^ ]+ [^ ]+ [^ ]+ [^ ]+ [^ ]+ " > /dev/null || echo "语法错误: $line"
done

// 4. 检查 cron 服务状态
systemctl status cron
ps aux | grep crond

// ========== 常见问题 ==========
// 1. 命令找不到 → 使用全路径
// 2. 脚本没执行权限 → chmod +x script.sh
// 3. 脚本中用了 ~ → HOME 不一定设置,用全路径
// 4. 时区不对 → timedatectl set-timezone Asia/Shanghai
// 5. 输出没看到 → 检查 MAILTO 和日志
```


> **Note:** 💡 cron 的最佳实践: (1) 脚本中设置 set -euo pipefail; (2) 使用全路径,不依赖 PATH; (3) 输出重定向到日志文件; (4) 脚本先手动测试再放到 cron; (5) 添加随机延迟避免多任务同时启动 (sleep $((RANDOM % 60))); (6) 复杂的任务可以换成 systemd timer。


## crontab 速查与模板


```
// ========== 常用周期速查 ==========
// 每分钟:        * * * * *
// 每5分钟:       */5 * * * *
// 每15分钟:      */15 * * * *
// 每小时:        0 * * * *
// 每天0点:       0 0 * * *
// 每天9:30:      30 9 * * *
// 每周一0点:     0 0 * * 1
// 每月1号0点:    0 0 1 * *
// 工作日9-18点:  0 9-18 * * 1-5
// 每季首月1号:   0 0 1 1,4,7,10 *

// ========== 安全控制 ==========
// 允许/禁止用户使用 cron:
// /etc/cron.allow    # 白名单 (一行一个用户名)
// /etc/cron.deny     # 黑名单

// ========== 系统级 cron 目录 ==========
// /etc/crontab               # 系统级 crontab (支持指定用户)
// /etc/cron.d/               # 分文件配置
// /etc/cron.hourly/          # 每小时执行
// /etc/cron.daily/           # 每天执行
// /etc/cron.weekly/          # 每周执行
// /etc/cron.monthly/         # 每月执行

// 系统级 crontab 格式 (多了用户字段):
// 0 2 * * * root /usr/local/bin/backup.sh

// ========== 完整 crontab 模板 ==========
# /etc/crontab 或 crontab -e

# 环境设置
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
MAILTO=admin@example.com
HOME=/

# 系统维护
0 3 * * * root /usr/bin/apt update && /usr/bin/apt upgrade -y
0 4 * * 0 root /sbin/reboot

# 应用维护
*/5 * * * * appuser /opt/myapp/scripts/health_check.sh
30 2 * * * appuser /opt/myapp/scripts/backup.sh
0 0 * * 0 appuser /opt/myapp/scripts/cleanup.sh

# 日志轮转
0 0 * * * root /usr/sbin/logrotate /etc/logrotate.conf
```


## 练习


<!-- Converted from: 34_cron与定时任务.html -->
