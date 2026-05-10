# netstat-ss-lsof


## 🔌 netstat/ss/lsof


网络连接查看、端口监听检查、socket 状态、进程与端口关联、文件打开情况。


## netstat — 网络统计


```
// ========== netstat 基础 ==========
// netstat 显示网络连接、路由表、接口统计
// 注意: 部分系统 netstat 已弃用,推荐 ss

// ========== 常用 netstat ==========
netstat -tlnp                    # TCP 监听端口 (+进程)
netstat -ulnp                    # UDP 监听端口
netstat -an                      # 所有连接
netstat -r                       # 路由表
netstat -i                       # 网络接口统计

// 选项解读:
// -t  TCP
// -u  UDP
// -l  仅监听 (listening)
// -n  数字显示 (不解析域名)
// -p  显示进程 PID/名称
// -a  所有 (包括 listen 和 established)
// -r  路由表
// -i  接口统计
// -s  协议统计 (每个协议的统计)

// netstat -tlnp 输出:
// Proto Recv-Q Send-Q Local Address    Foreign Address  State    PID/Program
// tcp    0      0     0.0.0.0:80       0.0.0.0:*        LISTEN   1234/nginx
// tcp    0      0     127.0.0.1:3306   0.0.0.0:*        LISTEN   2345/mysqld
// tcp    0      0     0.0.0.0:22       0.0.0.0:*        LISTEN   3456/sshd
// tcp    0      0     192.168.1.1:5432 0.0.0.0:*        LISTEN   4567/postgres

// Recv-Q/Send-Q: 接收/发送队列 (积压字节)
```


## ss — 现代 netstat


```
// ========== ss 优势 ==========
// ss = socket statistics
// 比 netstat 更快、信息更多
// netstat 已弃用,推荐使用 ss

// ========== 常用 ss ==========
ss -tlnp                        # TCP 监听端口
ss -ulnp                        # UDP 监听端口
ss -an                          # 所有连接
ss -s                           # socket 统计摘要
ss -t                           # TCP 连接
ss -u                           # UDP 连接
ss -o                           # 显示计时器信息
ss -e                           # 显示详细信息 (socket 选项等)

// ========== ss 状态过滤 ==========
ss -t state established         # 仅 ESTABLISHED 连接
ss -t state listening           # 仅 LISTEN 状态
ss -t state fin-wait-1          # FIN-WAIT-1
ss -t state all | more          # 所有 TCP 状态

// ========== ss 地址/端口过滤 ==========
ss -t src :80                   # 源端口 80
ss -t dst :443                  # 目标端口 443
ss -t src 192.168.1.1           # 源 IP
ss -t dst 10.0.0.0/8            # 目标网段
ss -t '( dport = :80 or sport = :80 )'  # 端口 80 相关

// ========== ss 连接统计 ==========
ss -s
// Total: 1234 (kernel 5678)
// TCP:   456 (estab 89, closed 300, orphaned 2, synrecv 0, timewait 45/0), ports 1234
// Transport Total     IP        IPv6
// *          5678     -         -
// RAW        0        0         0
// UDP        12       8         4
// TCP        456      432       24
// INET       468      440       28
// FRAG       0        0         0

// TCP 状态计数:
ss -t -a | awk '{print $1}' | sort | uniq -c
```


## lsof — 文件打开列表


```
// ========== lsof 概述 ==========
// lsof = list open files
// Unix 哲学:一切皆文件 (普通文件/目录/socket/pipe/设备)
// 可以查看进程打开了哪些文件

// ========== 网络相关 lsof ==========
lsof -i :80                       # 查看端口 80 的使用者
lsof -i TCP:22                    # TCP 端口 22
lsof -i UDP:53                    # UDP 端口 53
lsof -i @192.168.1.1              # 指定 IP 的连接
lsof -i @192.168.1.1:80          # IP+端口
lsof -i                           # 所有网络连接

// lsof -i :80 输出:
// COMMAND  PID  USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
// nginx   1234  root   6u   IPv4 12345      0t0  TCP *:80 (LISTEN)
// nginx   1235  nginx  8u   IPv4 12346      0t0  TCP *:80 (LISTEN)

// ========== 进程相关 lsof ==========
lsof -p 1234                      # PID 1234 打开的所有文件
lsof -c nginx                     # nginx 进程打开的文件
lsof -u alice                     # alice 打开的文件
lsof -u ^root                     # 非 root 用户

// ========== 文件相关 lsof ==========
lsof /var/log/syslog              # 谁在访问这个文件
lsof +D /var/log                  # 目录下所有打开的文件

// ========== 恢复删除的文件 ==========
// 如果已打开的文件被删除,可用 lsof 找回
lsof | grep "(deleted)"
// → 在 /proc//fd/ 中仍然可读

// ========== 统计打开文件数 ==========
lsof | wc -l                      # 总打开文件数
lsof -u alice | wc -l             # 用户打开文件数
lsof -p 1234 | wc -l              # 进程打开文件数

// ========== 实例: 排查端口被占用 ==========
$ lsof -i :3000
// COMMAND  PID   USER   FD   TYPE  DEVICE NODE NAME
// node    5678  alice  12u  IPv4  23456  TCP *:3000 (LISTEN)
// → 发现 node 进程占用 3000 端口
```


> **Note:** 💡 端口排查三板斧: lsof -i :
> 查谁占用了端口,ss -tlnp 看所有监听端口,ss -s 看连接统计。lsof 还可用于排查文件句柄泄漏——如果某个进程打开了大量文件,说明有问题。


## 实战场景


```
// ========== 场景: 查找占用端口 ==========
$ lsof -i :3000
$ kill -9

// ========== 场景: 查看所有监听端口 ==========
$ ss -tlnp
// 快速识别哪些服务在运行

// ========== 场景: 连接数统计 ==========
# 统计各 IP 的连接数
ss -t | grep ESTAB | awk '{print $5}' | cut -d: -f1 | sort | uniq -c | sort -nr | head -10

# 统计 TCP 状态分布
ss -t -a | awk '{print $1}' | sort | uniq -c

# TIME_WAIT 连接数 (过多可能影响性能)
ss -t state time-wait | wc -l

// ========== 场景: 进程句柄泄漏检查 ==========
# 检查某进程的打开文件数
lsof -p 1234 | wc -l
ls -la /proc/1234/fd | wc -l

# 正常范围: 几百
# 异常: 几万几十万 (句柄泄漏!)

// ========== 场景: 路由追踪 ==========
netstat -rn            # 路由表
ip route               # 现代路由查看 (iproute2)
ip neigh               # ARP 表 (邻居发现)

// ========== TCP 连接状态 ==========
// LISTEN       等待连接
// SYN-SENT     已发送 SYN
// SYN-RECEIVED 收到 SYN 回复
// ESTABLISHED  已建立 (正常通信)
// FIN-WAIT-1   关闭中 (等待对方的 FIN ACK)
// FIN-WAIT-2   关闭中 (等待对方 FIN)
// CLOSE-WAIT   对方关闭,等待自己关闭
// CLOSING      同时尝试关闭
// TIME-WAIT    等待 2MSL (保证远程收到 ACK)
// LAST-ACK     等待最后 ACK
// CLOSED       连接结束
```


## 练习


<!-- Converted from: 31_netstat-ss-lsof.html -->
