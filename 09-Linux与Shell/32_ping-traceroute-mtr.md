# ping-traceroute-mtr


## 📡 ping/traceroute/mtr


网络连通性测试、路由追踪、网络质量诊断工具。


## ping — 连通性测试


```
// ========== ping 原理 ==========
// ping 使用 ICMP (Internet Control Message Protocol) Echo Request
// 目标主机回复 Echo Reply
// 用于测试: 连通性、延迟 (RTT)、丢包率

// ========== 基础用法 ==========
ping google.com                  # 持续 ping (Ctrl+C 停止)
ping -c 5 google.com             # 发送 5 个包后停止
ping -4 google.com               # IPv4
ping -6 google.com               # IPv6

// ping 输出解读:
// PING google.com (142.250.80.14) 56(84) bytes of data.
// 64 bytes from 142.250.80.14: icmp_seq=1 ttl=118 time=12.3 ms
// 64 bytes from 142.250.80.14: icmp_seq=2 ttl=118 time=11.8 ms
// ...
// --- google.com ping statistics ---
// 5 packets transmitted, 5 received, 0% packet loss, time 4006ms
// rtt min/avg/max/mdev = 11.5/12.0/12.5/0.3 ms

// ========== 高级用法 ==========
ping -i 0.5 google.com           # 间隔 0.5 秒 (需要 root)
ping -s 1472 google.com          # 指定包大小 (MTU 测试)
ping -W 3 google.com             # 超时 3 秒
ping -q google.com               # 静默 (只显示摘要)
ping -f google.com               # 洪水 ping (需 root,压力测试)
ping -D google.com               # 显示时间戳

// ========== 测试场景 ==========
// 1. 网络是否通
ping -c 1 8.8.8.8 | grep "1 received" && echo "通" || echo "不通"

// 2. DNS 是否正常
ping -c 1 google.com && echo "DNS OK" || echo "DNS 可能有问题"

// 3. 延迟是否高
ping -c 10 google.com | tail -1   # 看 min/avg/max

// 4. 丢包率
ping -c 100 google.com | grep "packet loss"

// ========== 常见错误 ==========
// "Destination Host Unreachable"   → 路由不可达
// "Request timeout"                → 超时无响应
// "TTL expired in transit"         → 跳数过多
// ping: socket: Operation not permitted → 需 root 权限

// TTL 常见值:
// 64   Linux/Mac
// 128  Windows
// 255  Cisco/网络设备
```


## traceroute — 路由追踪


```
// ========== traceroute 原理 ==========
// 利用 TTL (Time To Live) 递增的 UDP/ICMP 包
// 每经过一个路由器 TTL-1,到 0 时路由器返回 ICMP Time Exceeded
// 从而记录路径上的每个跃点

// ========== 基础用法 ==========
traceroute google.com            # 追踪到 google 的路由
traceroute -n google.com         # 不解析域名 (更快)
traceroute -m 30 google.com      # 最大跳数 30 (默认)
traceroute -q 3 google.com       # 每跳发送 3 个探测包

// traceroute 输出:
// traceroute to google.com (142.250.80.14), 30 hops max, 60 byte packets
//  1  192.168.1.1 (192.168.1.1)  1.234 ms  1.123 ms  1.045 ms
//  2  10.0.0.1 (10.0.0.1)  5.234 ms  5.123 ms  5.045 ms
//  3  172.16.0.1 (172.16.0.1)  10.234 ms  12.123 ms  9.045 ms
//  4  * * *                    ← 路由器不响应 ICMP
//  5  203.0.113.1 (203.0.113.1)  30.234 ms  28.123 ms  29.045 ms
//  ...
// 12  142.250.80.14 (142.250.80.14)  12.234 ms  11.123 ms  11.045 ms

// ========== 选项 ==========
// -I  使用 ICMP Echo (绕过某些防火墙)
// -T  使用 TCP SYN (端口 80)
// -U  使用 UDP
// -p  指定目标端口
// -w  超时时间

// 使用 TCP 模式 (常能穿透防火墙):
traceroute -T -p 80 google.com

// ========== Windows 命令 ==========
tracert google.com               # Windows 路由追踪
pathping google.com              # 路由追踪+延迟统计

// ========== 诊断场景 ==========
// 1. 找出延迟在哪一跳突增 → 定位网络瓶颈
// 2. 找出哪一跳丢包 → 定位故障路由器
// 3. 确认路由是否经过预期路径 → 绕路检测
```


## mtr — 网络质量诊断


```
// ========== mtr ==========
// mtr = My Traceroute
// 结合 ping + traceroute,持续监控每跳延迟和丢包
// 是诊断网络问题的最佳工具

// 安装:
apt install mtr                      # Debian/Ubuntu
yum install mtr                      # CentOS/RHEL

// ========== 基础用法 ==========
mtr google.com                       # 交互式 (按 q 退出)
mtr -r google.com                    # 报告模式 (一次性)
mtr -c 10 google.com                 # 发送 10 个包后停止
mtr -n google.com                    # 不解析域名
mtr -4 google.com                    # IPv4

// mtr 输出:
//                                Packets               Drop
// Host                          Loss%   Snt   Last   Avg  Best  Wrst StDev
// 1. 192.168.1.1                0.0%    10    1.2   1.1   0.8   1.5   0.2
// 2. 10.0.0.1                   0.0%    10    5.2   5.5   4.8   6.1   0.4
// 3. 172.16.0.1                0.0%    10   12.1  12.5  11.8  13.2   0.5
// 4. 203.0.113.1               10.0%    10   30.2  31.1  29.5  35.2   1.8  ← 丢包!
// 5. 142.250.80.14               0.0%    10   11.8  12.2  11.5  13.0   0.5

// ========== 报告模式 ==========
mtr -r -c 50 google.com > mtr_report.txt  # 输出报告到文件

// ========== mtr 选项 ==========
// --report-wide   宽格式 (更多列)
// --json          JSON 输出
// --csv           CSV 输出
// --no-dns        不解析域名 (同 -n)
// -i <秒>         发送间隔

// ========== 诊断指南 ==========
// 1. 如果有跳显示 Loss > 0%,但下一跳 Loss=0%
//    → 该路由器限流 ICMP,不是真实丢包
// 2. 如果最后一跳 Loss > 0%
//    → 目标主机有真实丢包问题
// 3. 如果某跳延迟大幅增加
//    → 该链路可能存在拥塞
// 4. 如果路由绕路 (跳数异常多)
//    → 可能 BGP 路由问题

// ========== 其他网络工具 ==========
// dig/nslookup   DNS 查询
// whois          域名/IP 归属查询
// nmap           端口扫描
// tcpdump        抓包分析
// iperf3         带宽测试
```


> **Note:** 💡 mtr 是网络诊断"瑞士军刀"——持续显示每跳的延迟和丢包率。排查网络问题先用 mtr -r -c 100
> 收集 100 个包的数据,比单个 ping 或 traceroute 更全面。如果 mtr 不通,再用 tcpdump 抓包分析。


## 实战场景


```
// ========== 场景: 网站慢排查 ==========
# 1. DNS 是否慢
dig google.com | grep "Query time"

# 2. 连接是否慢
mtr -r -c 50 google.com

# 3. 到服务器的延迟
ping -c 50 google.com | tail -1

# 4. 带宽是否够
iperf3 -c iperf.example.com

// ========== 场景: 服务器不通排查 ==========
# 1. 能 ping 通吗
ping -c 3 $SERVER

# 2. 路由可达吗
traceroute -n $SERVER

# 3. 端口开放吗
nc -zv $SERVER 22
nmap -p 80,443 $SERVER

# 4. DNS 解析正常吗
nslookup $DOMAIN

// ========== 场景: MTU 问题排查 ==========
# 最大 MTU (以太网 1500,减去 IP+ICMP 头 = 1472)
ping -c 3 -M do -s 1472 google.com    # 正常应通
ping -c 3 -M do -s 1473 google.com    # 应提示 Frag needed

# -M do = 禁止分片 (DF 标志)
# 找到最大 MTU 后设置:
ip link set dev eth0 mtu 1500
```


## 练习


<!-- Converted from: 32_ping-traceroute-mtr.html -->
