# 进程管理ps-top


## 📊 进程管理 ps/top/htop


进程概念、ps 进程快照、top/htop 实时监控、进程状态、进程树。


## 进程基础


```
// ========== 进程概念 ==========
// 进程 = 运行中的程序实例
// 每个进程有唯一 PID (Process ID)
// PPID = 父进程 ID
// 所有进程的祖先是 systemd (PID 1) 或 init

// ========== 进程状态 ==========
// R (Running)        运行中或可运行
// S (Sleeping)       睡眠中,可中断 (等待事件)
// D (Uninterruptible) 不可中断睡眠 (I/O 等待)
// T (Stopped)        已停止 (信号暂停)
// Z (Zombie)         僵尸进程 (已结束但父进程未回收)
// X (Dead)           已死亡 (几乎看不到)

// 进程状态切换:
// 创建 → 就绪 → 运行 → 阻塞 → 就绪 ... → 终止 → 僵尸

// ========== 进程调度 ==========
// Linux 使用 CFS (Completely Fair Scheduler)
// 优先级: nice 值 -20 (最高) ~ 19 (最低), 默认 0
// 实时进程: SCHED_FIFO / SCHED_RR (优先级 1-99)
```


## ps — 进程快照


```
// ========== ps 基础 ==========
// ps -ef     # 显示所有进程 (标准格式)
// ps aux     # 显示所有进程 (BSD 格式)
// ps -e      # 所有进程
// ps -f      # 完整格式

// ps aux 输出字段:
// USER      进程所有者
// PID       进程 ID
// %CPU      CPU 使用率
// %MEM      内存使用率
// VSZ       虚拟内存大小 (KB)
// RSS       常驻内存大小 (KB)
// TTY       终端
// STAT      进程状态
// START     启动时间
// TIME      CPU 占用时间
// COMMAND   命令名

// ========== 常用 ps ==========
// 查看特定用户的进程
ps -u alice
ps -U root -u root

// 查看进程树
ps -ef --forest       # 树形显示
ps auxf               # BSD 树形

// 自定义输出格式
ps -eo pid,ppid,cmd,%cpu,%mem --sort=-%cpu
ps -eo pid,user,start,time,command

// 查看线程
ps -eLf               # 显示线程 (LWP)
ps -T -p         # 查看指定进程的线程

// 查找进程
ps aux | grep nginx
pgrep -l nginx        # 只找 PID + 名
pgrep -a nginx        # PID + 完整命令行

// ========== pstree — 进程树 ==========
pstree                # 树形进程
pstree -p             # 显示 PID
pstree -s        # 显示该进程的祖先链
pstree          # 指定用户的进程
```


## top / htop — 实时监控


```
// ========== top ==========
// 默认每 3 秒刷新一次

// top 输出概要:
// top - 10:00:00 up 3 days, 1 user, load average: 0.50, 0.30, 0.20
// Tasks: 120 total, 1 running, 119 sleeping, 0 stopped, 0 zombie
// %Cpu(s): 5.0 us, 1.0 sy, 0.0 ni, 94.0 id, 0.0 wa, 0.0 hi, 0.0 si
// MiB Mem : 7856.0 total, 1024.0 free, 4000.0 used, 2832.0 buff/cache
// MiB Swap: 2048.0 total, 1800.0 free, 248.0 used. 3000.0 avail Mem

// top 交互命令 (按按键):
// 1      查看每个 CPU 核心
// P      按 CPU 排序
// M      按内存排序
// k      杀死进程 (输入 PID)
// r      修改 nice 值 (renice)
// u      查看特定用户
// H      显示线程
// q      退出
// z      彩色显示
// c      显示完整命令行
// V      树形视图

// ========== top 批处理模式 ==========
// 用于脚本采集
top -b -n 1                    # 一次性输出
top -b -n 5 -d 2 > top.log    # 5次,间隔2秒
top -b -n 1 -p 1234,5678      # 监控指定 PID

// ========== htop ==========
// htop 是 top 的增强版 (需安装)
// 特性:
// - 彩色显示
// - 鼠标操作
// - 树形视图 (F5)
// - 垂直/水平滚动
// - 搜索进程 (F3)
// - 过滤进程 (F4)
// - 更直观的 CPU/内存条

// 安装: apt install htop / yum install htop

// ========== glances ==========
// glances 是跨平台的监控工具
// glances                    # 终端监控
// glances -c server.com      # 作为客户端连接
// glances -s                # 作为服务端
// glances -w                # Web 界面

// ========== 常用监控组合 ==========
// CPU 最高进程:   ps aux --sort=-%cpu | head -5
// 内存最高进程:   ps aux --sort=-%mem | head -5
// 总进程数:       ps -e | wc -l
// 某用户进程数:   ps -u alice | wc -l
```


> **Note:** 💡 ps aux --sort=-%cpu | head -5 是定位 CPU 瓶颈的快速方法。top 中按 P/M 可以交互式排序。需要实时监控推荐 htop,需要历史趋势推荐集成到 Prometheus/Grafana。


## 进程间关系


```
// ========== 进程关系 ==========
// 父子关系: 父进程 fork 子进程
// 孤儿进程: 父进程先结束,子进程被 init/systemd 收养
// 僵尸进程: 子进程结束但父进程未 wait,残留 PID

// ========== 孤儿进程示例 ==========
// $ sleep 100 &
// $ kill $PPID           # 父进程退出
// $ ps -o pid,ppid,cmd   # 子进程 PPID 变为 1 (systemd)

// ========== 僵尸进程 ==========
// 僵尸无法直接 kill,需要:
// 1. 杀死父进程 (让僵尸被 init 收养回收)
// 2. 或让父进程调用 wait()

// 查找僵尸:
ps aux | grep 'Z'
ps -eo pid,stat,cmd | awk '$2 ~ /Z/'

// ========== 守护进程 ==========
// 后台运行、无终端、PPID 为 1 的进程
// 通常以 d 结尾: sshd, nginx, systemd
// 创建守护进程步骤:
// 1. fork → 父进程退出
// 2. 设置新会话 setsid()
// 3. 关闭文件描述符
// 4. 切换工作目录到 /
// 5. 重定向 stdin/stdout/stderr 到 /dev/null
```


## 练习


<!-- Converted from: 20_进程管理ps-top.html -->
