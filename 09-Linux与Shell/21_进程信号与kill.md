# 进程信号与kill


## 🔔 进程信号与 kill


Linux 信号机制、kill/pkill/killall、trap 捕获信号、常见信号处理。


## 信号基础


```
// ========== 什么是信号 ==========
// 信号是进程间通信的一种异步方式
// 内核发送信号给进程,通知它发生了某个事件
// 进程可以:忽略、捕获、或按照默认方式处理

// ========== 常用信号一览 ==========
// 信号名    编号  默认动作  说明
// SIGHUP     1    终止      终端挂断或进程重读配置
// SIGINT     2    终止      键盘中断 Ctrl+C
// SIGQUIT    3    终止+core 键盘退出 Ctrl+\
// SIGKILL    9    终止      强制杀死 (不能捕获/忽略)
// SIGTERM   15    终止      优雅终止 (推荐)
// SIGSTOP   19    暂停      暂停进程 (不能捕获/忽略)
// SIGCONT   18    继续      恢复暂停的进程
// SIGTSTP   20    暂停      终端暂停 Ctrl+Z
// SIGUSR1   10    终止      用户自定义信号 1
// SIGUSR2   12    终止      用户自定义信号 2
// SIGCHLD   17    忽略      子进程状态改变
// SIGPIPE   13    终止      管道破裂 (写入无人读取的管道)
// SIGALRM   14    终止      定时器超时
// SIGABRT    6    终止+core 调用 abort()
// SIGBUS     7    终止+core 总线错误 (内存对齐)
// SIGSEGV   11    终止+core 段错误 (非法内存访问)

// ========== kill 命令语法 ==========
// kill [选项] ...
//
// kill -15 1234      # 默认信号 = SIGTERM (15)
// kill -9 1234       # SIGKILL (强制杀死)
// kill -2 1234       # SIGINT (同 Ctrl+C)
// kill -SIGTERM 1234 # 使用信号名

// 查看所有信号:
kill -l              # 列出所有信号名称
kill -l 9            # KILL (编号转名称)
kill -l KILL         # 9 (名称转编号)

// ========== 选择正确的信号 ==========
// 优先顺序: SIGTERM(15) → SIGINT(2) → SIGKILL(9)
//
// 1. SIGTERM — 首选,优雅关闭
// 2. SIGINT — 类似 Ctrl+C
// 3. SIGHUP — 重读配置 (nginx -s reload 原理)
// 4. SIGKILL — 最后手段 (可能导致数据丢失)

// kill 进程组: 负数表示进程组
kill -15 -1234       # 杀死进程组 1234

// ========== pkill / killall ==========
// 按名称杀死
pkill -9 nginx       # 杀死所有名为 nginx 的进程
pkill -u alice       # 杀死 alice 的所有进程

killall nginx        # 杀死所有 nginx 进程
killall -9 node      # 强制杀死所有 node 进程
killall -u alice     # 杀死 alice 的所有进程
killall -w nginx     # 等待进程真正结束

// pkill 支持正则:
pkill -f "node app.js"    # 匹配完整命令行
```


> **Note:** 💡 杀死进程的原则: 先 SIGTERM (15) 让进程优雅清理,等几秒无响应再用 SIGKILL (9)。SIGKILL 不能被子进程捕获,可能导致孤儿进程、文件损坏或数据丢失。pkill 可以按名称匹配,比 kill 更方便。


## trap — 捕获信号


```
// ========== trap 语法 ==========
// trap <命令> <信号列表>
// 捕获信号后执行指定命令

// ========== 清理临时文件 ==========
#!/bin/bash
TMPFILE=$(mktemp)

# 捕获 EXIT,无论什么原因退出都清理
trap 'rm -f "$TMPFILE"' EXIT

echo "临时文件: $TMPFILE"
# 脚本结束自动删除 $TMPFILE

// ========== 优雅关闭 ==========
#!/bin/bash
cleanup() {
    echo "正在清理..."
    rm -f /var/run/app.pid
    echo "再见!"
    exit 0
}

trap cleanup SIGTERM SIGINT

echo "服务启动, PID: $$"
# 模拟服务运行
while true; do
    sleep 1
done

// ========== 忽略信号 ==========
trap '' SIGINT SIGTERM   # 忽略 Ctrl+C 和终止信号
# 常用于关键脚本,防止被意外中断

// ========== 重置信号处理 ==========
trap - SIGINT            # 重置为默认行为

// ========== EXIT 特殊陷阱 ==========
// EXIT 不是信号,但可用 trap 捕获
// 不管正常退出 (exit) 还是异常退出,都会触发

trap 'echo "脚本退出"' EXIT

// ========== 调试相关 ==========
// DEBUG: 每个命令执行前触发
trap 'echo "执行: $BASH_COMMAND"' DEBUG

// ERR: 命令返回非 0 时触发 (配合 set -e)
trap 'echo "错误发生在行 $LINENO"' ERR

// RETURN: 函数返回时触发
trap 'echo "函数 ${FUNCNAME[0]} 返回"' RETURN
```


## 信号处理实战


```
// ========== 实战: 超时控制 ==========
#!/bin/bash
# 如果命令超过 5 秒未完成,杀死它

TIMEOUT=5
CMD="long_running_task"

# 后台执行命令
$CMD &
CMD_PID=$!

# 设置超时 alarm
(sleep $TIMEOUT && kill $CMD_PID 2>/dev/null) &
ALARM_PID=$!

# 等待命令结束
wait $CMD_PID 2>/dev/null
STATUS=$?

# 取消 alarm
kill $ALARM_PID 2>/dev/null

if [ $STATUS -eq 0 ]; then
    echo "命令成功完成"
else
    echo "命令超时或失败"
fi

// ========== 实战: 重新加载配置 ==========
#!/bin/bash
# 发送 SIGHUP 让守护进程重读配置

reload_service() {
    local service=$1
    local pid_file="/var/run/${service}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        kill -HUP $pid
        echo "已发送 HUP 信号给 $service (PID: $pid)"
    else
        echo "PID 文件不存在: $pid_file"
        exit 1
    fi
}

reload_service nginx

// ========== 查看进程的信号处理 ==========
// 查看进程对每个信号的处置
cat /proc//status | grep -E "^Sig"
cat /proc//status | grep -E "^SigCgt" | \
    awk '{print $2}' | while read mask; do
    # 将位掩码转为信号编号
    # 每个 bit 代表一个信号
done

// 更好用的命令:
pflags           # Solaris
# Linux 用 /proc
awk '{print $1}' /proc//status
```


## 练习


<!-- Converted from: 21_进程信号与kill.html -->
