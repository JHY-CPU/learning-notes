# Python while循环


## 🔄 Python while 循环


while 循环基础、无限循环、输入验证、循环控制、while-else 子句、常见应用场景。


## while 循环基础


```
// ========== 基本语法 ==========
// while 条件:
//     循环体

count = 0
while count < 5:
    print(count)
    count += 1
// 0, 1, 2, 3, 4

// ========== 计数器模式 ==========
// 用 while 实现 for 的效果
i = 0
while i < 10:
    print(i)
    i += 1
// 0~9

// ⚠️ 如果忘记 i += 1,就会无限循环!

// ========== 什么时候用 while ==========
// for:   遍历已知序列 (列表/字符串/range...)
// while: 条件循环,不知道要循环多少次

// 适合 while 的场景:
// - 输入验证 (直到输入正确)
// - 游戏循环 (直到游戏结束)
// - 轮询检查 (直到条件满足)
// - 文件读取 (直到文件末尾)
// - 无限循环 (直到 break)
```


## 输入验证


```
// ========== 直到输入正确 ==========
while True:
    user_input = input("输入 'quit' 退出: ")
    if user_input == "quit":
        break
    print(f"你输入了: {user_input}")

// ========== 验证数字输入 ==========
while True:
    try:
        age = int(input("请输入年龄 (0-150): "))
        if 0 <= age <= 150:
            break
        print("年龄范围 0-150")
    except ValueError:
        print("请输入有效数字!")

print(f"年龄: {age}")

// ========== yes/no 确认 ==========
while True:
    response = input("是否继续? (y/n): ").lower().strip()
    if response in ("y", "yes"):
        print("继续执行")
        break
    elif response in ("n", "no"):
        print("退出")
        break
    print("请输入 y 或 n")

// ========== 海象运算符简化 ==========
// Python 3.8+
while (text := input("输入: ")) != "quit":
    print(f"你输入了: {text}")
```


## 无限循环与控制


```
// ========== 无限循环 ==========
// 无限循环本身不是坏事,关键是确保有退出条件

// ✅ 有 break 的安全无限循环:
while True:
    cmd = input("> ")
    if cmd == "quit":
        break
    if cmd == "help":
        print("可用命令: quit, help, version")
        continue
    print(f"执行: {cmd}")

// ⚠️ 无限循环陷阱:
// while True:          # 没有 break,永远不退出
//     print("卡住了!")

// ========== break 与 continue ==========
// break:    退出整个循环
// continue: 跳过本次循环剩余部分,进入下一次

// break 示例:
n = 0
while n < 10:
    n += 1
    if n == 5:
        break           # n=5 时退出
    print(n)
// 1, 2, 3, 4

// continue 示例:
n = 0
while n < 10:
    n += 1
    if n % 2 == 0:
        continue        # 偶数跳过
    print(n)
// 1, 3, 5, 7, 9

// ========== while 模拟 do-while ==========
// Python 没有 do-while,但可以用 True + break 模拟

// do-while: 至少执行一次循环体
while True:
    data = get_data()
    if not data:
        break
    process(data)
```


## 实用场景


```
// ========== 场景 1: 文件读取 ==========
with open("data.txt") as f:
    while True:
        line = f.readline()
        if not line:   # 文件末尾
            break
        print(line.strip())

// 更简洁的方式:
for line in open("data.txt"):
    print(line.strip())

// ========== 场景 2: 轮询等待 ==========
import time

def wait_for_condition(max_retries=10, interval=1):
    """轮询直到条件满足或超时"""
    retries = 0
    while retries < max_retries:
        if check_condition():   # 检查条件
            return True
        retries += 1
        time.sleep(interval)
    return False

// ========== 场景 3: 重试机制 ==========
def fetch_data_with_retry(url, max_retries=3):
    """带重试的数据获取"""
    retries = 0
    while retries < max_retries:
        try:
            return requests.get(url)
        except Exception as e:
            retries += 1
            if retries == max_retries:
                raise
            print(f"重试 {retries}/{max_retries}")
            time.sleep(1)

// ========== 场景 4: 进度循环 ==========
import time
progress = 0
while progress < 100:
    progress += 10
    print(f"\r进度: {progress}%", end="", flush=True)
    time.sleep(0.5)
print("\n完成!")
```


> **Note:** 💡 while 要点: (1) while 用于"不知道要循环多少次"的场景; (2) while True + break 是常见模式; (3) 确保循环条件最终会变为 False; (4) continue 跳过本次, break 退出整个循环; (5) for 能解决的问题不用 while。


## 练习


<!-- Converted from: 29_Python while循环.html -->
