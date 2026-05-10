# Python流程控制综合


## 📊 Python 流程控制综合


流程控制总结、for vs while 选择、嵌套控制、综合案例、代码风格。


## 流程控制总览


```
// ========== Python 流程控制总结 ==========
// 条件分支:
//   if / elif / else         — 条件判断
//   match / case             — 模式匹配 (3.10+)
//   三元表达式                — 简单条件赋值

// 循环:
//   for ... in ...           — 遍历可迭代对象
//   while ...                — 条件循环
//   for-else / while-else    — 循环正常结束

// 循环控制:
//   break                    — 退出循环
//   continue                 — 跳过本次
//   pass                     — 占位符

// 其他控制:
//   return                   — 函数返回
//   raise                    — 抛出异常
//   yield                    — 生成器
//   async / await            — 异步

// ========== 选择指南 ==========
// 遍历序列 (列表/字符串/文件)   → for
// 遍历 range()               → for
// 需要索引                   → enumerate()
// 不确定循环次数             → while
// 输入验证                   → while True + break
// 无限循环                   → while True + 退出条件
```


## for vs while 对比


```
// ========== for ==========
// 适用: 遍历已知可迭代对象

// ✅ 遍历列表:
for fruit in ["apple", "banana"]:
    print(fruit)

// ✅ 固定次数:
for i in range(100):
    process(i)

// ✅ 遍历文件:
for line in open("file.txt"):
    print(line.strip())

// ✅ 并行遍历:
for name, age in zip(names, ages):
    print(name, age)

// ========== while ==========
// 适用: 条件驱动,未知次数

// ✅ 输入验证:
while (text := input("> ")) != "quit":
    process(text)

// ✅ 轮询:
while not server.ready():
    time.sleep(1)

// ✅ 重试:
while retries < max_retries:
    try:
        return fetch_data()
    except:
        retries += 1

// ✅ 游戏循环:
while player.health > 0:
    player.update()
    game.render()

// ========== 经验法则 ==========
// 如果能用 for 解决,就不用 while
// for 更安全 (不会忘记更新循环变量)
// while 更灵活 (条件可以是任何表达式)
```


## 综合案例


```
// ========== 案例: 简易计算器 ==========
def calculator():
    """命令行计算器,直到输入 quit 退出"""
    print("简易计算器 (输入 quit 退出)")

    while True:
        expr = input("\n表达式 (如: 2 + 3): ").strip()
        if expr.lower() == "quit":
            print("再见!")
            break

        parts = expr.split()
        if len(parts) != 3:
            print("格式: 数字 运算符 数字")
            continue

        a, op, b = parts
        try:
            a, b = float(a), float(b)
        except ValueError:
            print("无效数字")
            continue

        if op == "+":
            print(f"= {a + b}")
        elif op == "-":
            print(f"= {a - b}")
        elif op == "*":
            print(f"= {a * b}")
        elif op == "/":
            if b == 0:
                print("除数不能为零")
                continue
            print(f"= {a / b}")
        else:
            print(f"无效运算符: {op}")

// ========== 案例: 学生成绩统计 ==========
def analyze_scores(scores):
    """分析学生成绩"""
    if not scores:
        return "没有成绩数据"

    # 计算统计值
    total = sum(scores)
    count = len(scores)
    average = total / count
    highest = max(scores)
    lowest = min(scores)

    # 等级分布
    distribution = {"优秀(≥90)": 0, "良好(80-89)": 0,
                    "及格(60-79)": 0, "不及格(<60)": 0}

    for score in scores:
        if score >= 90:
            distribution["优秀(≥90)"] += 1
        elif score >= 80:
            distribution["良好(80-89)"] += 1
        elif score >= 60:
            distribution["及格(60-79)"] += 1
        else:
            distribution["不及格(<60)"] += 1

    # 找出最高分对应的学生 (假设有名字列表)
    return {
        "总数": count,
        "总分": total,
        "平均分": round(average, 2),
        "最高分": highest,
        "最低分": lowest,
        "分布": distribution,
    }

scores = [85, 92, 67, 78, 55, 90, 88, 73]
result = analyze_scores(scores)
for k, v in result.items():
    print(f"{k}: {v}")
```


## Pythonic 控制流风格


```
// ========== Pythonic 控制流原则 ==========
// 1. 避免深层嵌套 (提前返回)
// 2. 用 in 简化多条件
// 3. 用 any/all 简化循环
// 4. 用推导式替代简单循环
// 5. 用 for 替代 while (能用则用)

// ========== 提前返回 ==========
// ❌ 嵌套过深:
def validate_user(user):
    if user:
        if user.is_active:
            if user.has_permission:
                return True
    return False

// ✅ 提前返回:
def validate_user(user):
    if not user:
        return False
    if not user.is_active:
        return False
    if not user.has_permission:
        return False
    return True

// ========== any/all 替代循环 ==========
// 检查是否所有元素都满足条件:
all(x > 0 for x in numbers)      # 全部正数?
any(x < 0 for x in numbers)      # 有负数?

// 比显式循环简洁:
// ❌
all_positive = True
for x in numbers:
    if x <= 0:
        all_positive = False
        break

// ✅
all_positive = all(x > 0 for x in numbers)

// ========== 推导式替代 ==========
// ❌
result = []
for x in items:
    if condition(x):
        result.append(transform(x))

// ✅
result = [transform(x) for x in items if condition(x)]

// ========== 最终建议 ==========
// 清晰 > 简洁 > 性能
// 如果一行代码需要 3 秒才能看懂,拆开写
```


> **Note:** 💡 流程控制最佳实践: (1) 能用 for 不用 while; (2) 用 enumerate 避免 range(len()); (3) 用 any/all 简化条件循环; (4) 提前返回避免深层嵌套; (5) 推导式替代简单 for+append; (6) 代码清晰永远是第一位的。


## 练习


<!-- Converted from: 32_Python流程控制综合.html -->
