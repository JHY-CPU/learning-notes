# Python asyncio基础


## 🔄 Python asyncio 基础


异步编程概念、事件循环 event loop、协程 async/await、Task 任务、Future、asyncio.run/create_task/gather/wait 核心 API。


## 异步编程概念


```
// ========== 同步 vs 异步 ==========
// 同步: 按顺序执行,前一个完成才做下一个
// 异步: 遇到等待时切换做别的,不阻塞

// 传统同步:
import time

def task(name, seconds):
    print(f"{name} 开始")
    time.sleep(seconds)  # 阻塞!
    print(f"{name} 完成")
    return name

# 总耗时: 1+2+3 = 6 秒
task("A", 1)
task("B", 2)
task("C", 3)

// 异步 (asyncio):
import asyncio

async def task(name, seconds):
    print(f"{name} 开始")
    await asyncio.sleep(seconds)  # 非阻塞!
    print(f"{name} 完成")
    return name

# 总耗时: 3 秒 (并发执行)
async def main():
    await asyncio.gather(
        task("A", 1),
        task("B", 2),
        task("C", 3),
    )

asyncio.run(main())
```


## 协程 async/await


```
// ========== 定义协程 ==========
import asyncio

# async def → 定义协程函数
async def hello():
    print("Hello")
    await asyncio.sleep(1)  # await 挂起当前协程
    print("World")
    return "done"

# 调用协程函数 → 创建协程对象
coro = hello()
print(type(coro))  #

# 运行协程的三种方式:
# 1. asyncio.run() — 入口
result = asyncio.run(hello())

# 2. await — 在另一个协程中
async def main():
    result = await hello()
    print(result)

# 3. asyncio.create_task() — 并发执行
async def main():
    task = asyncio.create_task(hello())
    result = await task

# ========== await 只能用在协程中 ==========
# ✅ 正确:
async def foo():
    await asyncio.sleep(1)

# ❌ 错误 (不能在同步函数中 await):
# def bar():
#     await asyncio.sleep(1)

# ========== 可等待对象 ==========
# 1. 协程 (coroutine)
# 2. Task (任务)
# 3. Future (未来对象)
```


## asyncio.run()


```
// ========== asyncio.run() ==========
import asyncio

# asyncio.run() 是异步程序的入口
# 功能:
# 1. 创建事件循环
# 2. 运行协程直到完成
# 3. 关闭事件循环

async def main():
    print("异步程序开始")
    await asyncio.sleep(1)
    print("异步程序结束")
    return 42

result = asyncio.run(main())
print(result)  # 42

# Python 3.7+ 推荐方式
# 替代旧的:
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())
# loop.close()

# ========== 调试模式 ==========
# asyncio.run(main(), debug=True)
# 会启用:
# - 检查未等待的协程
# - 记录慢回调 (> 100ms)
# - 更详细的错误信息

# 环境变量:
# PYTHONASYNCIODEBUG=1
```


## Task 任务


```
// ========== create_task ==========
import asyncio

async def say_after(delay, msg):
    await asyncio.sleep(delay)
    print(msg)

async def main():
    # 创建任务 (立即开始,不等待)
    task1 = asyncio.create_task(say_after(2, "任务1完成"))
    task2 = asyncio.create_task(say_after(1, "任务2完成"))

    print("两个任务已启动,并发执行中...")

    # 等待任务完成 (此时已并发运行)
    await task1
    await task2

    print("全部完成")
    # 总耗时: 2 秒 (不是 2+1=3!)

asyncio.run(main())

# ========== Task 方法和属性 ==========
async def demo_task():
    task = asyncio.create_task(say_after(1, "Hello"))

    print(task.done())      # False (未完成)
    print(task.cancelled()) # False

    # 取消任务:
    # task.cancel()

    # 获取结果:
    # result = await task

    # 设置名称:
    task.set_name("my-task")
    print(task.get_name())  # 'my-task'
```


## asyncio.gather / wait


```
// ========== asyncio.gather ==========
import asyncio

async def fetch_data(id, delay):
    await asyncio.sleep(delay)
    return {"id": id, "data": f"data-{id}"}

async def main():
    # 并发执行多个协程,返回结果列表
    results = await asyncio.gather(
        fetch_data(1, 3),
        fetch_data(2, 1),
        fetch_data(3, 2),
    )
    print(results)
    # [{'id': 1, 'data': 'data-1'},
    #  {'id': 2, 'data': 'data-2'},
    #  {'id': 3, 'data': 'data-3'}]
    # 总耗时: 3 秒 (最长的任务)

    # return_exceptions=True: 异常当作结果返回
    results = await asyncio.gather(
        fetch_data(1, 1),
        might_fail(),          # 可能失败
        fetch_data(3, 2),
        return_exceptions=True  # 不抛出,返回异常对象
    )

# ========== asyncio.wait ==========
async def main():
    tasks = [
        asyncio.create_task(fetch_data(1, 3)),
        asyncio.create_task(fetch_data(2, 1)),
        asyncio.create_task(fetch_data(3, 2)),
    ]

    # 等待所有完成:
    done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    # 等待第一个完成:
    # done, pending = await asyncio.wait(tasks, return_when=FIRST_COMPLETED)

    # 等待第一个异常:
    # done, pending = await asyncio.wait(tasks, return_when=FIRST_EXCEPTION)

    for task in done:
        print(task.result())
```


> **Note:** 💡 asyncio 要点: (1) async def 定义协程,await 等待可等待对象; (2) asyncio.run(main()) 是入口函数; (3) create_task 创建任务实现并发; (4) gather 等待所有协程完成并收集结果; (5) wait 更灵活的等待策略 (完成/第一个/异常)。


## 练习


<!-- Converted from: 106_Python asyncio基础.html -->
