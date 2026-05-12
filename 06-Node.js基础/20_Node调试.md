# Node调试


## Node 调试


--inspect 标志、Chrome DevTools、VSCode launch.json、debugger 语句。


## Node 调试方式


```
// ========== debugger 语句 ==========
function calculate(x) {
    debugger; // 断点
    return x * 2;
}

// ========== 启动调试 ==========
// $ node --inspect app.js
// $ node --inspect-brk app.js  (在第一行暂停)
// $ node --inspect=0.0.0.0:9229 app.js

// ========== Chrome DevTools ==========
// chrome://inspect → 远程目标 → inspect

// ========== VSCode launch.json ==========
// {
//   "type": "node",
//   "request": "launch",
//   "name": "调试程序",
//   "program": "${workspaceFolder}/app.js"
// }

// ========== Console 调试 ==========
console.log('值:', x);
console.table(array);
console.time('label');
console.trace('堆栈');
console.dir(obj, { depth: 3, colors: true });
```


## 演示：调试技术

点击按钮查看


## 高级调试技巧

```javascript
// ========== 条件断点与日志断点 ==========
function processArray(arr) {
    for (let i = 0; i < arr.length; i++) {
        // VSCode 中右键行号 → Add Conditional Breakpoint
        // 条件: i === 42  (只在第42次迭代时暂停)
        debugger; // 也可以使用条件式
        if (arr[i].value > 100) {
            console.log('找到大值:', arr[i]);
        }
    }
}

// ========== 内存分析 ==========
// Chrome DevTools → Memory 面板
// 1. Take Heap Snapshot — 堆快照
// 2. Record Allocation Timeline — 分配时间线
// 3. Record Allocation Profile — 分配统计

// 代码中手动触发 GC (需要 --expose-gc)
if (global.gc) {
    global.gc();
    console.log(process.memoryUsage());
}

// ========== CPU 分析 ==========
// $ node --prof app.js        → 生成 isolate-*.log
// $ node --prof-process       → 解析为可读报告

// 或使用 Chrome DevTools
// $ node --inspect app.js
// → chrome://inspect → Profiler → Start
```

## VSCode 高级调试配置

```json
// ========== launch.json 完整配置 ==========
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "node",
            "request": "launch",
            "name": "启动程序",
            "program": "${workspaceFolder}/src/index.js",
            "args": ["--verbose"],
            "env": {
                "NODE_ENV": "development",
                "DEBUG": "app:*"
            },
            "console": "integratedTerminal",
            "restart": true
        },
        {
            "type": "node",
            "request": "launch",
            "name": "运行当前测试",
            "program": "${workspaceFolder}/node_modules/.bin/jest",
            "args": ["${file}", "--no-coverage"],
            "console": "integratedTerminal"
        },
        {
            "type": "node",
            "request": "attach",
            "name": "附加到进程",
            "port": 9229,
            "restart": true,
            "skipFiles": ["<node_internals>/**"]
        },
        {
            "type": "node",
            "request": "launch",
            "name": "调试 NPM 脚本",
            "runtimeExecutable": "npm",
            "runtimeArgs": ["run", "dev"],
            "console": "integratedTerminal"
        }
    ]
}
```

## 性能分析与诊断

```javascript
// ========== 性能测量 API ==========
const { performance, PerformanceObserver } = require('perf_hooks');

// 创建性能观察器
const obs = new PerformanceObserver((items) => {
    items.getEntries().forEach(entry => {
        console.log(`${entry.name}: ${entry.duration.toFixed(2)}ms`);
    });
});
obs.observe({ entryTypes: ['measure'] });

// 测量函数执行时间
performance.mark('start');
// ... 执行一些操作 ...
performance.mark('end');
performance.measure('操作耗时', 'start', 'end');

// ========== Node.js 诊断报告 ==========
// $ node --report-on-signal app.js  (收到信号时生成报告)
// $ node --report-uncaught-exception app.js (异常时生成报告)

// 代码中触发
process.report.writeReport();       // 写入当前报告
process.report.writeReport('诊断报告.json'); // 指定文件名

// ========== Event Loop 延迟检测 ==========
const { monitorEventLoopDelay } = require('perf_hooks');
const h = monitorEventLoopDelay({ resolution: 20 });
h.enable();

// ... 运行一段时间后 ...
console.log(`平均延迟: ${(h.mean / 1e6).toFixed(2)}ms`);
console.log(`最大延迟: ${(h.max / 1e6).toFixed(2)}ms`);
console.log(`99分位: ${(h.percentile(99) / 1e6).toFixed(2)}ms`);
h.disable();
```

<!-- Converted from: 20_Node调试.html -->
