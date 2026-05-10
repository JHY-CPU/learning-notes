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


<!-- Converted from: 20_Node调试.html -->
