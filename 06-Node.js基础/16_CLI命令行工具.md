# CLI命令行工具


## CLI 命令行工具


命令行参数解析、stdin/stdout/stderr、交互输入 readline。


## CLI 开发


```
// ========== shebang ==========
#!/usr/bin/env node
// 放在文件第一行，标识 Node.js 运行

// ========== 参数解析 ==========
// process.argv 获取原始参数
const args = process.argv.slice(2);

// 简易解析
const options = {};
for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith('--')) {
        const key = args[i].slice(2);
        options[key] = args[i + 1]?.startsWith('--') ? true : args[i + 1];
    }
}

// ========== 使用 commander (推荐) ==========
const { Command } = require('commander');
const program = new Command();
program
    .name('my-cli')
    .version('1.0.0')
    .option('-n, --name ', '你的名字')
    .option('-v, --verbose', '详细输出')
    .parse(process.argv);

// ========== stdin/stdout/stderr ==========
process.stdout.write('输出到控制台\n');
process.stderr.write('错误输出\n');

// 管道输入读取
process.stdin.on('data', (chunk) => {
    console.log('输入:', chunk.toString());
});

// ========== readline 交互 ==========
const readline = require('readline');
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

rl.question('你叫什么名字？', (answer) => {
    console.log(`你好, ${answer}!`);
    rl.close();
});
```


## 演示：CLI 工具

点击按钮查看


<!-- Converted from: 16_CLI命令行工具.html -->
