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


## 完整 CLI 工具示例

```javascript
#!/usr/bin/env node

// ========== 完整 CLI 工具 ==========
const { Command } = require('commander');
const fs = require('fs');
const path = require('path');

const program = new Command();

program
    .name('mycli')
    .description('示例命令行工具')
    .version('1.0.0');

// init 命令
program
    .command('init <project-name>')
    .description('初始化新项目')
    .option('-t, --template <type>', '项目模板', 'default')
    .option('--typescript', '使用 TypeScript')
    .action((name, options) => {
        console.log(`创建项目: ${name}`);
        console.log(`模板: ${options.template}`);
        console.log(`TypeScript: ${!!options.typescript}`);

        const projectDir = path.resolve(name);
        fs.mkdirSync(projectDir, { recursive: true });
        fs.writeFileSync(
            path.join(projectDir, 'package.json'),
            JSON.stringify({ name, version: '0.1.0', private: true }, null, 2)
        );
        console.log('项目创建成功!');
    });

// list 命令
program
    .command('list')
    .alias('ls')
    .description('列出文件')
    .option('-a, --all', '显示隐藏文件')
    .option('-l, --long', '详细信息')
    .action((options) => {
        const files = fs.readdirSync('.');
        files.forEach(file => {
            if (!options.all && file.startsWith('.')) return;
            if (options.long) {
                const stat = fs.statSync(file);
                console.log(`${stat.isDirectory() ? 'd' : '-'} ${stat.size.toString().padStart(8)} ${file}`);
            } else {
                console.log(file);
            }
        });
    });

// 全局选项
program
    .option('-d, --debug', '调试模式')
    .option('-q, --quiet', '静默模式');

program.parse(process.argv);
```

## 彩色输出与交互

```javascript
// ========== 彩色终端输出 ==========
// 使用 chalk 库
const chalk = require('chalk');

console.log(chalk.green('成功!'));
console.log(chalk.red.bold('错误!'));
console.log(chalk.blue.underline('链接'));
console.log(chalk.bgYellow.black('警告'));

// 渐变色
const gradient = require('gradient-string');
console.log(gradient.rainbow('渐变文本'));

// ========== 进度条 ==========
const cliProgress = require('cli-progress');
const bar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
bar.start(100, 0);
for (let i = 0; i <= 100; i++) {
    bar.update(i);
    // 模拟工作
}
bar.stop();

// ========== 交互式提示 ==========
const inquirer = require('inquirer');

const answers = await inquirer.prompt([
    { type: 'input', name: 'name', message: '你的名字?' },
    { type: 'list', name: 'lang', message: '选择语言:', choices: ['JavaScript', 'TypeScript', 'Python'] },
    { type: 'confirm', name: 'confirm', message: '确认创建?' },
    { type: 'checkbox', name: 'features', message: '选择功能:', choices: ['ESLint', 'Prettier', 'Jest'] },
]);

console.log(answers);
```

## 标准输入/输出与管道

```javascript
// ========== stdin 管道输入 ==========
// echo "hello" | node script.js
// cat file.txt | node script.js

process.stdin.setEncoding('utf-8');
let input = '';
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
    console.log(`收到输入: ${input.trim()}`);
});

// ========== 彩色输出 ==========
// stdout (标准输出，用于数据)
process.stdout.write('普通输出\n');

// stderr (标准错误，用于日志/错误)
process.stderr.write('错误信息\n');

// 区分 stdout/stderr 便于管道:
// node app.js > output.txt 2> errors.txt

// ========== 退出码 ==========
// 成功: process.exit(0)
// 失败: process.exit(1)
// 自定义: process.exit(其他数字)

// 常见退出码:
// 0 — 成功
// 1 — 通用错误
// 2 — 误用 shell 命令
// 126 — 不可执行
// 127 — 命令未找到
```

<!-- Converted from: 16_CLI命令行工具.html -->
