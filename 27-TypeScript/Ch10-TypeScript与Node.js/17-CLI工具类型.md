# CLI工具类型

## 一、概念说明

Node.js CLI 工具开发常用 `commander` 或 `yargs` 库，它们都提供了 TypeScript 类型支持。类型化的 CLI 定义可以确保命令参数、选项和帮助信息的正确性。

## 二、具体用法

### 2.1 Commander 类型

```typescript
import { Command } from 'commander';

const program = new Command();

program
  .name('my-cli')
  .version('1.0.0')
  .description('我的 CLI 工具');

// 定义命令
program
  .command('create <name>')
  .description('创建新项目')
  .option('-t, --template <type>', '模板类型', 'default')
  .option('-y, --yes', '跳过确认', false)
  .action((name: string, options: { template: string; yes: boolean }) => {
    console.log(`创建项目: ${name}, 模板: ${options.template}`);
  });

// 解析参数
program.parse();
```

### 2.2 Commander 泛型选项

```typescript
import { Command, Option } from 'commander';

// 定义选项接口
interface InitOptions {
  template: 'react' | 'vue' | 'node';
  typescript: boolean;
  git: boolean;
  install: boolean;
}

const initCmd = new Command('init')
  .description('初始化项目')
  .argument('<project-name>', '项目名称')
  .addOption(new Option('-t, --template <type>', '项目模板').choices(['react', 'vue', 'node']))
  .option('--typescript', '使用 TypeScript', false)
  .option('--no-git', '不初始化 Git')
  .option('--no-install', '不安装依赖')
  .action((projectName: string, options: InitOptions) => {
    console.log(projectName, options);
  });

program.addCommand(initCmd);
```

### 2.3 Yargs 类型

```typescript
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

// yargs 支持链式泛型
const argv = yargs(hideBin(process.argv))
  .command('serve [port]', '启动服务器', (yargs) => {
    return yargs
      .positional('port', {
        describe: '端口号',
        default: 3000,
        type: 'number',
      })
      .option('host', {
        alias: 'H',
        type: 'string',
        description: '主机地址',
        default: 'localhost',
      });
  }, (argv) => {
    // argv.port: number, argv.host: string
    console.log(`启动服务器: ${argv.host}:${argv.port}`);
  })
  .strict()
  .parse();
```

### 2.4 提示输入类型

```typescript
import inquirer from 'inquirer';

interface ProjectAnswers {
  projectName: string;
  template: 'react' | 'vue' | 'node';
  features: string[];
  confirm: boolean;
}

async function promptUser(): Promise<ProjectAnswers> {
  const answers = await inquirer.prompt<ProjectAnswers>([
    {
      type: 'input',
      name: 'projectName',
      message: '项目名称:',
      validate: (input: string) => input.length > 0 || '名称不能为空',
    },
    {
      type: 'list',
      name: 'template',
      message: '选择模板:',
      choices: ['react', 'vue', 'node'],
    },
    {
      type: 'checkbox',
      name: 'features',
      message: '选择特性:',
      choices: ['TypeScript', 'ESLint', 'Prettier', 'Testing'],
    },
    {
      type: 'confirm',
      name: 'confirm',
      message: '确认创建?',
    },
  ]);

  return answers;
}
```

### 2.5 进度条与输出类型

```typescript
import ora from 'ora';
import chalk from 'chalk';

async function downloadWithProgress(url: string) {
  const spinner = ora('下载中...').start();

  try {
    await download(url);
    spinner.succeed(chalk.green('下载完成'));
  } catch (err) {
    spinner.fail(chalk.red('下载失败'));
    throw err;
  }
}
```

## 三、注意事项与常见陷阱

1. **Commander 的 `action` 回调参数有类型**：确保参数类型匹配命令定义
2. **Yargs 的泛型在 `.parse()` 时确定**：使用 `await .parseAsync()` 获得类型
3. **`inquirer` 的 `prompt` 返回 Promise**：使用泛型参数定义答案类型
4. **`process.argv` 的类型是 `string[]`**：始终是字符串，需要手动转换
5. **CLI 工具的入口文件需要 `#!/usr/bin/env node`**：用于 Unix 系统
