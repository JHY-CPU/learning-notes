# CLI工具开发

## 一、概念说明

Rust 非常适合开发命令行工具，性能高、二进制小、跨平台。

```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "mycli")]
#[command(about = "我的 CLI 工具")]
struct Cli {
    #[arg(short, long)]
    name: String,

    #[arg(short, long, default_value_t = 1)]
    count: u8,
}

fn main() {
    let cli = Cli::parse();
    for _ in 0..cli.count {
        println!("你好，{}！", cli.name);
    }
}
```

## 二、具体用法

### 2.1 clap 子命令

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 添加项目
    Add { name: String },
    /// 删除项目
    Remove { id: u64 },
    /// 列出项目
    List,
}
```

### 2.2 输出美化

```rust
use colored::Colorize;
use indicatif::ProgressBar;

fn process() {
    let pb = ProgressBar::new(100);
    for i in 0..100 {
        pb.inc(1);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    pb.finish_with_message("完成".green().to_string());
}
```

### 2.3 配置文件

```rust
use confy;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Config {
    api_key: String,
    output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            api_key: String::new(),
            output_dir: "output".into(),
        }
    }
}

fn main() {
    let cfg: Config = confy::load("mycli").unwrap();
}
```

### 2.4 错误处理与退出码

```rust
use std::process;

fn main() {
    match run() {
        Ok(()) => process::exit(0),
        Err(e) => {
            eprintln!("错误: {}", e);
            process::exit(1);
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Add { name } => add_item(&name)?,
        Commands::Remove { id } => remove_item(id)?,
        Commands::List => list_items()?,
    }
    Ok(())
}
```

### 2.5 交互式提示

```rust
use dialoguer::{Input, Confirm, Select};

fn interactive_config() -> Config {
    let name: String = Input::new()
        .with_prompt("项目名称")
        .interact_text()
        .unwrap();

    let language: usize = Select::new()
        .with_prompt("选择语言")
        .items(&["Rust", "Python", "Go"])
        .interact()
        .unwrap();

    let confirm = Confirm::new()
        .with_prompt("确认创建?")
        .interact()
        .unwrap();

    Config { name, language, confirm }
}
```

### 2.6 Shell 补全生成

```rust
use clap::CommandFactory;
use clap_complete::{generate, shells::Bash};

fn generate_completions() {
    let mut cmd = Cli::command();
    generate(Bash, &mut cmd, "mycli", &mut std::io::stdout());
}

// 使用
// $ mycli generate-completions > /etc/bash_completion.d/mycli
```

## 四、CLI crate 推荐

| 功能 | crate | 说明 |
|------|-------|------|
| 参数解析 | clap | 最流行 |
| 输出着色 | colored | 简单 |
| 进度条 | indicatif | 功能丰富 |
| 交互提示 | dialoguer | 用户输入 |
| 配置管理 | confy | 自动持久化 |
| 表格输出 | comfy-table | 美观 |
| 文件选择 | rfd | 原生对话框 |

## 五、注意事项与常见陷阱

1. **用户体验**：提供清晰的帮助和错误信息，使用 `--help` 自动生成
2. **跨平台**：处理不同平台的路径和行结束符，使用 `std::path`
3. **性能**：CLI 应快速响应，避免不必要的初始化
4. **配置管理**：支持配置文件和环境变量，优先级：CLI > 环境变量 > 配置文件 > 默认值
5. **测试**：为 CLI 命令编写集成测试，使用 `assert_cmd`
6. **信号处理**：处理 Ctrl+C 信号，优雅退出
7. **输出格式**：支持多种输出格式（JSON、表格、纯文本）
