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

## 三、注意事项与常见陷阱

1. **用户体验**：提供清晰的帮助和错误信息
2. **跨平台**：处理不同平台的路径和行结束符
3. **性能**：CLI 应快速响应
4. **配置管理**：支持配置文件和环境变量
5. **测试**：为 CLI 命令编写集成测试
