# 嵌入式Rust

## 一、概念说明

Rust 适合嵌入式开发，提供内存安全、零成本抽象和无运行时。

```rust
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;

#[entry]
fn main() -> ! {
    loop {
        // 嵌入式主循环
    }
}
```

## 二、具体用法

### 2.1 基本设置

```toml
# Cargo.toml
[dependencies]
cortex-m = "0.7"
cortex-m-rt = "0.7"
panic-halt = "0.2"
embedded-hal = "0.2"

[[bin]]
name = "app"
test = false
bench = false
```

### 2.2 GPIO 操作

```rust
use embedded_hal::digital::v2::OutputPin;

fn blink<P: OutputPin>(led: &mut P) -> Result<(), P::Error> {
    led.set_high()?;
    delay_ms(1000);
    led.set_low()?;
    delay_ms(1000);
    Ok(())
}
```

### 2.3 中断处理

```rust
use cortex_m::interrupt;

#[interrupt]
fn TIM2() {
    // 中断处理
}

fn enable_interrupt() {
    interrupt::free(|cs| {
        // 临界区代码
    });
}
```

### 2.4 RTOS 集成

```rust
use rtic::app;

#[app(device = stm32f4xx_hal::pac, peripherals = true)]
mod app {
    #[shared]
    struct Shared {
        counter: u32,
    }

    #[init]
    fn init(cx: init::Context) -> (Shared, init::Monotonics) {
        (Shared { counter: 0 }, init::Monotonics())
    }

    #[task(shared = [counter])]
    fn task1(mut cx: task1::Context) {
        cx.shared.counter.lock(|c| *c += 1);
    }
}
```

### 2.5 RTIC 框架

```rust
use rtic::app;

#[app(device = stm32f4xx_hal::pac, peripherals = true, dispatchers = [TIM2])]
mod app {
    #[shared]
    struct Shared {
        counter: u32,
    }

    #[local]
    struct Local {
        led: PA5<Output<PushPull>>,
    }

    #[init]
    fn init(cx: init::Context) -> (Shared, Local, init::Monotonics) {
        // 初始化硬件
        let mut led = gpioa.pa5.into_push_pull_output();

        (Shared { counter: 0 }, Local { led }, init::Monotonics())
    }

    #[task(shared = [counter], priority = 1)]
    fn increment(mut cx: increment::Context) {
        cx.shared.counter.lock(|c| *c += 1);
    }

    #[task(local = [led], priority = 2)]
    fn blink(cx: blink::Context) {
        cx.local.led.toggle();
    }
}
```

### 2.6 调试工具

```bash
# OpenOCD + GDB
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg
arm-none-eabi-gdb target/thumbv7em-none-eabihf/release/app

# probe-rs
cargo install probe-rs
cargo embed --release

# defmt 日志（高效嵌入式日志）
# Cargo.toml: defmt = "0.3", defmt-rtt = "0.4"
```

## 四、嵌入式 crate 推荐

| 功能 | crate | 说明 |
|------|-------|------|
| HAL 抽象 | stm32f4xx-hal | STM32 支持 |
| 异步运行时 | RTIC | 实时中断驱动 |
| 日志 | defmt | 高效日志 |
| 驱动 | embedded-hal | HAL trait |
| 协议 | embedded-serial | UART/SPI/I2C |
| 网络 | smoltcp | TCP/IP 栈 |

## 五、注意事项与常见陷阱

1. **no_std**：嵌入式环境不支持标准库，使用 `#![no_std]` 和 `#![no_main]`
2. **内存限制**：注意堆和栈的使用，嵌入式设备通常没有堆
3. **实时性**：确保满足实时性要求，使用 RTIC 管理优先级
4. **调试工具**：使用调试器（OpenOCD、probe-rs）进行硬件调试
5. **安全认证**：嵌入式系统可能需要安全认证，遵循 MISRA 等标准
6. **中断安全**：正确处理中断优先级和临界区
7. **功耗管理**：考虑低功耗模式，使用 WFI/WFE 指令
