# ARM架构与GPIO


# ARM架构与GPIO

一、ARM架构概述

## 一、ARM Cortex-M架构概述


ARM Cortex-M系列是专为微控制器设计的处理器内核，广泛应用于嵌入式系统和物联网设备。


### 1.1 Cortex-M系列对比


| 型号 | 流水线 | 特点 | 应用场景 |
| --- | --- | --- | --- |
| Cortex-M0 | 3级 | 最低功耗、最小面积 | 简单传感器节点 |
| Cortex-M0+ | 2级 | M0的优化版 | 超低功耗IoT |
| Cortex-M3 | 3级 | 性能与功耗平衡 | 通用嵌入式 |
| Cortex-M4 | 3级 | DSP和浮点运算 | 信号处理、控制 |
| Cortex-M7 | 6级 | 最高性能 | 高端嵌入式应用 |


### 1.2 核心特性


- **Thumb/Thumb-2指令集：**
   16位/32位混合指令，节省存储空间
- **NVIC（嵌套向量中断控制器）：**
   低延迟中断响应
- **位带操作（Bit-banding）：**
   原子级别的位操作
- **SysTick定时器：**
   系统节拍定时器，支持RTOS
- **调试接口：**
   SWD/JTAG调试支持


### 1.3 存储架构


| 区域 | 地址范围 | 用途 |
| --- | --- | --- |
| Code | 0x0000_0000 - 0x1FFF_FFFF | Flash程序存储 |
| SRAM | 0x2000_0000 - 0x3FFF_FFFF | 内部SRAM |
| Peripheral | 0x4000_0000 - 0x5FFF_FFFF | 外设寄存器 |
| External RAM | 0x6000_0000 - 0x9FFF_FFFF | 外部存储 |
| PPB | 0xE000_0000 - 0xE00F_FFFF | 私有外设总线（NVIC等） |

二、GPIO

## 二、GPIO（通用输入输出）


GPIO是微控制器与外部世界交互的最基本接口，可以配置为数字输入或输出。


### 2.1 GPIO工作模式


| 模式 | 说明 | 应用场景 |
| --- | --- | --- |
| 推挽输出 | 可输出高/低电平，驱动能力强 | LED驱动、数字信号输出 |
| 开漏输出 | 只能拉低，需外部上拉 | I2C总线、电平转换 |
| 浮空输入 | 无上下拉，高阻态 | 外部已有上下拉的信号 |
| 上拉输入 | 内部上拉电阻，默认高电平 | 按键检测 |
| 下拉输入 | 内部下拉电阻，默认低电平 | 传感器输入 |
| 模拟输入 | 连接到ADC | 模拟信号采集 |
| 复用功能 | 连接到外设（UART/SPI/I2C等） | 通信接口 |


### 2.2 GPIO配置示例（STM32 HAL）


```
// 配置GPIO引脚为推挽输出
GPIO_InitTypeDef GPIO_InitStruct = {0};
GPIO_InitStruct.Pin = GPIO_PIN_5;
GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;  // 推挽输出
GPIO_InitStruct.Pull = GPIO_NOPULL;           // 无上下拉
GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW; // 低速
HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

// 控制输出
HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);   // 高电平
HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET); // 低电平
HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);                // 翻转

// 读取输入
GPIO_PinState state = HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_0);
```

三、中断

## 三、中断（Interrupts）


中断是嵌入式系统中实现异步事件响应的核心机制。


### 3.1 NVIC特性


- **嵌套中断：**
   高优先级中断可打断低优先级中断
- **可配置优先级：**
   优先级可动态调整
- **向量表：**
   每个中断源有独立的中断向量
- **低延迟：**
   12个时钟周期的中断响应


### 3.2 中断优先级


> **Note:** **优先级规则：**
> 数值越小优先级越高。优先级分为抢占优先级和子优先级。
>
>
> 优先级寄存器位数可配置（0-7位），STM32通常使用4位。

四、定时器

## 四、定时器（Timers）


| 定时器类型 | 特点 | 应用 |
| --- | --- | --- |
| SysTick | 24位倒计时，内核内置 | RTOS系统节拍 |
| 基本定时器 | 仅计数和中断 | 简单定时任务 |
| 通用定时器 | PWM、输入捕获、输出比较 | 电机控制、信号测量 |
| 高级定时器 | 互补输出、死区控制 | 三相电机驱动 |

五、ADC/DAC

## 五、ADC与DAC


### 5.1 ADC（模数转换器）


- **分辨率：**
   通常12位（0-4095）
- **采样率：**
   STM32可达数MSPS
- **转换模式：**
   单次转换、连续转换、扫描模式
- **触发源：**
   软件触发、定时器触发、外部触发


### 5.2 DAC（数模转换器）


- **分辨率：**
   通常8位或12位
- **输出模式：**
   可直接输出或通过DMA连续输出
- **应用场景：**
   音频输出、波形生成、模拟控制信号

**ADC电压计算：**


`V = ADC_value × V_ref / (2^resolution - 1)`


例如12位ADC，Vref=3.3V，ADC值=2048，则V = 2048 × 3.3 / 4095 ≈ 1.65V
========================================
  文件总结
========================================
  主题：ARM架构与GPIO
  内容概要：
    1. Cortex-M系列 - M0/M0+/M3/M4/M7，不同性能定位
    2. GPIO - 推挽/开漏输出，浮空/上拉/下拉输入，模拟输入，复用功能
    3. 中断 - NVIC嵌套中断，优先级配置，12周期响应
    4. 定时器 - SysTick/基本/通用/高级定时器
    5. ADC/DAC - 模数/数模转换，分辨率和采样率
  重点知识：
    - ARM存储映射：Code/SRAM/Peripheral/PPB
    - GPIO八种工作模式及应用场景
    - 优先级数值越小优先级越高
    - ADC电压公式：V = ADC_value × V_ref / (2^resolution - 1)
========================================


<!-- Converted from: 01_ARM架构与GPIO.html -->
