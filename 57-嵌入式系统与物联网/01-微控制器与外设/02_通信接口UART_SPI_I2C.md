# 通信接口UART/SPI/I2C


# 通信接口UART/SPI/I2C

一、通信接口概述

## 一、嵌入式通信接口概述


在嵌入式系统中，微控制器需要与各种外设进行数据交换。UART、SPI和I2C是最常用的三种串行通信接口，各有其特点和适用场景。


### 1.1 三种接口总览


| 接口 | 通信方式 | 信号线数 | 速率范围 | 典型应用 |
| --- | --- | --- | --- | --- |
| UART | 异步全双工 | 2（TX/RX） | 960bps ~ 3Mbps | 调试串口、GPS、蓝牙模块 |
| SPI | 同步全双工 | 3~4（MOSI/MISO/SCK/CS） | 可达50Mbps+ | Flash、屏幕、传感器 |
| I2C | 同步半双工 | 2（SDA/SCL） | 100k~3.4Mbps | EEPROM、传感器、PMIC |

二、UART

## 二、UART（通用异步收发传输器）


UART是一种异步串行通信协议，无需时钟信号线，通过约定的波特率实现数据同步。它是最简单的点对点通信方式。


### 2.1 UART帧格式


UART数据以帧为单位传输，每帧包含以下部分：


| 位 | 名称 | 说明 |
| --- | --- | --- |
| 起始位 | Start Bit | 1位低电平，标志帧开始 |
| 数据位 | Data Bits | 5~9位数据（通常8位），低位在前 |
| 校验位 | Parity | 可选，奇校验/偶校验/无校验 |
| 停止位 | Stop Bit | 1、1.5或2位高电平，标志帧结束 |


### 2.2 波特率（Baud Rate）


波特率表示每秒传输的符号数。在UART中，每个符号对应1位数据，因此波特率等于比特率。

**波特率计算：**


波特率 = 时钟频率 / (16 × USARTDIV)


例如：系统时钟72MHz，目标波特率115200


USARTDIV = 72000000 / (16 × 115200) = 39.0625


整数部分 = 39（0x27），小数部分 = 0.0625 × 16 = 1


USART_BRR = 0x271

### 2.3 波特率容差


收发双方的时钟存在偏差，累积误差会导致采样错误。一般要求总误差不超过±2%~±3%。


| 波特率 | 常用时钟 | 实际波特率 | 误差 |
| --- | --- | --- | --- |
| 9600 | 72MHz | 9600.38 | 0.004% |
| 115200 | 72MHz | 115211.5 | 0.01% |
| 921600 | 72MHz | 923076.9 | 0.16% |


### 2.4 UART配置示例（STM32 HAL）


```
// UART初始化配置
UART_HandleTypeDef huart1;
huart1.Instance = USART1;
huart1.Init.BaudRate = 115200;           // 波特率115200
huart1.Init.WordLength = UART_WORDLENGTH_8B; // 8位数据
huart1.Init.StopBits = UART_STOPBITS_1;      // 1位停止位
huart1.Init.Parity = UART_PARITY_NONE;       // 无校验
huart1.Init.Mode = UART_MODE_TX_RX;          // 收发模式
huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE; // 无硬件流控
HAL_UART_Init(&huart1);

// 发送数据
uint8_t txData[] = "Hello UART!";
HAL_UART_Transmit(&huart1, txData, sizeof(txData), 100);

// 接收数据（阻塞方式）
uint8_t rxData;
HAL_UART_Receive(&huart1, &rxData, 1, 1000);

// 接收数据（中断方式）
HAL_UART_Receive_IT(&huart1, &rxData, 1);

// 接收完成回调
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    // 处理接收到的数据
    HAL_UART_Receive_IT(huart, &rxData, 1); // 重新开启接收
}
```


### 2.5 硬件流控


当数据速率较高或接收方处理能力有限时，可使用RTS/CTS硬件流控防止数据丢失。


- **RTS（Request To Send）：**
   接收方拉低表示可以接收数据
- **CTS（Clear To Send）：**
   发送方检测CTS，低电平时才发送
- **适用场景：**
   高速传输、蓝牙/WiFi模块通信

三、SPI

## 三、SPI（串行外设接口）


SPI是Motorola开发的同步串行通信协议，采用主从架构，支持全双工通信，速率远高于UART和I2C。


### 3.1 SPI信号线


| 信号 | 全称 | 方向 | 说明 |
| --- | --- | --- | --- |
| SCLK | Serial Clock | Master → Slave | 时钟信号，由主机产生 |
| MOSI | Master Out Slave In | Master → Slave | 主机发送数据线 |
| MISO | Master In Slave Out | Slave → Master | 从机发送数据线 |
| CS/SS | Chip Select | Master → Slave | 片选信号，低电平有效 |


### 3.2 SPI工作模式（CPOL/CPHA）


SPI有四种工作模式，由时钟极性（CPOL）和时钟相位（CPHA）决定：


| 模式 | CPOL | CPHA | 空闲时钟 | 数据采样边沿 | 数据移出边沿 |
| --- | --- | --- | --- | --- | --- |
| Mode 0 | 0 | 0 | 低电平 | 上升沿 | 下降沿 |
| Mode 1 | 0 | 1 | 低电平 | 下降沿 | 上升沿 |
| Mode 2 | 1 | 0 | 高电平 | 下降沿 | 上升沿 |
| Mode 3 | 1 | 1 | 高电平 | 上升沿 | 下降沿 |


> **Note:** **CPOL/CPHA含义：**
>
>
> CPOL = 0：时钟空闲时为低电平；CPOL = 1：时钟空闲时为高电平
>
>
> CPHA = 0：在第一个时钟边沿采样数据；CPHA = 1：在第二个时钟边沿采样数据


### 3.3 全双工通信


SPI在每个时钟周期同时进行发送和接收：主机通过MOSI发送一位数据，从机通过MISO同时发送一位数据。因此读写操作是同时进行的。

**SPI全双工特点：**


- 发送和接收共享同一个时钟信号


- 每次传输都是双向的：即使主机只想读取，也必须发送（可以是dummy数据）


- 从机在接收到主机命令后，在后续时钟周期返回数据


- 数据位宽通常为8位或16位

### 3.4 多从机连接


SPI支持多从机配置，每增加一个从机只需增加一条CS线。


- **独立片选：**
   每个从机一条CS线，最常用方式
- **菊花链：**
   从机的MISO连到下一个的MOSI，适合移位寄存器
- **注意：**
   未被选中的从机MISO应为高阻态，避免总线冲突


### 3.5 SPI配置示例（STM32 HAL）


```
// SPI初始化
SPI_HandleTypeDef hspi1;
hspi1.Instance = SPI1;
hspi1.Init.Mode = SPI_MODE_MASTER;            // 主机模式
hspi1.Init.Direction = SPI_DIRECTION_2LINES;  // 全双工
hspi1.Init.DataSize = SPI_DATASIZE_8BIT;      // 8位数据
hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;    // CPOL=0
hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;        // CPHA=0 (Mode 0)
hspi1.Init.NSS = SPI_NSS_SOFT;                // 软件管理片选
hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_4; // 4分频
hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;       // 高位在前
HAL_SPI_Init(&hspi1);

// SPI收发数据
uint8_t txData = 0x80;  // 读取寄存器命令
uint8_t rxData;
HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET); // CS拉低
HAL_SPI_Transmit(&hspi1, &txData, 1, 100);    // 发送命令
HAL_SPI_Receive(&hspi1, &rxData, 1, 100);     // 接收数据
HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);   // CS拉高

// 同时收发（全双工）
HAL_SPI_TransmitReceive(&hspi1, txBuf, rxBuf, length, 100);
```

四、I2C

## 四、I2C（内部集成电路总线）


I2C是Philips开发的两线式串行总线，只需SDA和SCL两条线即可连接多个器件，每个器件有唯一的7位或10位地址。


### 4.1 I2C信号线


| 信号 | 全称 | 说明 |
| --- | --- | --- |
| SDA | Serial Data | 双向数据线，开漏输出需外部上拉 |
| SCL | Serial Clock | 时钟线，由主机控制，开漏输出需外部上拉 |


### 4.2 I2C通信时序


1. **起始条件（START）：**
   SCL为高时，SDA从高变低
2. **发送地址：**
   7位从机地址 + 1位读写位（0写/1读）
3. **等待ACK：**
   从机在第9个时钟脉冲拉低SDA应答
4. **数据传输：**
   每次8位数据，高位在前，每字节后跟ACK/NACK
5. **停止条件（STOP）：**
   SCL为高时，SDA从低变高

**I2C帧结构：**


[START] [7位地址 + R/W] [ACK] [数据字节] [ACK] ... [STOP]


写操作：S | Addr+W | A | Data | A | Data | A | P


读操作：S | Addr+R | A | Data | A | Data | A̅ | P（最后一个NACK表示读完）

### 4.3 I2C地址机制


I2C使用7位或10位寻址。7位地址可寻址128个设备（其中16个保留），实际可用112个。


| 地址范围 | 用途 |
| --- | --- |
| 0x00 | 广播呼叫地址 |
| 0x01~0x07 | 保留 |
| 0x08~0x77 | 通用设备地址 |
| 0x78~0x7F | 10位地址模式或保留 |


### 4.4 时钟拉伸（Clock Stretching）


从机可以通过将SCL线拉低来"暂停"通信，迫使主机等待，直到从机准备好数据。


> **Note:** **时钟拉伸原理：**
>
>
> I2C总线使用开漏输出，任何设备都可以拉低SCL。当从机需要更多时间处理数据时，它将SCL保持在低电平。主机在发送下一个时钟脉冲前会检测SCL是否已释放（变高）。这个机制对主机完全透明。


### 4.5 I2C配置示例（STM32 HAL）


```
// I2C初始化
I2C_HandleTypeDef hi2c1;
hi2c1.Instance = I2C1;
hi2c1.Init.ClockSpeed = 400000;        // 400kHz 快速模式
hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2; // 标准占空比
hi2c1.Init.OwnAddress1 = 0x00;         // 本机地址
hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
HAL_I2C_Init(&hi2c1);

// 写入数据到从机（寄存器操作）
uint8_t regAddr = 0x75;  // 寄存器地址
uint8_t data = 0x80;     // 写入值
HAL_I2C_Mem_Write(&hi2c1, (0x68 << 1), regAddr,
    I2C_MEMADD_SIZE_8BIT, &data, 1, 100);

// 从从机读取数据
uint8_t rxData;
HAL_I2C_Mem_Read(&hi2c1, (0x68 << 1), regAddr,
    I2C_MEMADD_SIZE_8BIT, &rxData, 1, 100);

// 检测设备是否在线
if (HAL_I2C_IsDeviceReady(&hi2c1, (0x68 << 1), 3, 100) == HAL_OK) {
    // 设备在线
}
```

五、对比总结

## 五、三种接口对比


| 特性 | UART | SPI | I2C |
| --- | --- | --- | --- |
| 信号线数 | 2（TX/RX） | 3~4 | 2（SDA/SCL） |
| 通信方式 | 异步全双工 | 同步全双工 | 同步半双工 |
| 最大速率 | ~3Mbps | ~50Mbps+ | ~3.4Mbps（HS模式） |
| 拓扑结构 | 点对点 | 一主多从 | 多主多从 |
| 寻址方式 | 无（点对点） | 片选线 | 地址帧 |
| 硬件复杂度 | 最低 | 中等 | 低 |
| 协议开销 | 起始/停止/校验位 | 无额外开销 | 地址+ACK位 |
| 抗干扰能力 | 一般 | 好（短距离） | 一般 |
| 线缆长度 | 可达15m | 通常<30cm | 通常<1m |

**接口选择指南：**


- 需要简单点对点通信（调试、GPS） → UART


- 需要高速数据传输（Flash、屏幕） → SPI


- 需要连接多个低速设备（传感器、EEPROM） → I2C


- 距离较远 → UART（配合RS-232/RS-485）


- 引脚资源紧张 → I2C（只需2根线）
========================================
  文件总结
========================================
  主题：通信接口UART/SPI/I2C
  内容概要：
    1. UART - 异步串行通信，波特率、帧格式、硬件流控
    2. SPI - 同步全双工，CPOL/CPHA四种模式，多从机片选
    3. I2C - 两线同步半双工，7位寻址，时钟拉伸，ACK机制
    4. 接口对比 - 信号线、速率、拓扑、适用场景
  重点知识：
    - UART帧：起始位+数据位+校验位+停止位
    - SPI四种模式由CPOL/CPHA决定
    - I2C起始条件：SCL高时SDA由高变低；停止条件：SCL高时SDA由低变高
    - I2C时钟拉伸允许从机暂停通信
    - SPI全双工意味着每次传输同时收发
========================================


<!-- Converted from: 02_通信接口UART_SPI_I2C.html -->
