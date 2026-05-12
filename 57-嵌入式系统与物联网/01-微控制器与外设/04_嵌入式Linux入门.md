# 嵌入式Linux入门


## 嵌入式Linux入门

一、嵌入式Linux概述

## 一、嵌入式Linux概述


嵌入式Linux是将Linux操作系统裁剪和定制后运行在嵌入式设备上的方案。相比裸机开发和RTOS，嵌入式Linux提供了完整的操作系统功能，包括进程管理、文件系统、网络协议栈等，适合功能复杂的嵌入式设备。


### 1.1 适用场景


| 方案 | 适用场景 | 特点 |
| --- | --- | --- |
| 裸机开发 | 极简单控制任务 | 无OS，直接操作硬件 |
| RTOS | 实时控制、多任务 | 轻量级，确定性响应 |
| 嵌入式Linux | 复杂应用、网络、GUI | 功能强大，生态丰富 |


### 1.2 常见嵌入式Linux平台


- **Raspberry Pi：**
   ARM Cortex-A系列，Debian系Linux
- **BeagleBone：**
   TI AM335x，支持PRU实时单元
- **NXP i.MX：**
   工业级ARM处理器
- **Allwinner/全志：**
   低成本消费电子方案
- **NVIDIA Jetson：**
   AI边缘计算

二、构建系统

## 二、构建系统：Buildroot与Yocto


嵌入式Linux需要交叉编译整个系统（内核、引导程序、根文件系统），手动管理非常复杂，因此需要自动化构建工具。


### 2.1 Buildroot


Buildroot是一个简单高效的嵌入式Linux构建系统，通过Makefile和Kconfig配置，可以一键编译出完整的系统镜像。

**Buildroot基本使用：**


`make menuconfig`
— 图形化配置


`make`
— 编译整个系统


生成产物：


-
`output/images/zImage`
— 内核镜像


-
`output/images/rootfs.tar`
— 根文件系统


-
`output/images/sdcard.img`
— 完整SD卡镜像

### 2.2 Yocto Project


Yocto是一个更强大但也更复杂的构建框架，使用BitBake构建引擎和Layer机制。适合需要高度定制的商业产品。


| 维度 | Buildroot | Yocto |
| --- | --- | --- |
| 复杂度 | 简单 | 复杂 |
| 学习曲线 | 低 | 陡峭 |
| 包管理 | 不支持运行时包管理 | 支持RPM/DEB/IPK |
| 定制能力 | 中等 | 非常强大 |
| 构建时间 | 较短 | 较长 |
| 适用场景 | 原型、小型项目 | 商业产品、大型项目 |

三、交叉编译

## 三、交叉编译


嵌入式设备通常使用ARM处理器，而开发在x86 PC上进行。交叉编译工具链在x86上编译出ARM可执行文件。


### 3.1 工具链组成


| 工具 | 说明 |
| --- | --- |
| arm-linux-gnueabihf-gcc | C编译器 |
| arm-linux-gnueabihf-g++ | C++编译器 |
| arm-linux-gnueabihf-ld | 链接器 |
| arm-linux-gnueabihf-objdump | 反汇编工具 |
| arm-linux-gnueabihf-strip | 去除调试符号 |

**交叉编译示例：**


`arm-linux-gnueabihf-gcc -o hello hello.c`


编译出的hello是ARM可执行文件，不能在x86 PC运行


传输到目标板：


`scp hello root@192.168.1.100:/home/root/`
四、设备树

## 四、设备树（Device Tree）


设备树是一种描述硬件配置的数据结构，让Linux内核可以在不修改源码的情况下支持不同的硬件平台。


### 4.1 设备树基本概念


- **DTS（Device Tree Source）：**
   设备树源文件，文本格式
- **DTC（Device Tree Compiler）：**
   编译器，将DTS编译为DTB
- **DTB（Device Tree Blob）：**
   二进制设备树文件，由U-Boot传给内核
- **DTSI（Device Tree Include）：**
   可被多个DTS引用的设备树片段


### 4.2 设备树语法


```
// 设备树节点示例
/dts-v1/;
/include/ "skeleton.dtsi"

/ {
    model = "My Board";
    compatible = "vendor,myboard";

    cpus {
        cpu@0 {
            compatible = "arm,cortex-a7";
            reg = <0>;
        };
    };

    memory {
        device_type = "memory";
        reg = <0x40000000 0x20000000>; // 起始地址512MB
    };

    soc {
        uart0: serial@1c28000 {
            compatible = "ns16550a";
            reg = <0x1c28000 0x400>;
            reg-shift = <2>;
            clock-frequency = <24000000>;
            status = "okay";
        };

        gpio0: gpio@1c20800 {
            compatible = "vendor,gpio";
            reg = <0x1c20800 0x400>;
            gpio-controller;
            #gpio-cells = <2>;
        };
    };
};
```


### 4.3 常用属性


| 属性 | 说明 | 示例 |
| --- | --- | --- |
| compatible | 驱动匹配依据 | "ti,omap3-uart" |
| reg | 寄存器地址和长度 | <0x1c28000 0x400> |
| #address-cells / #size-cells | 子节点地址格式 | <1> / <1> |
| interrupts | 中断号和触发方式 | <0 32 4> |
| status | 设备状态 | "okay" / "disabled" |
| clocks | 时钟引用 | = <&clk_uart0> |

五、U-Boot引导加载程序

## 五、U-Boot引导加载程序


U-Boot（Das U-Boot）是最常用的嵌入式Linux引导加载程序，负责初始化硬件、加载内核和设备树到内存，并启动内核。


### 5.1 启动流程


1. **ROM Boot：**
   芯片内部ROM代码执行（固化不可修改）
2. **SPL（Secondary Program Loader）：**
   加载U-Boot到SRAM
3. **U-Boot：**
   初始化DDR、加载内核和DTB到RAM
4. **Linux内核：**
   解压缩、初始化、挂载根文件系统
5. **用户空间：**
   执行init进程，启动系统服务


### 5.2 U-Boot常用命令


| 命令 | 说明 |
| --- | --- |
| `printenv` | 打印环境变量 |
| `setenv bootargs "console=ttyS0 ..."` | 设置内核启动参数 |
| `tftp 0x40000000 zImage` | 通过TFTP下载内核 |
| `fatload mmc 0:1 0x40000000 zImage` | 从SD卡加载内核 |
| `bootz 0x40000000 - 0x42000000` | 启动内核（内核地址 - DTB地址） |
| `mmc list / mmc info` | 查看SD卡信息 |
| `dhcp` | 通过DHCP获取IP地址 |


> **Note:** **典型bootcmd：**
>
>
> `setenv bootcmd "fatload mmc 0:1 0x40000000 zImage; fatload mmc 0:1 0x42000000 board.dtb; bootz 0x40000000 - 0x42000000"`
>
>
> 这条命令从SD卡第一个分区加载内核和设备树到内存，然后启动内核。

六、内核模块

## 六、Linux内核模块


内核模块是可以动态加载到运行中的Linux内核的代码片段，用于扩展内核功能而无需重新编译内核。驱动程序通常以内核模块形式提供。


### 6.1 内核模块编程基础


```
// hello_module.c - 最简单的内核模块
#include <linux/init.h>
#include <linux/module.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Developer");
MODULE_DESCRIPTION("Hello Kernel Module");

// 模块加载函数
static int __init hello_init(void) {
    printk(KERN_INFO "Hello, Kernel!\n");
    return 0; // 返回0表示成功
}

// 模块卸载函数
static void __exit hello_exit(void) {
    printk(KERN_INFO "Goodbye, Kernel!\n");
}

module_init(hello_init);
module_exit(hello_exit);
```


### 6.2 模块操作命令


| 命令 | 说明 |
| --- | --- |
| `insmod hello.ko` | 加载模块（不处理依赖） |
| `modprobe hello` | 加载模块（自动处理依赖） |
| `rmmod hello` | 卸载模块 |
| `lsmod` | 列出已加载模块 |
| `modinfo hello.ko` | 查看模块信息 |
| `dmesg` | 查看内核日志（printk输出） |


### 6.3 Makefile（内核模块编译）


```
# 内核模块编译Makefile
obj-m += hello_module.o

# 指定内核源码路径和目标架构
KDIR := /path/to/kernel/source
ARCH := arm
CROSS_COMPILE := arm-linux-gnueabihf-

all:
    make -C $(KDIR) M=$(PWD) modules ARCH=$(ARCH) \
        CROSS_COMPILE=$(CROSS_COMPILE)

clean:
    make -C $(KDIR) M=$(PWD) clean
```

========================================
  文件总结
========================================
  主题：嵌入式Linux入门
  内容概要：
    1. 构建系统 - Buildroot（简单）vs Yocto（强大复杂）
    2. 交叉编译 - ARM工具链，在x86上编译ARM程序
    3. 设备树 - DTS描述硬件，DTB传给内核
    4. U-Boot - ROM→SPL→U-Boot→Kernel→用户空间启动流程
    5. 内核模块 - 动态加载驱动，insmod/modprobe/rmmod
  重点知识：
    - Buildroot适合小型项目，Yocto适合商业产品
    - 设备树compatible属性用于驱动匹配
    - U-Boot通过bootcmd环境变量配置启动流程
    - 内核模块两个关键函数：module_init和module_exit
    - printk输出到内核日志，用dmesg查看
========================================


<!-- Converted from: 04_嵌入式Linux入门.html -->
