# FPGA设计流程

## FPGA 设计流程概述

FPGA（Field Programmable Gate Array，现场可编程门阵列）设计流程是从硬件描述语言代码到实际硬件配置的完整过程。整个流程包括多个阶段，每个阶段都有对应的工具和方法。

FPGA 设计流程的主要步骤：

```
HDL代码编写 → 综合(Synthesis) → 实现(Implementation) → 生成比特流(Bitstream) → 下载配置
                     |                    |
                约束文件(SDC)        布局布线(P&R)
```

## 综合（Synthesis）

综合是将 HDL 代码转换为门级网表的过程。综合工具（如 Vivado Synthesis、Quartus Compiler）将 RTL 代码映射为目标 FPGA 器件中的基本逻辑单元（LUT、FF、BRAM、DSP 等）。

综合阶段的关键操作：
- **逻辑优化**：化简布尔表达式，消除冗余逻辑
- **资源共享**：复用算术运算单元
- **状态机编码**：将状态编码映射为触发器
- **时钟推断**：识别时钟信号和时钟使能

```verilog
// 综合工具会根据代码推断硬件结构
// 推断出加法器
assign sum = a + b;

// 推断出乘法器（可能映射到 DSP）
assign product = a * b;

// 推断出块 RAM
reg [7:0] mem [0:1023];
always @(posedge clk) begin
    if (we)
        mem[addr] <= wdata;
    rdata <= mem[addr];
end

// 推断出移位寄存器（可能映射到 SRL）
always @(posedge clk) begin
    shift_reg <= {shift_reg[6:0], serial_in};
end
```

综合约束文件（SDC/XDC）用于指导综合工具：

```tcl
# Xilinx XDC 约束文件示例

# 时钟约束：定义100MHz主时钟
create_clock -period 10.000 -name sys_clk [get_ports clk]

# 生成时钟（由PLL/MMCM产生）
create_generated_clock -name clk_50m \
    -source [get_pins u_pll/CLKIN] \
    -divide_by 2 [get_pins u_pll/CLKOUT]

# 输入延迟约束
set_input_delay -clock sys_clk -max 2.0 [get_ports data_in[*]]
set_input_delay -clock sys_clk -min 0.5 [get_ports data_in[*]]

# 输出延迟约束
set_output_delay -clock sys_clk -max 3.0 [get_ports data_out[*]]
set_output_delay -clock sys_clk -min 1.0 [get_ports data_out[*]]

# 多周期路径约束
set_multicycle_path -setup 4 -from [get_pins u_slow_reg/C] \
    -to [get_pins u_slow_out_reg/D]
set_multicycle_path -hold 3 -from [get_pins u_slow_reg/C] \
    -to [get_pins u_slow_out_reg/D]

# 伪路径约束（不需要时序分析的路径）
set_false_path -from [get_ports rst_n]
set_false_path -from [get_clocks clk_100m] -to [get_clocks clk_50m]
```

## 布局布线（Place & Route）

布局布线是将综合后的网表映射到 FPGA 物理资源上的过程：

1. **翻译（Translate）**：合并多个网表和约束
2. **映射（Map）**：将逻辑映射到具体 FPGA 原语（CLB、IOB 等）
3. **布局（Place）**：确定每个逻辑单元在芯片上的物理位置
4. **布线（Route）**：通过可编程互连资源连接各个逻辑单元

布局布线的优化目标：
- 满足所有时序约束
- 合理利用芯片资源（不超过利用率上限）
- 优化功耗和信号完整性

## 时序约束与时序分析

时序约束是指导 FPGA 实现工具进行优化的关键输入。正确的时序约束确保设计在目标时钟频率下可靠运行。

```tcl
# 完整的时序约束示例（XDC格式）

# === 时钟定义 ===
create_clock -period 10.000 -name clk_in [get_ports sys_clk_p]
set_clock_uncertainty -setup 0.500 [get_clocks clk_in]
set_clock_uncertainty -hold  0.250 [get_clocks clk_in]

# === 跨时钟域约束 ===
set_clock_groups -asynchronous \
    -group [get_clocks clk_100m] \
    -group [get_clocks clk_25m]

# === I/O 约束 ===
set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]
set_property PACKAGE_PIN  F5    [get_ports {led[0]}]
set_property DRIVE 8             [get_ports {led[*]}]
set_property SLEW SLOW           [get_ports {led[*]}]

# === 时序例外 ===
# 异步复位不需要时序分析
set_false_path -from [get_ports rst_n]

# 多周期路径：使能信号每4个时钟周期有效一次
set_multicycle_path -setup 4 -from [get_pins u_ce_div/q_reg[*]/C] \
    -to [get_pins u_data_reg[*]/D]
set_multicycle_path -hold 3 -from [get_pins u_ce_div/q_reg[*]/C] \
    -to [get_pins u_data_reg[*]/D]
```

## Vivado 与 Quartus 工具链

### Xilinx Vivado

Vivado 是 Xilinx（现 AMD）的 FPGA 开发套件，支持 7 系列及以后的器件。

```tcl
# Vivado Tcl 脚本示例：自动化构建流程
# 创建工程
create_project my_project ./my_project -part xc7z020clg400-1

# 添加源文件
add_files -norecurse {./src/top.v ./src/alu.v ./src/fsm.v}
add_files -fileset sim_1 -norecurse {./sim/tb_top.v}

# 添加约束文件
add_files -fileset constrs_1 -norecurse {./xdc/top.xdc}

# 设置顶层模块
set_property top top [current_fileset]
set_property top tb_top [get_filesets sim_1]

# 运行综合
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# 运行实现
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# 生成比特流
write_bitstream -force ./output/my_project.bit

# 导出硬件（用于 Zynq 开发）
write_hw_platform -fixed -include_bit -force ./output/my_project.xsa
```

### Intel Quartus

Quartus 是 Intel（原 Altera）的 FPGA 开发工具，支持 Cyclone、Arria、Stratix 等系列。

```tcl
# Quartus Tcl 脚本示例
# 创建工程
project_new -overwrite my_project -revision my_project

# 设置器件
set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CSEBA6U23I7
set_global_assignment -name TOP_LEVEL_ENTITY top

# 添加源文件
set_global_assignment -name VERILOG_FILE ./src/top.v
set_global_assignment -name VERILOG_FILE ./src/alu.v

# 时序约束
set_global_assignment -name SDC_FILE ./sdc/top.sdc

# 编译
load_package flow
execute_flow -compile

# 生成编程文件
execute_module -tool asm
```

设计流程注意事项：
1. 先进行行为仿真验证逻辑正确性，再进行综合
2. 约束文件应尽早编写，避免后期时序收敛困难
3. 关注综合报告中的警告（warnings），很多 warning 暗示代码存在问题
4. 资源利用率不宜超过 85%，留有余量给布线和时序优化
