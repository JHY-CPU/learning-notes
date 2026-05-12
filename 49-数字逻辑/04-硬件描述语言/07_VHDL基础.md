# VHDL基础

## VHDL 概述

VHDL（VHSIC Hardware Description Language）是一种强类型的硬件描述语言，由美国国防部在 1980 年代开发。与 Verilog 相比，VHDL 具有更强的类型检查、更严格的语法结构，广泛应用于航空航天、军事和工业领域。

VHDL 的设计层次与 Verilog 相同，包括行为级描述、寄存器传输级（RTL）和门级描述。VHDL 的代码结构更加严谨，需要显式声明库和包。

## Entity 与 Architecture

VHDL 设计的基本单元由 `entity`（实体）和 `architecture`（结构体）组成。`entity` 定义模块的外部接口，`architecture` 描述模块的内部实现。

```vhdl
-- 库声明
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- 实体声明：定义端口接口
entity and_gate is
    port (
        a : in  std_logic;
        b : in  std_logic;
        y : out std_logic
    );
end entity and_gate;

-- 结构体：描述内部实现
architecture rtl of and_gate is
begin
    y <= a and b;
end architecture rtl;
```

一个 entity 可以有多个 architecture，通过配置（configuration）选择使用哪个：

```vhdl
-- 行为级描述
architecture behavioral of full_adder is
begin
    process(a, b, cin)
        variable sum_v : std_logic_vector(1 downto 0);
    begin
        sum_v := ('0' & a) + ('0' & b) + ('0' & cin);
        sum   <= sum_v(0);
        cout  <= sum_v(1);
    end process;
end architecture behavioral;

-- 数据流描述
architecture dataflow of full_adder is
begin
    sum  <= a xor b xor cin;
    cout <= (a and b) or (a and cin) or (b and cin);
end architecture dataflow;
```

## Signal 与 Variable

VHDL 中有两种赋值对象：`signal`（信号）和 `variable`（变量），它们有着根本性的区别。

| 特性 | Signal | Variable |
|------|--------|----------|
| 赋值符号 | `<=` | `:=` |
| 作用域 | architecture 全局 | process/function 内部 |
| 更新时机 | 进程挂起时更新 | 立即更新 |
| 使用场景 | 硬件连线/寄存器 | 临时计算 |

```vhdl
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity signal_var_demo is
    port (
        clk   : in  std_logic;
        rst   : in  std_logic;
        din   : in  std_logic_vector(7 downto 0);
        dout  : out std_logic_vector(7 downto 0)
    );
end entity signal_var_demo;

architecture rtl of signal_var_demo is
    -- 信号声明（architecture 级别）
    signal sig_data : std_logic_vector(7 downto 0);
    signal sig_sum  : std_logic_vector(7 downto 0);
begin
    process(clk, rst)
        -- 变量声明（process 级别）
        variable var_temp : std_logic_vector(7 downto 0);
        variable var_cnt  : integer range 0 to 255;
    begin
        if rst = '1' then
            sig_data <= (others => '0');
            dout     <= (others => '0');
        elsif rising_edge(clk) then
            -- 变量立即更新
            var_temp := din;
            var_cnt  := to_integer(unsigned(var_temp));

            -- 信号在进程挂起时更新
            sig_data <= var_temp;
            sig_sum  <= std_logic_vector(to_unsigned(var_cnt + 1, 8));
            dout     <= sig_data;  -- 输出的是上一个时钟周期的 sig_data
        end if;
    end process;
end architecture rtl;
```

## Process 与并发语句

VHDL 中有两大类语句结构：
- **顺序语句**：在 `process` 内部按顺序执行，包括 `if`、`case`、`for loop`、`variable` 赋值
- **并发语句**：在 `process` 外部同时执行，包括信号赋值、`component` 实例化、`generate`

```vhdl
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity concurrent_vs_sequential is
    port (
        sel  : in  std_logic_vector(1 downto 0);
        din0 : in  std_logic_vector(7 downto 0);
        din1 : in  std_logic_vector(7 downto 0);
        din2 : in  std_logic_vector(7 downto 0);
        din3 : in  std_logic_vector(7 downto 0);
        dout : out std_logic_vector(7 downto 0)
    );
end entity concurrent_vs_sequential;

architecture rtl of concurrent_vs_sequential is
    signal a, b, c, y : std_logic;
begin
    -- ========== 并发语句（process 外部）==========
    -- 并发信号赋值
    a <= '1' when sel = "00" else '0';
    b <= din0(0) and din1(0);

    -- 并发条件信号赋值（类似三目运算符）
    with sel select
        c <= din0(0) when "00",
             din1(0) when "01",
             din2(0) when "10",
             din3(0) when others;

    -- ========== 顺序语句（process 内部）==========
    mux_proc : process(sel, din0, din1, din2, din3)
    begin
        case sel is
            when "00" =>
                dout <= din0;
            when "01" =>
                dout <= din1;
            when "10" =>
                dout <= din2;
            when "11" =>
                dout <= din3;
            when others =>
                dout <= (others => 'X');
        end case;
    end process mux_proc;

    -- 时序逻辑 process
    reg_proc : process(clk, rst)
    begin
        if rst = '1' then
            y <= '0';
        elsif rising_edge(clk) then
            y <= a or b or c;
        end if;
    end process reg_proc;
end architecture rtl;
```

## 元件实例化与层次化设计

VHDL 使用元件（component）实例化来构建层次化设计。

```vhdl
-- 底层模块：D触发器
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity d_ff is
    port (
        clk : in  std_logic;
        d   : in  std_logic;
        q   : out std_logic
    );
end entity d_ff;

architecture rtl of d_ff is
begin
    process(clk)
    begin
        if rising_edge(clk) then
            q <= d;
        end if;
    end process;
end architecture rtl;

-- 上层模块：4位移位寄存器
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity shift_reg_4 is
    port (
        clk        : in  std_logic;
        serial_in  : in  std_logic;
        serial_out : out std_logic
    );
end entity shift_reg_4;

architecture structural of shift_reg_4 is
    -- 元件声明
    component d_ff is
        port (
            clk : in  std_logic;
            d   : in  std_logic;
            q   : out std_logic
        );
    end component d_ff;

    signal chain : std_logic_vector(3 downto 0);
begin
    -- 直接实例化（VHDL-93 推荐方式）
    ff0 : d_ff port map(clk => clk, d => serial_in, q => chain(0));
    ff1 : d_ff port map(clk => clk, d => chain(0),   q => chain(1));
    ff2 : d_ff port map(clk => clk, d => chain(1),   q => chain(2));
    ff3 : d_ff port map(clk => clk, d => chain(2),   q => chain(3));

    serial_out <= chain(3);
end architecture structural;
```

VHDL 与 Verilog 的主要区别：
1. VHDL 强类型，Verilog 弱类型
2. VHDL 使用 `process` 描述行为，Verilog 使用 `always` 块
3. VHDL 的 `signal` 和 `variable` 有本质区别，Verilog 中 `wire` 和 `reg` 的区分更灵活
4. VHDL 更冗长但更严谨，Verilog 更简洁但更容易出错
