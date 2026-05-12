# Verilog测试平台Testbench

## Testbench 基础结构

Testbench 是用于验证硬件设计正确性的测试平台，它不属于可综合的硬件电路，仅用于仿真。Testbench 的主要功能包括：生成激励信号、实例化被测设计（DUT）、监控输出结果并与预期值比较。

Testbench 的基本结构如下：

```verilog
`timescale 1ns / 1ps   // 时间单位/精度

module tb_example;
    // 1. 信号声明
    reg        clk;
    reg        rst_n;
    reg  [7:0] din;
    wire [7:0] dout;

    // 2. 实例化被测设计（DUT）
    dut_module u_dut (
        .clk   (clk),
        .rst_n (rst_n),
        .din   (din),
        .dout  (dout)
    );

    // 3. 时钟生成
    initial clk = 0;
    always #5 clk = ~clk;  // 10ns 周期，100MHz

    // 4. 复位与激励生成
    initial begin
        rst_n = 0;
        din   = 8'h00;
        #20 rst_n = 1;     // 20ns 后释放复位
        #10 din = 8'hA5;
        #10 din = 8'h3C;
        #100 $finish;       // 结束仿真
    end

    // 5. 结果监控
    initial begin
        $monitor("Time=%0t rst_n=%b din=%h dout=%h",
                 $time, rst_n, din, dout);
    end
endmodule
```

## initial 与 always 块

`initial` 块在仿真开始时执行一次，常用于初始化信号和生成一次性激励。`always` 块反复执行，用于生成时钟等周期性信号。

```verilog
module tb_timing;
    reg clk;
    reg rst_n;
    reg [3:0] counter;

    // 时钟生成：50MHz（20ns 周期）
    initial clk = 0;
    always #10 clk = ~clk;

    // 复位序列
    initial begin
        rst_n   = 0;
        counter = 0;
        #25 rst_n = 1;     // 异步复位
    end

    // 计数器激励
    initial begin
        wait(rst_n == 1);  // 等待复位释放
        repeat (20) begin
            @(posedge clk); // 等待时钟上升沿
            counter = counter + 1;
        end
        $display("Final count: %d", counter);
        $finish;
    end
endmodule
```

## 显示与监控任务

Verilog 提供了多种系统任务用于仿真调试：

| 系统任务 | 功能 | 使用场景 |
|---------|------|---------|
| `$display` | 立即打印一行信息 | 条件触发时的调试输出 |
| `$write` | 打印但不换行 | 格式化输出 |
| `$monitor` | 持续监控信号变化 | 自动跟踪关键信号 |
| `$strobe` | 在时间步结束时打印 | 避免竞争冒险 |
| `$time` | 返回当前仿真时间 | 时间戳记录 |
| `$random` | 生成随机数 | 随机激励生成 |

```verilog
module tb_display;
    reg [7:0] a, b;
    reg [3:0] sel;

    initial begin
        a = 8'd100;
        b = 8'd200;

        // $display 立即输出
        $display("a = %d, b = %d", a, b);
        $display("a + b = %d (0x%0h)", a + b, a + b);

        // 十六进制、二进制格式输出
        $display("Binary: a = %b", a);
        $display("Octal:  a = %o", a);
        $display("Hex:    a = %h", a);

        // $monitor 持续监控
        $monitor("Time=%0t a=%0d b=%0d sum=%0d",
                 $time, a, b, a + b);

        // 生成随机激励
        repeat (8) begin
            sel = $random;
            #10;
        end
    end
endmodule
```

## 激励生成与测试向量

实际项目中，激励生成通常更加系统化。常见的方法包括：直接赋值、循环生成、文件读取、随机生成等。

```verilog
module tb_alu;
    reg  [7:0] a, b;
    reg  [2:0] op;
    wire [7:0] result;
    wire       zero, carry;

    alu_8bit u_alu (
        .a(a), .b(b), .op(op),
        .result(result), .zero(zero), .carry(carry)
    );

    // 测试向量数组
    reg [7:0] test_a [0:7];
    reg [7:0] test_b [0:7];
    reg [2:0] test_op [0:7];
    reg [7:0] expected [0:7];

    integer i;
    integer pass_count = 0;
    integer fail_count = 0;

    initial begin
        // 加载测试向量
        test_a[0] = 8'd10;  test_b[0] = 8'd20;  test_op[0] = 3'b000; expected[0] = 8'd30;
        test_a[1] = 8'd50;  test_b[1] = 8'd30;  test_op[1] = 3'b001; expected[1] = 8'd20;
        test_a[2] = 8'hFF;  test_b[2] = 8'h0F;  test_op[2] = 3'b010; expected[2] = 8'h0F;
        test_a[3] = 8'hF0;  test_b[3] = 8'h0F;  test_op[3] = 3'b011; expected[3] = 8'hFF;

        // 执行测试
        for (i = 0; i < 4; i = i + 1) begin
            a  = test_a[i];
            b  = test_b[i];
            op = test_op[i];
            #10;
            if (result === expected[i]) begin
                $display("[PASS] Test %0d: op=%b a=%h b=%h result=%h",
                         i, op, a, b, result);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Test %0d: op=%b a=%h b=%h exp=%h got=%h",
                         i, op, a, b, expected[i], result);
                fail_count = fail_count + 1;
            end
        end

        $display("========== Summary ==========");
        $display("PASS: %0d, FAIL: %0d", pass_count, fail_count);
        $finish;
    end
endmodule
```

良好的测试平台应包含以下要素：
1. 完整的测试向量覆盖所有功能和边界情况
2. 自动化结果检查（而非人工比对波形）
3. 清晰的通过/失败报告
4. 合理的仿真时间控制
