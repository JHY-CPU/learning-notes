# Verilog有限状态机

## FSM 基本概念

有限状态机（Finite State Machine, FSM）是数字系统设计中最重要的建模工具之一。状态机由一组有限的状态、状态之间的转移条件和输出逻辑组成。典型应用包括：通信协议控制器、序列检测器、交通灯控制器、CPU 控制单元等。

FSM 分为两种基本类型：
- **Moore 型**：输出仅取决于当前状态
- **Mealy 型**：输出取决于当前状态和当前输入

在 Verilog 中，状态机通常采用三段式写法：
1. **状态寄存器**：在时钟驱动下更新当前状态
2. **次态逻辑**：组合逻辑，根据当前状态和输入计算下一个状态
3. **输出逻辑**：根据当前状态（Moore）或当前状态+输入（Mealy）产生输出

## 状态编码方式

状态编码直接影响电路的面积、速度和功耗。常用的编码方式有：

| 编码方式 | 特点 | 适用场景 |
|---------|------|---------|
| **Binary** | 每个状态用二进制递增编码 | 状态数多，面积敏感 |
| **One-Hot** | 每个状态用一个触发器 | 状态数少，速度优先 |
| **Gray** | 相邻状态仅一位不同 | 低功耗设计 |

```verilog
// 使用 localparam 定义状态编码（Binary 编码）
module fsm_binary (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire done,
    output reg  busy
);
    // 状态定义
    localparam IDLE  = 2'd0;
    localparam WORK  = 2'd1;
    localparam FINISH = 2'd2;

    reg [1:0] current_state, next_state;

    // 第一段：状态寄存器
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            current_state <= IDLE;
        else
            current_state <= next_state;
    end

    // 第二段：次态逻辑
    always @(*) begin
        case (current_state)
            IDLE:   next_state = start ? WORK : IDLE;
            WORK:   next_state = done  ? FINISH : WORK;
            FINISH: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    // 第三段：输出逻辑（Moore 型）
    always @(*) begin
        case (current_state)
            IDLE:   busy = 1'b0;
            WORK:   busy = 1'b1;
            FINISH: busy = 1'b0;
            default: busy = 1'b0;
        endcase
    end
endmodule
```

## 交通灯控制器设计实例

交通灯控制器是一个经典的状态机设计实例，需要控制东西和南北两个方向的红、黄、绿灯按规则切换。

```verilog
module traffic_light_controller (
    input  wire       clk,
    input  wire       rst_n,
    output reg  [2:0] ns_light,   // 南北方向：R=100, Y=010, G=001
    output reg  [2:0] ew_light    // 东西方向
);
    // 状态定义
    localparam S_NS_GREEN  = 3'd0;  // 南北绿灯
    localparam S_NS_YELLOW = 3'd1;  // 南北黄灯
    localparam S_EW_GREEN  = 3'd2;  // 东西绿灯
    localparam S_EW_YELLOW = 3'd3;  // 东西黄灯

    // 灯光编码：R=100, Y=010, G=001
    localparam LIGHT_R = 3'b100;
    localparam LIGHT_Y = 3'b010;
    localparam LIGHT_G = 3'b001;

    reg [2:0] state;
    reg [5:0] timer;       // 6位计数器，最大63
    localparam GREEN_TIME  = 6'd50;  // 绿灯持续50个时钟
    localparam YELLOW_TIME = 6'd10;  // 黄灯持续10个时钟

    // 状态转移与计时
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_NS_GREEN;
            timer <= 0;
        end else begin
            case (state)
                S_NS_GREEN: begin
                    if (timer >= GREEN_TIME - 1) begin
                        state <= S_NS_YELLOW;
                        timer <= 0;
                    end else begin
                        timer <= timer + 1;
                    end
                end
                S_NS_YELLOW: begin
                    if (timer >= YELLOW_TIME - 1) begin
                        state <= S_EW_GREEN;
                        timer <= 0;
                    end else begin
                        timer <= timer + 1;
                    end
                end
                S_EW_GREEN: begin
                    if (timer >= GREEN_TIME - 1) begin
                        state <= S_EW_YELLOW;
                        timer <= 0;
                    end else begin
                        timer <= timer + 1;
                    end
                end
                S_EW_YELLOW: begin
                    if (timer >= YELLOW_TIME - 1) begin
                        state <= S_NS_GREEN;
                        timer <= 0;
                    end else begin
                        timer <= timer + 1;
                    end
                end
                default: begin
                    state <= S_NS_GREEN;
                    timer <= 0;
                end
            endcase
        end
    end

    // 输出逻辑
    always @(*) begin
        case (state)
            S_NS_GREEN:  begin ns_light = LIGHT_G; ew_light = LIGHT_R; end
            S_NS_YELLOW: begin ns_light = LIGHT_Y; ew_light = LIGHT_R; end
            S_EW_GREEN:  begin ns_light = LIGHT_R; ew_light = LIGHT_G; end
            S_EW_YELLOW: begin ns_light = LIGHT_R; ew_light = LIGHT_Y; end
            default:     begin ns_light = LIGHT_R; ew_light = LIGHT_R; end
        endcase
    end
endmodule
```

## 序列检测器

序列检测器用于在输入数据流中检测特定的比特序列，是通信系统中的常见模块。

```verilog
// 检测序列 "1101" 的 Moore 型状态机
module seq_detector_1101 (
    input  wire clk,
    input  wire rst_n,
    input  wire din,
    output reg  detected
);
    localparam S0 = 3'd0;  // 初始状态
    localparam S1 = 3'd1;  // 检测到 1
    localparam S2 = 3'd2;  // 检测到 11
    localparam S3 = 3'd3;  // 检测到 110
    localparam S4 = 3'd4;  // 检测到 1101（输出有效）

    reg [2:0] state, next_state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S0;
        else
            state <= next_state;
    end

    always @(*) begin
        case (state)
            S0: next_state = din ? S1 : S0;
            S1: next_state = din ? S2 : S0;
            S2: next_state = din ? S2 : S3;
            S3: next_state = din ? S4 : S0;
            S4: next_state = din ? S2 : S0;
            default: next_state = S0;
        endcase
    end

    always @(*) begin
        detected = (state == S4);
    end
endmodule
```

状态机设计要点：
1. 使用 `localparam` 或 `parameter` 定义状态，避免使用魔法数字
2. 三段式写法结构清晰，便于调试和维护
3. 始终包含 `default` 分支，防止状态机进入未定义状态
4. 对于关键设计，考虑添加安全状态恢复机制
