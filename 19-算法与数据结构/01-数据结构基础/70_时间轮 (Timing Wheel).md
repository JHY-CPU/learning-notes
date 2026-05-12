# Timing Wheel

### 什么是时间轮

时间轮（Timing Wheel）是一种高效的定时器数据结构，将时间分成若干个槽（slot），每个槽存放到期时间相近的任务。指针周期性转动，处理到期任务。比优先队列更适合管理大量定时任务。

### 关键特性

- **环形结构**：类似循环队列，指针转完一圈回到起点
- **时间精度**：tickInterval 决定最小精度
- **批量处理**：每次 tick 处理一个槽中的所有任务
- **多级时间轮**：高层时间轮处理长时间延迟

### 时间与空间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 添加任务 | O(1) | 计算槽位后直接放入 |
| tick | O(k) | k 为当前槽中的任务数 |
| 空间 | O(n) | n 为任务总数 |

### 适用场景 vs 替代方案

- **大量定时任务**：时间轮 O(1) vs 优先队列 O(log n)
- **网络协议**：TCP 超时重传、心跳检测
- **替代**：任务少时用 setTimeout/setInterval 即可
- **替代**：精度要求高时用层级时间轮或时间轮+堆

### 常见陷阱

- 延迟超过一轮时需要额外处理（层级时间轮或取模）
- 槽位数选择不当影响精度和性能
- 时间漂移问题：tick 间隔不准确会累积误差

```
class TimingWheel {
  constructor(slotCount=8, tickInterval=100) {
    this.slots = new Array(slotCount).fill(null).map(() => []);
    this.tickInterval = tickInterval;
    this.currentSlot = 0;
  }
  addTask(task, delay) {
    const idx = (this.currentSlot + Math.ceil(delay / this.tickInterval)) % this.slots.length;
    this.slots[idx].push(task);
  }
  tick() {
    const tasks = this.slots[this.currentSlot];
    this.slots[this.currentSlot] = [];
    this.currentSlot = (this.currentSlot + 1) % this.slots.length;
    return tasks;
  }
}
```


### 实际应用

- **Netty 框架**：HashedWheelTimer 实现高效的定时任务管理
- **Kafka**：用时间轮管理延迟消息和会话超时
- **Nginx**：用时间轮管理连接超时和缓存过期
- **游戏服务器**：管理技能冷却、Buff 持续时间等大量定时事件

  点击按钮查看结果
