## Timing Wheel


```javascript
时间轮是一种高效的定时器数据结构，用于管理大量定时任务。```


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
}```


  点击按钮查看结果
