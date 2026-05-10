# Canvas动画


## Canvas 动画


rAF 循环、粒子系统、运动轨迹、碰撞检测、性能优化。


## Canvas 动画模式


```
// ========== requestAnimationFrame 循环 ==========
function animate() {
    ctx.clearRect(0, 0, width, height);
    // 更新状态
    // 绘制
    requestAnimationFrame(animate);
}
animate();

// ========== FPS 控制 ==========
let lastTime = 0;
const FPS = 60;
const interval = 1000 / FPS;

function animate(time) {
    const delta = time - lastTime;
    if (delta >= interval) {
        // 更新和绘制
        lastTime = time - (delta % interval);
    }
    requestAnimationFrame(animate);
}

// ========== 动画对象 ==========
class Particle {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.vx = (Math.random() - 0.5) * 4;
        this.vy = (Math.random() - 0.5) * 4;
        this.size = Math.random() * 5 + 2;
        this.color = `hsl(${Math.random() * 360}, 80%, 60%)`;
    }
    update() { this.x += this.vx; this.y += this.vy; }
    draw(ctx) { ctx.fillRect(this.x, this.y, this.size, this.size); }
}
```


## 演示：Canvas 动画

点击按钮查看


<!-- Converted from: 50_Canvas动画.html -->
