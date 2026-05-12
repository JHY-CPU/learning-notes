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


## 粒子系统与碰撞检测

```javascript
// ========== 完整粒子系统 ==========
class Particle {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.vx = (Math.random() - 0.5) * 6;
        this.vy = (Math.random() - 0.5) * 6;
        this.size = Math.random() * 4 + 1;
        this.life = 1;
        this.decay = Math.random() * 0.02 + 0.005;
        this.color = `hsl(${Math.random() * 60 + 10}, 100%, 60%)`;
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.vy += 0.05; // 重力
        this.life -= this.decay;
    }

    draw(ctx) {
        ctx.globalAlpha = this.life;
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
    }

    get isDead() { return this.life <= 0; }
}

class ParticleSystem {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.running = false;
    }

    emit(x, y, count = 20) {
        for (let i = 0; i < count; i++) {
            this.particles.push(new Particle(x, y));
        }
    }

    update() {
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.particles = this.particles.filter(p => !p.isDead);
        for (const p of this.particles) {
            p.update();
            p.draw(this.ctx);
        }
    }

    start() {
        this.running = true;
        const loop = () => {
            if (!this.running) return;
            this.update();
            requestAnimationFrame(loop);
        };
        loop();
    }

    stop() { this.running = false; }
}

// ========== 碰撞检测 ==========
// 圆形碰撞
function circleCollision(x1, y1, r1, x2, y2, r2) {
    const dx = x2 - x1, dy = y2 - y1;
    return Math.sqrt(dx * dx + dy * dy) < r1 + r2;
}

// 矩形碰撞 (AABB)
function rectCollision(a, b) {
    return a.x < b.x + b.w && a.x + a.w > b.x &&
           a.y < b.y + b.h && a.y + a.h > b.y;
}

// 弹球反弹
function bounce(particle, width, height) {
    if (particle.x <= 0 || particle.x >= width) particle.vx *= -1;
    if (particle.y <= 0 || particle.y >= height) particle.vy *= -1;
    particle.x = Math.max(0, Math.min(width, particle.x));
    particle.y = Math.max(0, Math.min(height, particle.y));
}
```

## 缓动函数

```javascript
// ========== 缓动函数 ==========
const easing = {
    linear: t => t,
    easeInQuad: t => t * t,
    easeOutQuad: t => t * (2 - t),
    easeInOutQuad: t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
    easeInCubic: t => t * t * t,
    easeOutCubic: t => (--t) * t * t + 1,
    easeInOutCubic: t => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
    easeOutBounce: t => {
        if (t < 1/2.75) return 7.5625 * t * t;
        if (t < 2/2.75) return 7.5625 * (t -= 1.5/2.75) * t + 0.75;
        if (t < 2.5/2.75) return 7.5625 * (t -= 2.25/2.75) * t + 0.9375;
        return 7.5625 * (t -= 2.625/2.75) * t + 0.984375;
    },
    easeOutElastic: t => {
        if (t === 0 || t === 1) return t;
        return Math.pow(2, -10 * t) * Math.sin((t - 0.1) * 5 * Math.PI) + 1;
    },
};

// ========== 动画封装 ==========
function animate({ from, to, duration, easing: ease = 'linear', onUpdate, onComplete }) {
    const start = performance.now();
    const easeFn = easing[ease] || easing.linear;

    function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const value = from + (to - from) * easeFn(progress);
        onUpdate(value);
        if (progress < 1) requestAnimationFrame(tick);
        else if (onComplete) onComplete();
    }
    requestAnimationFrame(tick);
}

// 使用
animate({
    from: 0, to: 300, duration: 1000,
    easing: 'easeOutBounce',
    onUpdate: (v) => { element.style.left = v + 'px'; },
    onComplete: () => console.log('动画完成'),
});
```

<!-- Converted from: 50_Canvas动画.html -->
