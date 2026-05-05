## Monte Carlo


```javascript
蒙特卡洛方法通过随机采样近似数值结果，用于积分、概率计算等。```


```
// 蒙特卡洛估算 PI
function estimatePi(samples) {
  let inside = 0;
  for (let i = 0; i < samples; i++) {
    const x = Math.random() * 2 - 1;
    const y = Math.random() * 2 - 1;
    if (x*x + y*y <= 1) inside++;
  }
  return 4 * inside / samples;
}
console.log(`PI ≈ ${estimatePi(100000)}`);```


  点击按钮查看结果
