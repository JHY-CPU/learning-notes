## Chessboard Cover


```javascript
用L形骨牌覆盖棋盘，每次将棋盘分为四个子棋盘递归求解。```


```
// 棋盘覆盖分治
function chessboardCover(n) {
  // n 为棋盘大小 (2^k × 2^k)
  if (n === 1) return 0; // 1×1 不需要骨牌
  return 1 + 4 * chessboardCover(n/2); // 中间放1个L形骨牌 + 4个子棋盘
}
console.log('棋盘大小 | 骨牌数');
for (let k = 1; k <= 5; k++) {
  const size = Math.pow(2, k);
  console.log(`${size}×${size}  | ${chessboardCover(size)}`);
}```


  点击按钮查看结果
