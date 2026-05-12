# 14-棋盘覆盖 (Chessboard Cover)

给定 2^k x 2^k 棋盘，一个特殊方格被占用，用 L 形骨牌（覆盖 3 格）覆盖其余方格。

## 分治思路

1. 棋盘分为 4 个 2^(k-1) x 2^(k-1) 子棋盘
2. 在中心放置 L 形骨牌，使每个子棋盘各占一个被覆盖的格
3. 递归处理 4 个子棋盘
4. 基础情况：1x1 棋盘不需要骨牌

```javascript
let tileId = 1;
function chessboardCover(board, tr, tc, dr, dc, size) {
  if (size === 1) return;
  const half = size >> 1;
  const t = tileId++; // 当前 L 形骨牌编号

  // 中心位置
  const mr = tr + half, mc = tc + half;

  // 判断特殊方格在哪个子棋盘，放置 L 形骨牌覆盖其余三个子棋盘的中心
  if (dr < mr && dc < mc) { // 左上
    chessboardCover(board, tr, tc, dr, dc, half);
  } else {
    board[mr - 1][mc - 1] = t;
    chessboardCover(board, tr, tc, mr - 1, mc - 1, half);
  }

  if (dr < mr && dc >= mc) { // 右上
    chessboardCover(board, tr, mc, dr, dc, half);
  } else {
    board[mr - 1][mc] = t;
    chessboardCover(board, tr, mc, mr - 1, mc, half);
  }

  if (dr >= mr && dc < mc) { // 左下
    chessboardCover(board, mr, tc, dr, dc, half);
  } else {
    board[mr][mc - 1] = t;
    chessboardCover(board, mr, tc, mr, mc - 1, half);
  }

  if (dr >= mr && dc >= mc) { // 右下
    chessboardCover(board, mr, mc, dr, dc, half);
  } else {
    board[mr][mc] = t;
    chessboardCover(board, mr, mc, mr, mc, half);
  }
}

// 使用
const n = 8;
const board = Array.from({length: n}, () => new Array(n).fill(0));
board[0][0] = -1; // 特殊方格
chessboardCover(board, 0, 0, 0, 0, n);
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

int tileId = 1;

void cover(vector<vector<int>>& board, int tr, int tc, int dr, int dc, int size) {
    if (size == 1) return;
    int half = size / 2;
    int t = tileId++;
    int mr = tr + half, mc = tc + half;

    if (dr < mr && dc < mc) cover(board, tr, tc, dr, dc, half);
    else { board[mr-1][mc-1] = t; cover(board, tr, tc, mr-1, mc-1, half); }

    if (dr < mr && dc >= mc) cover(board, tr, mc, dr, dc, half);
    else { board[mr-1][mc] = t; cover(board, tr, mc, mr-1, mc, half); }

    if (dr >= mr && dc < mc) cover(board, mr, tc, dr, dc, half);
    else { board[mr][mc-1] = t; cover(board, mr, tc, mr, mc-1, half); }

    if (dr >= mr && dc >= mc) cover(board, mr, mc, dr, dc, half);
    else { board[mr][mc] = t; cover(board, mr, mc, mr, mc, half); }
}
```

## 复杂度

骨牌数 = (4^k - 1) / 3。递推 T(n) = 4T(n/2) + O(1) -> O(n^2)。

| 参数 | 值 |
|------|-----|
| 时间 | O(n^2) |
| 空间 | O(log n) 递归栈 |

## 应用

- 四叉树图像分割
- 分治教学经典案例
- 空间填充曲线的前置知识
