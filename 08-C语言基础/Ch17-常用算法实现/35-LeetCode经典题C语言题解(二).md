# LeetCode经典题C语言题解(二)——树/图/DP

## 1. 二叉树的中序遍历（LC 94）

```c
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

void inorder(int* result, int* size, struct TreeNode* root) {
    if (!root) return;
    inorder(result, size, root->left);
    result[(*size)++] = root->val;
    inorder(result, size, root->right);
}

int* inorderTraversal(struct TreeNode* root, int* returnSize) {
    int* result = (int*)malloc(100 * sizeof(int));
    *returnSize = 0;
    inorder(result, returnSize, root);
    return result;
}
```

## 2. 二叉树的最大深度（LC 104）

```c
int maxDepth(struct TreeNode* root) {
    if (!root) return 0;
    int l = maxDepth(root->left);
    int r = maxDepth(root->right);
    return (l > r ? l : r) + 1;
}
```

## 3. 翻转二叉树（LC 226）

```c
struct TreeNode* invertTree(struct TreeNode* root) {
    if (!root) return NULL;
    struct TreeNode* temp = root->left;
    root->left = invertTree(root->right);
    root->right = invertTree(temp);
    return root;
}
```

## 4. 二叉树的层序遍历（LC 102）

```c
int** levelOrder(struct TreeNode* root, int* retSize, int** retCols) {
    if (!root) { *retSize = 0; return NULL; }
    int** result = (int**)malloc(2000 * sizeof(int*));
    *retCols = (int*)malloc(2000 * sizeof(int));
    *retSize = 0;

    struct TreeNode* queue[2000];
    int front = 0, rear = 0;
    queue[rear++] = root;

    while (front < rear) {
        int levelSize = rear - front;
        result[*retSize] = (int*)malloc(levelSize * sizeof(int));
        (*retCols)[*retSize] = levelSize;
        int curRear = rear;
        for (int i = front; i < curRear; i++) {
            result[*retSize][i - front] = queue[i]->val;
            if (queue[i]->left) queue[rear++] = queue[i]->left;
            if (queue[i]->right) queue[rear++] = queue[i]->right;
        }
        front = curRear;
        (*retSize)++;
    }
    return result;
}
```

## 5. 验证二叉搜索树（LC 98）

```c
bool isValidBSTHelper(struct TreeNode* root, long minVal, long maxVal) {
    if (!root) return true;
    if (root->val <= minVal || root->val >= maxVal) return false;
    return isValidBSTHelper(root->left, minVal, root->val) &&
           isValidBSTHelper(root->right, root->val, maxVal);
}

bool isValidBST(struct TreeNode* root) {
    return isValidBSTHelper(root, LONG_MIN, LONG_MAX);
}
```

## 6. 二叉树的最近公共祖先（LC 236）

```c
struct TreeNode* lowestCommonAncestor(struct TreeNode* root,
                                       struct TreeNode* p, struct TreeNode* q) {
    if (!root || root == p || root == q) return root;
    struct TreeNode* left = lowestCommonAncestor(root->left, p, q);
    struct TreeNode* right = lowestCommonAncestor(root->right, p, q);
    if (left && right) return root;        // p和q分别在左右子树
    return left ? left : right;            // 都在同一侧
}
```

## 7. 爬楼梯（LC 70）

```c
int climbStairs(int n) {
    if (n <= 2) return n;
    int a = 1, b = 2, c;
    for (int i = 3; i <= n; i++) {
        c = a + b; a = b; b = c;
    }
    return b;
}
```

## 8. 零钱兑换（LC 322）

```c
int coinChange(int* coins, int n, int amount) {
    int* dp = (int*)malloc((amount + 1) * sizeof(int));
    dp[0] = 0;
    for (int i = 1; i <= amount; i++) {
        dp[i] = INT_MAX;
        for (int j = 0; j < n; j++) {
            if (coins[j] <= i && dp[i - coins[j]] != INT_MAX) {
                int val = dp[i - coins[j]] + 1;
                if (val < dp[i]) dp[i] = val;
            }
        }
    }
    int result = dp[amount] == INT_MAX ? -1 : dp[amount];
    free(dp);
    return result;
}
```

## 9. 最长递增子序列（LC 300）

```c
int lengthOfLIS(int* nums, int n) {
    int* tails = (int*)malloc(n * sizeof(int));
    int len = 0;
    for (int i = 0; i < n; i++) {
        int lo = 0, hi = len;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (tails[mid] < nums[i]) lo = mid + 1;
            else hi = mid;
        }
        tails[lo] = nums[i];
        if (lo == len) len++;
    }
    free(tails);
    return len;
}
```

## 10. 编辑距离（LC 72）

```c
int min3(int a, int b, int c) {
    int m = a < b ? a : b;
    return m < c ? m : c;
}

int minDistance(char* word1, char* word2) {
    int m = strlen(word1), n = strlen(word2);
    int dp[m + 1][n + 1];
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            if (word1[i-1] == word2[j-1])
                dp[i][j] = dp[i-1][j-1];
            else
                dp[i][j] = 1 + min3(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]);
    return dp[m][n];
}
```

## 11. 最长公共子序列（LC 1143）

```c
int longestCommonSubsequence(char* text1, char* text2) {
    int m = strlen(text1), n = strlen(text2);
    int dp[m + 1][n + 1];
    for (int i = 0; i <= m; i++) dp[i][0] = 0;
    for (int j = 0; j <= n; j++) dp[0][j] = 0;
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            if (text1[i-1] == text2[j-1])
                dp[i][j] = dp[i-1][j-1] + 1;
            else
                dp[i][j] = dp[i-1][j] > dp[i][j-1] ? dp[i-1][j] : dp[i][j-1];
    return dp[m][n];
}
```

## 12. 岛屿数量（LC 200）

```c
void dfsIsland(char** grid, int n, int m, int i, int j) {
    if (i < 0 || i >= n || j < 0 || j >= m || grid[i][j] != '1') return;
    grid[i][j] = '0';  // 标记已访问
    dfsIsland(grid, n, m, i+1, j);
    dfsIsland(grid, n, m, i-1, j);
    dfsIsland(grid, n, m, i, j+1);
    dfsIsland(grid, n, m, i, j-1);
}

int numIslands(char** grid, int gridSize, int* gridColSize) {
    int count = 0;
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridColSize[i]; j++)
            if (grid[i][j] == '1') {
                count++;
                dfsIsland(grid, gridSize, gridColSize[i], i, j);
            }
    return count;
}
```

## 13. 课程表（LC 207）——拓扑排序

```c
bool canFinish(int numCourses, int** prerequisites, int pSize, int* pColSize) {
    int indegree[200005] = {0};
    int adj[200005][200], adjSize[200005] = {0};

    for (int i = 0; i < pSize; i++) {
        int a = prerequisites[i][0], b = prerequisites[i][1];
        adj[b][adjSize[b]++] = a;
        indegree[a]++;
    }

    int queue[200005], front = 0, rear = 0;
    for (int i = 0; i < numCourses; i++)
        if (indegree[i] == 0) queue[rear++] = i;

    int count = 0;
    while (front < rear) {
        int u = queue[front++];
        count++;
        for (int i = 0; i < adjSize[u]; i++)
            if (--indegree[adj[u][i]] == 0)
                queue[rear++] = adj[u][i];
    }
    return count == numCourses;
}
```

## 14. 打家劫舍（LC 198）

```c
int rob(int* nums, int n) {
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    int a = nums[0], b = (nums[0] > nums[1]) ? nums[0] : nums[1];
    for (int i = 2; i < n; i++) {
        int c = (a + nums[i] > b) ? a + nums[i] : b;
        a = b; b = c;
    }
    return b;
}
```

## 15. 单词搜索（LC 79）——回溯

```c
bool existDFS(char** board, int n, int m, char* word, int idx, int i, int j) {
    if (word[idx] == '\0') return true;
    if (i < 0 || i >= n || j < 0 || j >= m || board[i][j] != word[idx])
        return false;
    char temp = board[i][j];
    board[i][j] = '#';  // 标记已访问
    bool found = existDFS(board, n, m, word, idx+1, i+1, j)
              || existDFS(board, n, m, word, idx+1, i-1, j)
              || existDFS(board, n, m, word, idx+1, i, j+1)
              || existDFS(board, n, m, word, idx+1, i, j-1);
    board[i][j] = temp;  // 恢复
    return found;
}

bool exist(char** board, int n, int* colSize, char* word) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < colSize[i]; j++)
            if (existDFS(board, n, colSize[i], word, 0, i, j))
                return true;
    return false;
}
```
