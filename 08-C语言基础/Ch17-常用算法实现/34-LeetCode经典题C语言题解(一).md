# LeetCode经典题C语言题解(一)——数组/链表/字符串

## 1. 两数之和（LC 1）

给定数组和目标值，返回两数下标。

```c
// 哈希表解法 O(n)
// C语言中手动实现简单哈希
#define HSIZE 20007
typedef struct { int key, val; } Pair;
Pair hashTable[HSIZE];

int hashFunc(int key) { return ((key % HSIZE) + HSIZE) % HSIZE; }

int* twoSum(int* nums, int n, int target, int* returnSize) {
    memset(hashTable, -1, sizeof(hashTable));
    int* result = (int*)malloc(2 * sizeof(int));
    *returnSize = 2;
    for (int i = 0; i < n; i++) {
        int complement = target - nums[i];
        int h = hashFunc(complement);
        // 查找complement
        int found = -1;
        for (int j = 0; j < HSIZE; j++) {
            int idx = (h + j) % HSIZE;
            if (hashTable[idx].key == -1) break;
            if (hashTable[idx].key == complement) { found = hashTable[idx].val; break; }
        }
        if (found != -1) { result[0] = found; result[1] = i; return result; }
        // 插入
        h = hashFunc(nums[i]);
        for (int j = 0; j < HSIZE; j++) {
            int idx = (h + j) % HSIZE;
            if (hashTable[idx].key == -1) {
                hashTable[idx].key = nums[i]; hashTable[idx].val = i; break;
            }
        }
    }
    return result;
}
```

## 2. 合并两个有序链表（LC 21）

```c
struct ListNode {
    int val;
    struct ListNode *next;
};

struct ListNode* mergeTwoLists(struct ListNode* l1, struct ListNode* l2) {
    struct ListNode dummy;
    struct ListNode *tail = &dummy;
    dummy.next = NULL;
    while (l1 && l2) {
        if (l1->val <= l2->val) {
            tail->next = l1; l1 = l1->next;
        } else {
            tail->next = l2; l2 = l2->next;
        }
        tail = tail->next;
    }
    tail->next = l1 ? l1 : l2;
    return dummy.next;
}
```

## 3. 反转链表（LC 206）

```c
struct ListNode* reverseList(struct ListNode* head) {
    struct ListNode *prev = NULL, *curr = head, *next;
    while (curr) {
        next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}
```

## 4. 有效的括号（LC 20）

```c
bool isValid(char* s) {
    int n = strlen(s);
    char stack[n];
    int top = -1;
    for (int i = 0; s[i]; i++) {
        if (s[i] == '(' || s[i] == '[' || s[i] == '{') {
            stack[++top] = s[i];
        } else {
            if (top < 0) return false;
            char c = stack[top--];
            if ((s[i] == ')' && c != '(') ||
                (s[i] == ']' && c != '[') ||
                (s[i] == '}' && c != '{'))
                return false;
        }
    }
    return top == -1;
}
```

## 5. 最大子数组和（LC 53）

```c
// Kadane算法
int maxSubArray(int* nums, int n) {
    int maxSum = nums[0], curSum = nums[0];
    for (int i = 1; i < n; i++) {
        curSum = (curSum + nums[i] > nums[i]) ? curSum + nums[i] : nums[i];
        if (curSum > maxSum) maxSum = curSum;
    }
    return maxSum;
}
```

## 6. 合并区间（LC 56）

```c
int cmpInterval(const void* a, const void* b) {
    return (*(int**)a)[0] - (*(int**)b)[0];
}

int** merge(int** intervals, int n, int* cols, int* retSize, int** retCols) {
    qsort(intervals, n, sizeof(int*), cmpInterval);
    int** result = (int**)malloc(n * sizeof(int*));
    *retCols = (int*)malloc(n * sizeof(int));
    int count = 0;
    result[count] = (int*)malloc(2 * sizeof(int));
    result[count][0] = intervals[0][0];
    result[count][1] = intervals[0][1];
    (*retCols)[count] = 2;

    for (int i = 1; i < n; i++) {
        if (intervals[i][0] <= result[count][1]) {
            if (intervals[i][1] > result[count][1])
                result[count][1] = intervals[i][1];
        } else {
            count++;
            result[count] = (int*)malloc(2 * sizeof(int));
            result[count][0] = intervals[i][0];
            result[count][1] = intervals[i][1];
            (*retCols)[count] = 2;
        }
    }
    *retSize = count + 1;
    return result;
}
```

## 7. 两数相加（LC 2）——链表版

```c
struct ListNode* addTwoNumbers(struct ListNode* l1, struct ListNode* l2) {
    struct ListNode dummy;
    struct ListNode *curr = &dummy;
    int carry = 0;
    while (l1 || l2 || carry) {
        int sum = carry;
        if (l1) { sum += l1->val; l1 = l1->next; }
        if (l2) { sum += l2->val; l2 = l2->next; }
        carry = sum / 10;
        curr->next = (struct ListNode*)malloc(sizeof(struct ListNode));
        curr = curr->next;
        curr->val = sum % 10;
        curr->next = NULL;
    }
    return dummy.next;
}
```

## 8. 无重复字符的最长子串（LC 3）

```c
int lengthOfLongestSubstring(char* s) {
    int last[256];
    memset(last, -1, sizeof(last));
    int maxLen = 0, start = 0;
    for (int i = 0; s[i]; i++) {
        if (last[(unsigned char)s[i]] >= start)
            start = last[(unsigned char)s[i]] + 1;
        last[(unsigned char)s[i]] = i;
        int len = i - start + 1;
        if (len > maxLen) maxLen = len;
    }
    return maxLen;
}
```

## 9. 环形链表（LC 141）

```c
bool hasCycle(struct ListNode *head) {
    struct ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}
```

## 10. 盛最多水的容器（LC 11）

```c
int maxArea(int* height, int n) {
    int left = 0, right = n - 1, maxA = 0;
    while (left < right) {
        int h = height[left] < height[right] ? height[left] : height[right];
        int area = (right - left) * h;
        if (area > maxA) maxA = area;
        if (height[left] < height[right]) left++;
        else right--;
    }
    return maxA;
}
```

## 11. 三数之和（LC 15）

```c
int cmp15(const void* a, const void* b) { return *(int*)a - *(int*)b; }

int** threeSum(int* nums, int n, int* retSize, int** retCols) {
    qsort(nums, n, sizeof(int), cmp15);
    int** result = (int**)malloc(n * n * sizeof(int*));
    *retCols = (int*)malloc(n * n * sizeof(int));
    *retSize = 0;
    for (int i = 0; i < n - 2; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;
        int l = i + 1, r = n - 1, target = -nums[i];
        while (l < r) {
            int sum = nums[l] + nums[r];
            if (sum == target) {
                result[*retSize] = (int*)malloc(3 * sizeof(int));
                result[*retSize][0] = nums[i];
                result[*retSize][1] = nums[l];
                result[*retSize][2] = nums[r];
                (*retCols)[*retSize] = 3;
                (*retSize)++;
                while (l < r && nums[l] == nums[l + 1]) l++;
                while (l < r && nums[r] == nums[r - 1]) r--;
                l++; r--;
            } else if (sum < target) l++;
            else r--;
        }
    }
    return result;
}
```

## 12. 最小栈（LC 155）

```c
#define MAXS 10000
typedef struct {
    int data[MAXS], minData[MAXS], top;
} MinStack;

MinStack* minStackCreate() {
    MinStack* s = (MinStack*)malloc(sizeof(MinStack));
    s->top = -1;
    return s;
}
void minStackPush(MinStack* s, int val) {
    s->data[++s->top] = val;
    s->minData[s->top] = (s->top == 0) ? val : (val < s->minData[s->top-1] ? val : s->minData[s->top-1]);
}
void minStackPop(MinStack* s) { s->top--; }
int minStackTop(MinStack* s) { return s->data[s->top]; }
int minStackGetMin(MinStack* s) { return s->minData[s->top]; }
```
