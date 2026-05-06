# 32 - 字典树（Trie）

## 概述

Trie树（字典树/前缀树）是一种基于前缀的树形数据结构，主要用于字符串的高效存储和检索，常用于自动补全、拼写检查等场景。

---

## 1. 基本概念

- 每个节点存储一个字符
- 从根到某个节点的路径组成一个前缀
- 叶子节点或标记节点表示一个完整的单词
- 兄弟节点之间用指针（数组或链表）连接

```
       root
      / | \
     a  b  c
     |  |  |
     p  a  a
     |  |  |
     p  t  t
     *  *  *
   (apple) (bat) (cat)
```

---

## 2. 基本实现（固定字母表）

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define ALPHABET_SIZE 26

typedef struct TrieNode {
    struct TrieNode *children[ALPHABET_SIZE];
    bool is_end;       // 是否为某个单词的结尾
    int count;         // 经过该节点的单词数
    int prefix_count;  // 以该节点为前缀的单词数
} TrieNode;

// 创建新节点
TrieNode *trie_create_node(void) {
    TrieNode *node = (TrieNode *)calloc(1, sizeof(TrieNode));
    node->is_end = false;
    node->count = 0;
    node->prefix_count = 0;
    return node;
}

// 插入单词
void trie_insert(TrieNode *root, const char *word) {
    TrieNode *curr = root;
    for (int i = 0; word[i]; i++) {
        int idx = word[i] - 'a';
        if (curr->children[idx] == NULL) {
            curr->children[idx] = trie_create_node();
        }
        curr = curr->children[idx];
        curr->prefix_count++;
    }
    curr->is_end = true;
    curr->count++;
}

// 查找完整单词
bool trie_search(TrieNode *root, const char *word) {
    TrieNode *curr = root;
    for (int i = 0; word[i]; i++) {
        int idx = word[i] - 'a';
        if (curr->children[idx] == NULL)
            return false;
        curr = curr->children[idx];
    }
    return curr->is_end;
}

// 查找前缀是否存在
bool trie_starts_with(TrieNode *root, const char *prefix) {
    TrieNode *curr = root;
    for (int i = 0; prefix[i]; i++) {
        int idx = prefix[i] - 'a';
        if (curr->children[idx] == NULL)
            return false;
        curr = curr->children[idx];
    }
    return true;
}

// 统计以prefix为前缀的单词数
int trie_count_prefix(TrieNode *root, const char *prefix) {
    TrieNode *curr = root;
    for (int i = 0; prefix[i]; i++) {
        int idx = prefix[i] - 'a';
        if (curr->children[idx] == NULL)
            return 0;
        curr = curr->children[idx];
    }
    return curr->prefix_count;
}

// 删除单词
bool trie_delete(TrieNode *root, const char *word, int depth) {
    if (root == NULL) return false;

    if (word[depth] == '\0') {
        if (!root->is_end) return false;
        root->is_end = false;
        root->count = 0;
        // 如果没有子节点，可以删除此节点
        for (int i = 0; i < ALPHABET_SIZE; i++) {
            if (root->children[i] != NULL)
                return false;
        }
        return true;  // 可以删除
    }

    int idx = word[depth] - 'a';
    if (trie_delete(root->children[idx], word, depth + 1)) {
        free(root->children[idx]);
        root->children[idx] = NULL;
        root->prefix_count--;
        // 当前节点也可以删除
        if (!root->is_end) {
            for (int i = 0; i < ALPHABET_SIZE; i++) {
                if (root->children[i] != NULL)
                    return false;
            }
            return true;
        }
    }

    root->prefix_count--;
    return false;
}

// 释放Trie
void trie_free(TrieNode *root) {
    if (root == NULL) return;
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        trie_free(root->children[i]);
    }
    free(root);
}
```

---

## 3. 自动补全功能

```c
// 收集所有以prefix为前缀的单词
void trie_autocomplete(TrieNode *root, const char *prefix,
                       char buffer[], int buf_len, int depth) {
    if (root == NULL) return;

    if (root->is_end) {
        buffer[depth] = '\0';
        printf("  %s%s\n", prefix, buffer);
    }

    for (int i = 0; i < ALPHABET_SIZE; i++) {
        if (root->children[i]) {
            buffer[depth] = 'a' + i;
            trie_autocomplete(root->children[i], prefix, buffer,
                            buf_len, depth + 1);
        }
    }
}

// 自动补全入口
void trie_suggest(TrieNode *root, const char *prefix) {
    TrieNode *curr = root;
    // 找到前缀对应的节点
    for (int i = 0; prefix[i]; i++) {
        int idx = prefix[i] - 'a';
        if (curr->children[idx] == NULL) {
            printf("无匹配项\n");
            return;
        }
        curr = curr->children[idx];
    }

    char buffer[256];
    strcpy(buffer, prefix);
    printf("以 \"%s\" 开头的单词:\n", prefix);
    trie_autocomplete(curr, prefix, buffer, 256, 0);
}
```

---

## 4. 测试代码

```c
int main(void) {
    TrieNode *root = trie_create_node();

    // 插入单词
    trie_insert(root, "apple");
    trie_insert(root, "app");
    trie_insert(root, "application");
    trie_insert(root, "banana");
    trie_insert(root, "band");
    trie_insert(root, "cat");

    // 查找
    printf("搜索 'apple': %s\n", trie_search(root, "apple") ? "找到" : "未找到");
    printf("搜索 'app': %s\n", trie_search(root, "app") ? "找到" : "未找到");
    printf("搜索 'ap': %s\n", trie_search(root, "ap") ? "找到" : "未找到");

    // 前缀匹配
    printf("前缀 'app' 存在: %s\n", trie_starts_with(root, "app") ? "是" : "否");
    printf("前缀 'ap' 存在: %s\n", trie_starts_with(root, "ap") ? "是" : "否");
    printf("前缀 'xyz' 存在: %s\n", trie_starts_with(root, "xyz") ? "是" : "否");

    // 统计前缀
    printf("以 'app' 为前缀的单词数: %d\n", trie_count_prefix(root, "app"));
    printf("以 'ban' 为前缀的单词数: %d\n", trie_count_prefix(root, "ban"));

    // 自动补全
    printf("\n自动补全 'app':\n");
    trie_suggest(root, "app");

    // 删除
    trie_delete(root, "app", 0);
    printf("\n删除 'app' 后搜索: %s\n",
           trie_search(root, "app") ? "找到" : "未找到");

    trie_free(root);
    return 0;
}
```

---

## 5. 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 插入 | O(m) | O(m) |
| 查找 | O(m) | O(1) |
| 前缀查找 | O(m) | O(1) |
| 删除 | O(m) | O(1) |

其中 m 为单词长度。空间复杂度在最坏情况下为 O(ALPHABET_SIZE * N * M)。

---

## 要点总结

1. Trie树以空间换时间，适合大量字符串的前缀操作
2. 主要应用场景：自动补全、拼写检查、IP路由表
3. 空间开销较大，可考虑压缩Trie（如Radix Tree/Trie）
4. 对于小数据集，哈希表可能更高效
