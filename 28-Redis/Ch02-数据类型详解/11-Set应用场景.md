# Set应用场景

## 一、概念说明

Set的无序不重复特性和集合运算能力，使其在标签系统、共同好友、抽奖系统、去重统计等场景中非常实用。

## 二、核心应用场景

### 标签系统

```bash
# 用户标签
SADD user:1001:tags "Java" "Redis" "后端" "微服务"
SADD user:1002:tags "Python" "Redis" "数据分析"

# 共同标签
SINTER user:1001:tags user:1002:tags
# 输出: 1) "Redis"

# 用户兴趣并集（推荐内容）
SUNION user:1001:tags user:1002:tags

# 内容标签匹配
SADD article:1001:tags "Java" "Spring" "Redis"
SINTER user:1001:tags article:1001:tags
# 输出: 1) "Java" 2) "Redis"（匹配的标签）
```

### 共同好友

```bash
# 好友关系
SADD user:1001:friends "user2" "user3" "user4" "user5"
SADD user:1002:friends "user3" "user4" "user6"

# 共同好友
SINTER user:1001:friends user:1002:friends
# 输出: 1) "user3" 2) "user4"

# 推荐好友
SDIFF user:1002:friends user:1001:friends
# 输出: 1) "user6"（你可能认识的人）

# 好友数量
SCARD user:1001:friends
# 输出: (integer) 4
```

### 抽奖系统

```bash
# 添加参与者
SADD lottery:202401 "user1" "user2" "user3" "user4" "user5"
SADD lottery:202401 "user6" "user7" "user8"

# 随机抽取3名中奖者
SRANDMEMBER lottery:202401 3
# 输出: 1) "user2" 2) "user5" 3) "user7"

# 抽取并移除（不重复中奖）
SPOP lottery:202401 3
# 输出: 1) "user3" 2) "user1" 3) "user8"

# 查看剩余参与者
SCARD lottery:202401
# 输出: (integer) 5
```

### 去重统计

```bash
# 独立IP统计
SADD stats:202401:ips "192.168.1.1" "192.168.1.2" "192.168.1.3"
SCARD stats:202401:ips
# 输出: 独立IP数

# 去重用户统计
SADD online:users "user1" "user2" "user3"
SCARD online:users
# 输出: 在线用户数

# 活跃用户统计（天级）
SADD active:202401:day1 "user1" "user2"
SADD active:202401:day2 "user2" "user3"
SUNION active:202401:day1 active:202401:day2
SCARD # 输出: 3（两天的活跃用户数）
```

### 权限管理

```bash
# 用户权限
SADD user:1001:permissions "read" "write" "delete"
SADD user:1002:permissions "read"

# 检查权限
SISMEMBER user:1001:permissions "admin"
# 输出: (integer) 0（无管理员权限）

SISMEMBER user:1001:permissions "write"
# 输出: (integer) 1（有写权限）
```

## 三、注意事项与常见陷阱

1. **SRANDMEMBER不删除**：只是获取，不从集合移除
2. **SPOP不可恢复**：弹出后元素被删除
3. **大集合遍历**：SMEMBERS大集合阻塞，使用SSCAN
4. **集合运算是O(N*M)**：大集合运算注意性能
5. **内存考虑**：大量小Set可能比大Set更省内存
