# String位操作

## 一、概念说明

Redis的String类型支持位级操作，可以对字符串的每一位进行设置、获取和统计。位操作在处理布尔值集合、用户签到等场景下非常高效，内存占用极低。

## 二、具体用法

### SETBIT与GETBIT

```bash
# 设置某一位的值（0或1）
SETBIT sign:202401 0 1
# 输出: (integer) 0（原值）

SETBIT sign:202401 1 1
SETBIT sign:202401 5 1
SETBIT sign:202401 10 1

# 获取某一位的值
GETBIT sign:202401 0
# 输出: (integer) 1

GETBIT sign:202401 2
# 输出: (integer) 0（未设置默认0）
```

### BITCOUNT统计

```bash
# 统计值为1的位数
BITCOUNT sign:202401
# 输出: (integer) 4（有4个1）

# 统计指定字节范围
BITCOUNT sign:202401 0 0
# 输出: 统计第一个字节中1的个数

# 应用：用户签到统计
# 第0位代表第1天，第1位代表第2天...
# 统计本月签到天数
BITCOUNT sign:202401
```

### BITOP位运算

```bash
# 位运算：AND、OR、XOR、NOT
SETBIT bitmap1 0 1
SETBIT bitmap1 2 1
SETBIT bitmap1 4 1

SETBIT bitmap2 0 1
SETBIT bitmap2 1 1
SETBIT bitmap2 4 1

# AND运算
BITOP AND result bitmap1 bitmap2
# result: 仅第0位和第4位为1

# OR运算
BITOP OR result bitmap1 bitmap2
# result: 第0、1、2、4位为1

# XOR运算（异或）
BITOP XOR result bitmap1 bitmap2
# result: 第1、2位为1

# NOT运算（取反）
BITOP NOT result bitmap1
```

### BITPOS查找

```bash
# 查找第一个为1或0的位
SETBIT mybits 10 1
SETBIT mybits 20 1

BITPOS mybits 1
# 输出: (integer) 10（第一个1的位置）

BITPOS mybits 0
# 输出: (integer) 0（第一个0的位置）

# 指定范围查找
BITPOS mybits 1 0 1
# 输出: 在字节0-1范围内查找
```

## 三、实际应用示例

```bash
# 用户签到系统
# 用户1001在2024年1月的签到记录
# 第i位表示第i+1天是否签到

# 第1天签到
SETBIT sign:1001:202401 0 1
# 第3天签到
SETBIT sign:1001:202401 2 1
# 第5天签到
SETBIT sign:1001:202401 4 1

# 统计签到天数
BITCOUNT sign:1001:202401
# 输出: (integer) 3

# 检查某天是否签到
GETBIT sign:1001:202401 4
# 输出: (integer) 1（第5天已签到）

# 查找连续签到
# 使用BITOP AND查找两天都签到的情况
```

## 四、注意事项与常见陷阱

1. **位操作是自动扩展的**：设置超出范围的位会自动扩展字符串
2. **内存效率极高**：100万用户签到仅需约125KB内存
3. **BITCOUNT是O(N)**：统计需要遍历整个字符串
4. **字节序问题**：Redis使用大端字节序
5. **位偏移从0开始**：SETBIT key 0 value设置第一个位
6. **默认值为0**：未设置的位默认为0
