# Python运算符优先级与复合表达式


## 📐 Python 运算符优先级与复合表达式


完整优先级表、结合性、常见陷阱、复杂表达式拆解、括号最佳实践。


## 完整优先级表 (高 → 低)


```
// ========== Python 运算符优先级 (Top → Bottom) ==========
// 优先级	运算符	            说明
// 1 (最高)	(exp), {exp}, [exp]  括号/字典/列表
// 2	    x[i], x.attr, f()   索引/属性/调用
// 3	    await x              异步等待
// 4	    **                   幂运算 (右结合)
// 5	    +x, -x, ~x           一元运算符
// 6	    *, /, //, %          乘除取余
// 7	    +, -                 加减
// 8	    <<, >>               移位
// 9	    &                    按位与
// 10	    ^                    按位异或
// 11	    |                    按位或
// 12	    in, is, <, <=, >, >=, ==, !=  比较/身份/成员
// 13	    not x                逻辑非
// 14	    and                  逻辑与
// 15	    or                   逻辑或
// 16	    if-else              条件表达式
// 17	    :=                   海象运算符
// 18 (最低)	lambda               匿名函数

// ========== 实际例子 ==========
2 + 3 * 4                # 14 (* 优先于 +)
(2 + 3) * 4              # 20 (括号覆盖优先级)

2 ** 3 ** 2              # 512 (** 右结合: 2**(3**2))
(2 ** 3) ** 2            # 64

3 * 2 ** 3               # 24 (** 优先于 *)
not True and False       # False (not > and)
True or False and False  # True (and > or)

1 << 2 + 3               # 32 (+ 优先于 <<: 1 << 5)
(1 << 2) + 3             # 7
```


## 常见陷阱与误区


```
// ========== 陷阱 1: and / or 优先级 ==========
// and 优先于 or
True or False and False   # True → True or (False and False) → True
(True or False) and False # False (括号改变结果)

// 实际应用:
user = get_user()
result = user and user.is_admin or False
# 等价于 (user and user.is_admin) or False

// ========== 陷阱 2: == 优先级高于 not ==========
not "hello" == "world"    # True (等价于 not ("hello" == "world"))
not "hello" == "hello"    # False

// ========== 陷阱 3: + 优先于 << ==========
1 << 2 + 3                # 32 (等价于 1 << (2+3) = 1 << 5)
(1 << 2) + 3              # 7

// ========== 陷阱 4: in / == 一起用 ==========
"a" in "abc" == True      # False! ⚠️
// 等价于 ("a" in "abc") and ("abc" == True) → True and False → False

// 正确:
("a" in "abc") == True    # True
// 但更简单:
"a" in "abc"              # True (直接使用)

// ========== 陷阱 5: 链式比较的优先级 ==========
// 链式比较会被展开!
a < b < c                 # (a < b) and (b < c) (不是 a < (b < c))
```


## 复合表达式拆解


```
// ========== 逐步拆解复杂表达式 ==========
// 复杂表达式:
result = 2 * 3 + 4 ** 2 // 8 - 1

// 拆解步骤:
// 1. ** 优先: 4 ** 2 = 16
//    result = 2 * 3 + 16 // 8 - 1
// 2. * 和 // 从左到右: 2 * 3 = 6, 16 // 8 = 2
//    result = 6 + 2 - 1
// 3. + 和 - 从左到右: 6 + 2 = 8, 8 - 1 = 7
//    result = 7

// ========== 复杂逻辑表达式 ==========
condition = not has_error and status >= 200 or status < 300

// 等价于:
condition = (not has_error and (status >= 200)) or (status < 300)

// 本意可能是:
condition = not has_error and (status >= 200 or status < 300)

// 加括号明确意图!
condition = not has_error and (200 <= status < 300)

// ========== 混合位运算与比较 ==========
flags = 0b1100
mask = 0b1010

result = flags & mask == 0b1000  # False!
# 等价于: flags & (mask == 0b1000) → 0b1100 & False → 0b1100 & 0 → 0

result = (flags & mask) == 0b1000  # True ✅

// ========== 拆解建议 ==========
// 1. 不确定优先级就加括号
// 2. 复杂表达式拆成多步
// 3. 中间变量命名提高可读性
// 4. 不要写"聪明"的代码,要写清晰的代码
```


## 最佳实践


```
// ========== 加括号,不要依赖记忆 ==========
// ❌ 依赖优先级:
if a and b or c and not d:
    pass

// ✅ 加括号:
if (a and b) or (c and not d):
    pass

// ❌
value = 1 << 2 + 3 * 4

// ✅ 拆解:
shift = 2 + 3 * 4
value = 1 << shift

// ========== 常见模式 ==========
// 算术:
total = price * (1 + tax_rate) - discount  # ✅

// 逻辑:
is_valid = (score >= 60) and (score <= 100)  # ✅

// 位运算:
permission = (user_role & ADMIN) == ADMIN  # ✅

// ========== 优先级速记口诀 ==========
// 一 括 属 调  (括号/属性/调用)
// 二 幂        (**)
// 三 一元      (+x, -x, ~x)
// 四 乘除      (*, /, //, %)
// 五 加减      (+, -)
// 六 位运算    (<< >> & | ^)
// 七 比较      (in, is, <, >, ==)
// 八 逻辑      (not, and, or)
// 九 条件      (if-else)
// 十 λ         (lambda)

// 不确定时加括号,永远是对的!
```


> **Note:** 💡 优先级要点: (1) ** 是右结合,其余大部分左结合; (2) and 优先于 or; (3) 比较运算符可以链式; (4) 位运算优先级容易出错,加括号; (5) 不确定优先级时加括号,或者拆成多步。


## 练习


<!-- Converted from: 17_Python运算符优先级与复合表达式.html -->
