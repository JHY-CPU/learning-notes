# 类型推断与Hindley-Milner


## 一、Hindley-Milner类型系统概述


Hindley-Milner (HM) 类型系统是ML家族语言（ML, OCaml, Haskell）的基础，也是许多现代类型系统的灵感来源。


### 核心特性


- **多态类型：**
   支持参数多态（泛型）
- **类型推断：**
   完全自动推断，无需类型注解
- **let-多态：**
   let绑定获得多态类型
- **主类型：**
   每个表达式有最通用的类型


### 类型语法


```
类型 τ ::= α          -- 类型变量 (如 a, b, c)
          | T          -- 基础类型 (如 Int, Bool)
          | τ₁ → τ₂   -- 函数类型
          | ∀α. τ      -- 多态类型 (type scheme)
```


## 二、HM类型规则


### 推断规则


> **Example:** ```
> -- 变量（实例化多态类型）
> (x : ∀α. τ) ∈ Γ
> ────────────────────
> Γ ⊢ x : τ[β/α]       -- 用新类型变量替换
>
> -- 常量
> ────────────
> Γ ⊢ n : Int
>
> ────────────
> Γ ⊢ True : Bool
>
> -- 函数应用
> Γ ⊢ e₁ : τ₁ → τ₂    Γ ⊢ e₂ : τ₁
> ───────────────────────────────────
> Γ ⊢ e₁ e₂ : τ₂
>
> -- λ抽象
> Γ, x : τ₁ ⊢ e : τ₂
> ──────────────────────
> Γ ⊢ λx. e : τ₁ → τ₂
>
> -- let绑定（引入let-多态）
> Γ ⊢ e₁ : τ₁    Γ, x : Gen(Γ, τ₁) ⊢ e₂ : τ₂
> ──────────────────────────────────────────────
> Γ ⊢ let x = e₁ in e₂ : τ₂
> ```


> **Theorem:** **关键：**let-多态使得 `let id = λx. x in (id 5, id True)` 合法。每次使用 `id` 时，它的类型被**重新实例化**为新变量。


## 三、合一算法 (Unification)


合一算法是HM类型推断的核心——它找到使两个类型相等的**最通用替换**。


### 合一的定义


对类型 τ₁ 和 τ₂，找到替换 θ 使得 θ(τ₁) = θ(τ₂)。


### 算法


```
def unify(t1, t2):
    if t1 == t2:
        return {}                          -- 相同类型，空替换
    if is_type_var(t1) and t1 not in free_vars(t2):
        return {t1: t2}                    -- 替换变量
    if is_type_var(t2) and t2 not in free_vars(t1):
        return {t2: t1}                    -- 替换变量
    if is_function(t1) and is_function(t2):
        # t1 = a→b, t2 = c→d
        s1 = unify(t1.arg, t2.arg)
        s2 = unify(apply(s1, t1.ret), apply(s1, t2.ret))
        return compose(s2, s1)
    raise TypeError(f"Cannot unify {t1} and {t2}")
```


### 示例：推断 id 5 的类型


> **Example:** **表达式：**`(λx. x) 5`
>
> 1.
> 给 λx.x 分配类型 α → α
> 2.
> 5 的类型是 Int
> 3.
> 应用规则：α → α 应用于 Int，需要 α = Int
> 4.
> 合一：α ~ Int，替换 θ = {α ↦ Int}
> 5.
> 结果类型：
> **Int**


### 更复杂的示例：推断 λf. λx. f (f x) 的类型


> **Example:** 1.
> 设 f : α, x : β, 表达式 f (f x)
> 2.
> f x 中 f 需要是 β → γ，所以 α = β → γ
> 3.
> f (f x) 中外层 f 需要是 γ → δ，所以 α = γ → δ
> 4.
> 合一：β → γ = γ → δ，得 β = γ, γ = δ
> 5.
> 所以 β = γ = δ，α = β → β
> 6.
> 最终类型：
> **(β → β) → β → β**
> （多态！）


## 四、Occurs Check（出现检查）


合一算法中必须检查：不能将类型变量替换为包含自身的类型。


> **Example:** ```
> -- 尝试合一 α 和 α → Int
> -- 如果允许 α = α → Int，会导致无限类型！
> -- α = (α→Int)→Int = ((α→Int)→Int)→Int = ... 无穷展开
>
> def occurs_check(var, typ):
>     """检查 var 是否出现在 typ 的自由变量中"""
>     return var in free_type_variables(typ)
>
> # 合一失败示例
> unify(α, α → Int)
> # occurs_check(α, α→Int) = True
> # → TypeError: "Cannot construct infinite type: α ~ α → Int"
> ```


没有occurs check会导致无限递归类型，这是非法的（在标准HM中）。


## 五、HM系统的局限与扩展


| 局限 | 描述 | 扩展方案 |
| --- | --- | --- |
| 无高阶多态 | 不能写 `∀f. f Int` | System F (显式类型抽象) |
| 无子类型 | 不能表达继承关系 | 带子类型的HM |
| 无递归类型 | 不能表达自引用类型 | μ类型或隐式递归 |
| 无类型类 | 不能表达约束 (Eq a =>) | HM + 类型类 (Haskell) |
| 单相性 | let参数不能用于不同用途 | MLF, Quick Look (GHC 9.0+) |


### Haskell对HM的扩展


```
-- Haskell类型类（对HM的扩展）
class Eq a where
    (==) :: a -> a -> Bool

-- 约束多态
elem :: Eq a => a -> [a] -> Bool
elem _ [] = False
elem x (y:ys) = x == y || elem x ys

-- GADTs（广义代数数据类型）
data Expr a where
    Lit  :: Int -> Expr Int
    Bool :: Bool -> Expr Bool
    Add  :: Expr Int -> Expr Int -> Expr Int
    If   :: Expr Bool -> Expr a -> Expr a -> Expr a
```


<!-- Converted from: 02_类型推断与Hindley-Milner.html -->
