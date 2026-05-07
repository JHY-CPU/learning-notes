# syn与quote库

## 一、概念说明

`syn` 解析 Rust 代码为语法树，`quote` 将语法树转换回 Rust 代码。它们是过程宏开发的核心依赖。

```rust
use syn::{parse_str, DeriveInput};
use quote::quote;

// 解析
let ast: DeriveInput = parse_str("struct Foo { x: i32 }").unwrap();

// 生成
let code = quote! { fn hello() {} };
```

## 二、具体用法

### 2.1 syn 解析

```rust
use syn::{parse_macro_input, DeriveInput, ItemFn, Attribute};

// 解析派生宏输入
let ast = parse_macro_input!(input as DeriveInput);

// 解析函数
let func = parse_macro_input!(input as ItemFn);

// 提取信息
let name = &ast.ident;
let fields = match &ast.data {
    syn::Data::Struct(data) => &data.fields,
    _ => panic!("只支持结构体"),
};
```

### 2.2 quote 生成

```rust
use quote::quote;

let name = syn::Ident::new("MyStruct", proc_macro2::Span::call_site());

let output = quote! {
    impl #name {
        fn new() -> Self {
            #name {}
        }
    }
};

// 使用重复
let fields = vec!["a", "b", "c"];
let output = quote! {
    struct MyStruct {
        #(#fields: i32),*
    }
};
```

### 2.3 调试技巧

```rust
// 打印生成的代码
let code = quote! { ... };
eprintln!("{}", code);

// 使用 cargo expand
// cargo expand --lib
```

## 三、注意事项与常见陷阱

1. **版本匹配**：syn 和 quote 版本需匹配
2. **Span 信息**：保留 Span 信息有助于错误定位
3. **内存占用**：大型语法树会占用较多内存
4. **性能优化**：避免重复解析和生成
5. **错误处理**：使用 syn::Error 提供友好的错误信息
