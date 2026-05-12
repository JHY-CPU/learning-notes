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

### 2.4 自定义解析器

```rust
use syn::parse::{Parse, ParseStream};

struct MyMacroInput {
    name: String,
    values: Vec<i32>,
}

impl Parse for MyMacroInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: syn::Ident = input.parse()?;
        input.parse::<syn::Token![=]>()?;
        let values: syn::punctuated::Punctuated<syn::LitInt, syn::Token![,]> =
            input.parse_terminated(syn::LitInt::parse)?;

        Ok(MyMacroInput {
            name: name.to_string(),
            values: values.iter()
                .map(|lit| lit.base10_parse().unwrap())
                .collect(),
        })
    }
}

// 使用
#[proc_macro]
pub fn my_config(input: TokenStream) -> TokenStream {
    let config = parse_macro_input!(input as MyMacroInput);
    let name = &config.name;
    let values = &config.values;

    quote! {
        let config_name = #name;
        let config_values: Vec<i32> = vec![#(#values),*];
    }.into()
}
```

### 2.5 结构体字段遍历

```rust
fn process_struct_fields(ast: &DeriveInput) -> proc_macro2::TokenStream {
    let fields = match &ast.data {
        syn::Data::Struct(data) => &data.fields,
        _ => panic!("仅支持结构体"),
    };

    let field_inits: Vec<_> = fields.iter().map(|f| {
        let name = f.ident.as_ref().unwrap();
        let ty = &f.ty;

        quote! {
            #name: <#ty as Default>::default()
        }
    }).collect();

    let field_names: Vec<_> = fields.iter()
        .filter_map(|f| f.ident.as_ref())
        .collect();

    quote! {
        fn new() -> Self {
            Self {
                #(#field_inits,)*
            }
        }

        fn field_names() -> Vec<&'static str> {
            vec![#(stringify!(#field_names),)*]
        }
    }
}
```

### 2.6 泛型与生命周期处理

```rust
fn handle_generics(ast: &DeriveInput) -> proc_macro2::TokenStream {
    let name = &ast.ident;
    let generics = &ast.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote! {
        impl #impl_generics MyTrait for #name #ty_generics #where_clause {
            fn do_something(&self) {
                println!("类型: {}", stringify!(#name));
            }
        }
    }
}
```

### 2.7 特性检测与条件代码

```rust
fn generate_conditional_code(ast: &DeriveInput) -> proc_macro2::TokenStream {
    let has_serde = ast.attrs.iter().any(|attr| {
        attr.path().is_ident("serde")
    });

    let extra_code = if has_serde {
        quote! {
            fn to_json(&self) -> String {
                serde_json::to_string(self).unwrap()
            }
        }
    } else {
        quote! {}
    };

    extra_code
}
```

## 四、syn 常用类型速查

| 类型 | 用途 |
|------|------|
| `DeriveInput` | 派生宏输入 |
| `ItemFn` | 函数项 |
| `ItemStruct` | 结构体项 |
| `ItemEnum` | 枚举项 |
| `ItemImpl` | impl 块 |
| `Expr` | 表达式 |
| `Type` | 类型 |
| `Pat` | 模式 |
| `Lit` | 字面量 |
| `Attribute` | 属性 |
| `Fields` | 字段集合 |
| `Field` | 单个字段 |

## 五、注意事项与常见陷阱

1. **版本匹配**：syn 和 quote 版本需匹配，通常同时升级
2. **Span 信息**：保留 Span 信息有助于错误定位，使用 `Span::call_site()` 作为默认
3. **内存占用**：大型语法树会占用较多内存，考虑优化
4. **性能优化**：避免重复解析和生成，缓存解析结果
5. **错误处理**：使用 `syn::Error` 提供友好的错误信息，支持多错误报告
6. **trait bounds**：处理泛型时需要正确生成 where 子句
7. **属性处理**：过滤和处理 derive helper 属性
