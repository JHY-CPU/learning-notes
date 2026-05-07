# SIMD加速

## 一、概念说明

WASM SIMD 提供 128 位向量指令，可显著加速数值计算、图像处理、加密等场景。

```rust
use std::arch::wasm32::*;

pub fn vector_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in (0..a.len()).step_by(4) {
        let va = v128_load(a.as_ptr().add(i) as *const v128);
        let vb = v128_load(b.as_ptr().add(i) as *const v128);
        let result = f32x4_add(va, vb);
        v128_store(out.as_mut_ptr().add(i) as *mut v128, result);
    }
}
```

## 二、具体用法

### 2.1 基本向量运算

```rust
use std::arch::wasm32::*;

pub fn f32x4_operations() {
    let a = f32x4(1.0, 2.0, 3.0, 4.0);
    let b = f32x4(5.0, 6.0, 7.0, 8.0);

    let sum = f32x4_add(a, b);        // [6.0, 8.0, 10.0, 12.0]
    let diff = f32x4_sub(b, a);       // [4.0, 4.0, 4.0, 4.0]
    let prod = f32x4_mul(a, b);       // [5.0, 12.0, 21.0, 32.0]
    let neg = f32x4_neg(a);           // [-1.0, -2.0, -3.0, -4.0]

    let max_val = f32x4_max(a, b);    // [5.0, 6.0, 7.0, 8.0]
    let sqrt_a = f32x4_sqrt(a);       // [1.0, 1.414, 1.732, 2.0]
}
```

### 2.2 整数向量运算

```rust
pub fn i32x4_operations() {
    let a = i32x4(1, 2, 3, 4);
    let b = i32x4(5, 6, 7, 8);

    let sum = i32x4_add(a, b);
    let diff = i32x4_sub(a, b);
    let min = i32x4_min(a, b);
    let max = i32x4_max(a, b);

    // 比较
    let eq = i32x4_eq(a, b);     // 全 false
    let gt = i32x4_gt(b, a);     // 全 true
}
```

### 2.3 图像灰度转换

```rust
pub fn rgb_to_grayscale_simd(input: &[u8], output: &mut [u8]) {
    // 灰度 = 0.299*R + 0.587*G + 0.114*B
    let r_coeff = f32x4(0.299, 0.299, 0.299, 0.299);
    let g_coeff = f32x4(0.587, 0.587, 0.587, 0.587);
    let b_coeff = f32x4(0.114, 0.114, 0.114, 0.114);

    for i in (0..input.len()).step_by(12) {
        // 加载 4 个像素的 RGB
        let r = f32x4(
            input[i] as f32, input[i+3] as f32,
            input[i+6] as f32, input[i+9] as f32,
        );
        let g = f32x4(
            input[i+1] as f32, input[i+4] as f32,
            input[i+7] as f32, input[i+10] as f32,
        );
        let b = f32x4(
            input[i+2] as f32, input[i+5] as f32,
            input[i+8] as f32, input[i+11] as f32,
        );

        let gray = f32x4_add(
            f32x4_add(f32x4_mul(r, r_coeff), f32x4_mul(g, g_coeff)),
            f32x4_mul(b, b_coeff),
        );

        let pixel = i / 3;
        output[pixel] = f32x4_extract_lane::<0>(gray) as u8;
        output[pixel+1] = f32x4_extract_lane::<1>(gray) as u8;
        output[pixel+2] = f32x4_extract_lane::<2>(gray) as u8;
        output[pixel+3] = f32x4_extract_lane::<3>(gray) as u8;
    }
}
```

### 2.4 条件编译

```rust
#[cfg(target_feature = "simd128")]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    // SIMD 版本
    unsafe { dot_product_simd(a, b) }
}

#[cfg(not(target_feature = "simd128"))]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    // 标量版本
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// 编译时启用 SIMD
// RUSTFLAGS="-C target-feature=+simd128" cargo build --release
```

### 2.5 矩阵乘法

```rust
pub fn matmul_4x4_simd(a: &[f32; 16], b: &[f32; 16], c: &mut [f32; 16]) {
    for i in 0..4 {
        let row = f32x4_load(&a[i * 4..]);
        for j in 0..4 {
            let col = f32x4(b[j], b[4+j], b[8+j], b[12+j]);
            let prod = f32x4_mul(row, col);
            // 水平求和
            c[i*4+j] = f32x4_extract_lane::<0>(prod)
                + f32x4_extract_lane::<1>(prod)
                + f32x4_extract_lane::<2>(prod)
                + f32x4_extract_lane::<3>(prod);
        }
    }
}
```

## 三、注意事项与常见陷阱

1. **浏览器支持**：Chrome 91+、Firefox 89+、Safari 16.4+ 支持
2. **对齐要求**：v128_load/store 推荐 16 字节对齐
3. **数据类型**：WASM SIMD 只有 128 位宽度，无 256/512 位
4. **剩余处理**：SIMD 循环末尾需用标量处理剩余元素
5. **调试困难**：SIMD 代码难以调试，建议先实现标量版本
