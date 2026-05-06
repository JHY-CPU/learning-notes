# BMP图像处理

## 概述

BMP（Bitmap）是Windows系统标准的位图格式，结构简单，适合学习图像文件格式的读写操作。

---

## 1. BMP文件格式

### 1.1 文件结构

```
+-------------------+
| BMP文件头 (14字节) |  - 文件大小、数据偏移等
+-------------------+
| DIB信息头 (40字节) |  - 图像宽高、色深等
+-------------------+
| 调色板 (可选)      |  - 8位及以下图像使用
+-------------------+
| 像素数据          |  - 实际图像数据
+-------------------+
```

### 1.2 数据结构定义

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#pragma pack(push, 1)  // 按1字节对齐

/* BMP文件头 (14字节) */
typedef struct {
    uint16_t type;        // 文件类型，必须为 "BM" (0x4D42)
    uint32_t file_size;   // 文件大小（字节）
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;      // 像素数据偏移量
} BMPFileHeader;

/* DIB信息头 (BITMAPINFOHEADER, 40字节) */
typedef struct {
    uint32_t header_size;   // 信息头大小，通常为40
    int32_t  width;         // 图像宽度（像素）
    int32_t  height;        // 图像高度（像素，正值=从下到上）
    uint16_t planes;        // 颜色平面数，必须为1
    uint16_t bits_per_pixel;// 每像素位数 (1/4/8/16/24/32)
    uint32_t compression;   // 压缩方式 (0=不压缩)
    uint32_t image_size;    // 像素数据大小
    int32_t  x_ppm;         // 水平分辨率
    int32_t  y_ppm;         // 垂直分辨率
    uint32_t colors_used;   // 使用的颜色数
    uint32_t colors_important;
} BMPInfoHeader;

#pragma pack(pop)

/* RGBA像素 */
typedef struct {
    uint8_t b, g, r, a;  // BMP中像素按BGR顺序存储
} BMPPixel;

/* BMP图像结构 */
typedef struct {
    BMPFileHeader file_header;
    BMPInfoHeader info_header;
    uint8_t *palette;     // 调色板（8位图使用）
    uint8_t *pixels;      // 像素数据
    int width, height;
    int bits_per_pixel;
} BMPImage;
```

---

## 2. BMP文件读取

```c
/*
 * 读取BMP文件
 * 支持 24位和32位无压缩BMP
 */
BMPImage *bmp_read(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return NULL;
    }

    BMPImage *bmp = (BMPImage *)calloc(1, sizeof(BMPImage));

    // 读取文件头
    fread(&bmp->file_header, sizeof(BMPFileHeader), 1, fp);

    // 验证文件类型
    if (bmp->file_header.type != 0x4D42) {
        fprintf(stderr, "不是有效的BMP文件\n");
        free(bmp);
        fclose(fp);
        return NULL;
    }

    // 读取信息头
    fread(&bmp->info_header, sizeof(BMPInfoHeader), 1, fp);

    bmp->width = bmp->info_header.width;
    bmp->height = abs(bmp->info_header.height);
    bmp->bits_per_pixel = bmp->info_header.bits_per_pixel;

    // 读取调色板（8位及以下）
    if (bmp->bits_per_pixel <= 8) {
        int palette_size = 1 << bmp->bits_per_pixel;
        bmp->palette = (uint8_t *)malloc(palette_size * 4);
        fread(bmp->palette, palette_size * 4, 1, fp);
    }

    // 计算像素数据大小
    int row_size = (bmp->width * bmp->bits_per_pixel + 31) / 32 * 4;
    int data_size = row_size * bmp->height;

    // 读取像素数据
    bmp->pixels = (uint8_t *)malloc(data_size);
    fseek(fp, bmp->file_header.offset, SEEK_SET);
    fread(bmp->pixels, data_size, 1, fp);

    fclose(fp);
    printf("成功读取BMP: %dx%d, %d bpp\n",
           bmp->width, bmp->height, bmp->bits_per_pixel);

    return bmp;
}
```

---

## 3. BMP文件写入

```c
/*
 * 写入24位BMP文件
 * 输入像素数据按RGB顺序，函数内部转换为BGR
 */
int bmp_write_24(const char *filename, int width, int height, uint8_t *rgb_data) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "无法创建文件: %s\n", filename);
        return -1;
    }

    int row_size = (width * 3 + 3) / 4 * 4;  // 每行按4字节对齐
    int data_size = row_size * height;

    BMPFileHeader fh;
    fh.type = 0x4D42;
    fh.file_size = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + data_size;
    fh.reserved1 = 0;
    fh.reserved2 = 0;
    fh.offset = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

    BMPInfoHeader ih;
    memset(&ih, 0, sizeof(ih));
    ih.header_size = 40;
    ih.width = width;
    ih.height = height;  // 正值表示从下到上
    ih.planes = 1;
    ih.bits_per_pixel = 24;
    ih.compression = 0;
    ih.image_size = data_size;

    fwrite(&fh, sizeof(fh), 1, fp);
    fwrite(&ih, sizeof(ih), 1, fp);

    // 写入像素数据（BMP从下到上存储）
    uint8_t *row_buffer = (uint8_t *)calloc(row_size, 1);
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int src_idx = (y * width + x) * 3;
            int dst_idx = x * 3;
            row_buffer[dst_idx + 0] = rgb_data[src_idx + 2]; // B
            row_buffer[dst_idx + 1] = rgb_data[src_idx + 1]; // G
            row_buffer[dst_idx + 2] = rgb_data[src_idx + 0]; // R
        }
        // 写入一行（含填充字节）
        fwrite(row_buffer, row_size, 1, fp);
    }

    free(row_buffer);
    fclose(fp);
    printf("成功写入BMP: %s (%dx%d)\n", filename, width, height);
    return 0;
}
```

---

## 4. BMP图像操作

### 4.1 获取/设置像素

```c
/* 获取24位BMP的像素 (R, G, B) */
void bmp_get_pixel_24(BMPImage *bmp, int x, int y, uint8_t *r, uint8_t *g, uint8_t *b) {
    int row_size = (bmp->width * 3 + 3) / 4 * 4;
    // BMP从下到上存储
    int row = (bmp->info_header.height > 0) ? (bmp->height - 1 - y) : y;
    int offset = row * row_size + x * 3;
    *b = bmp->pixels[offset + 0];
    *g = bmp->pixels[offset + 1];
    *r = bmp->pixels[offset + 2];
}

/* 设置24位BMP的像素 */
void bmp_set_pixel_24(BMPImage *bmp, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    int row_size = (bmp->width * 3 + 3) / 4 * 4;
    int row = (bmp->info_header.height > 0) ? (bmp->height - 1 - y) : y;
    int offset = row * row_size + x * 3;
    bmp->pixels[offset + 0] = b;
    bmp->pixels[offset + 1] = g;
    bmp->pixels[offset + 2] = r;
}
```

### 4.2 灰度化

```c
/* BMP图像灰度化 */
BMPImage *bmp_to_gray(BMPImage *bmp) {
    BMPImage *gray = (BMPImage *)calloc(1, sizeof(BMPImage));
    memcpy(gray, bmp, sizeof(BMPImage));

    int row_size = (bmp->width * 3 + 3) / 4 * 4;
    gray->pixels = (uint8_t *)malloc(row_size * bmp->height);

    for (int y = 0; y < bmp->height; y++) {
        for (int x = 0; x < bmp->width; x++) {
            uint8_t r, g, b;
            bmp_get_pixel_24(bmp, x, y, &r, &g, &b);
            uint8_t gray_val = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
            bmp_set_pixel_24(gray, x, y, gray_val, gray_val, gray_val);
        }
    }

    return gray;
}
```

### 4.3 水平翻转

```c
/* 水平翻转 */
BMPImage *bmp_flip_horizontal(BMPImage *bmp) {
    BMPImage *result = (BMPImage *)calloc(1, sizeof(BMPImage));
    memcpy(result, bmp, sizeof(BMPImage));

    int row_size = (bmp->width * 3 + 3) / 4 * 4;
    result->pixels = (uint8_t *)malloc(row_size * bmp->height);

    for (int y = 0; y < bmp->height; y++) {
        for (int x = 0; x < bmp->width; x++) {
            uint8_t r, g, b;
            bmp_get_pixel_24(bmp, x, y, &r, &g, &b);
            bmp_set_pixel_24(result, bmp->width - 1 - x, y, r, g, b);
        }
    }

    return result;
}
```

### 4.4 90度旋转

```c
/* 顺时针旋转90度 */
BMPImage *bmp_rotate_90(BMPImage *bmp) {
    BMPImage *result = (BMPImage *)calloc(1, sizeof(BMPImage));
    memcpy(result, bmp, sizeof(BMPImage));
    result->width = bmp->height;
    result->height = bmp->width;
    result->info_header.width = result->width;
    result->info_header.height = result->height;

    int row_size = (result->width * 3 + 3) / 4 * 4;
    result->pixels = (uint8_t *)calloc(row_size * result->height, 1);

    for (int y = 0; y < bmp->height; y++) {
        for (int x = 0; x < bmp->width; x++) {
            uint8_t r, g, b;
            bmp_get_pixel_24(bmp, x, y, &r, &g, &b);
            // (x, y) -> (bmp.height - 1 - y, x)
            bmp_set_pixel_24(result, bmp->height - 1 - y, x, r, g, b);
        }
    }

    return result;
}
```

### 4.5 裁剪

```c
/* 裁剪区域 */
BMPImage *bmp_crop(BMPImage *bmp, int x, int y, int w, int h) {
    if (x < 0 || y < 0 || x + w > bmp->width || y + h > bmp->height) {
        fprintf(stderr, "裁剪区域超出图像范围\n");
        return NULL;
    }

    BMPImage *result = (BMPImage *)calloc(1, sizeof(BMPImage));
    memcpy(result, bmp, sizeof(BMPImage));
    result->width = w;
    result->height = h;
    result->info_header.width = w;
    result->info_header.height = h;

    int row_size = (w * 3 + 3) / 4 * 4;
    result->pixels = (uint8_t *)calloc(row_size * h, 1);

    for (int cy = 0; cy < h; cy++) {
        for (int cx = 0; cx < w; cx++) {
            uint8_t r, g, b;
            bmp_get_pixel_24(bmp, x + cx, y + cy, &r, &g, &b);
            bmp_set_pixel_24(result, cx, cy, r, g, b);
        }
    }

    return result;
}
```

---

## 5. 内存释放

```c
void bmp_free(BMPImage *bmp) {
    if (bmp) {
        free(bmp->palette);
        free(bmp->pixels);
        free(bmp);
    }
}
```

---

## 6. 测试示例

```c
int main() {
    // 创建一个测试图像
    int w = 256, h = 256;
    uint8_t *data = (uint8_t *)malloc(w * h * 3);

    // 生成渐变图案
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;
            data[idx + 0] = (uint8_t)x;        // R
            data[idx + 1] = (uint8_t)y;        // G
            data[idx + 2] = (uint8_t)(x ^ y);  // B
        }
    }

    bmp_write_24("test_output.bmp", w, h, data);
    free(data);

    // 读取并处理
    BMPImage *bmp = bmp_read("test_output.bmp");
    if (bmp) {
        BMPImage *gray = bmp_to_gray(bmp);
        bmp_write_24("test_gray.bmp", gray->width, gray->height, gray->pixels);
        bmp_free(gray);
        bmp_free(bmp);
    }

    return 0;
}
```

---

## 小结

- BMP文件头和信息头需要按1字节对齐（`#pragma pack(1)`）
- BMP像素数据按BGR顺序存储，且通常从下到上排列
- 每行像素数据按4字节对齐，不足部分填充零
- 24位BMP没有调色板，直接存储像素值
