# Buffer


## Buffer


Buffer 创建、toString 编码转换、读写、concat、slice。


## Buffer API


```
// ========== Buffer 创建 ==========
Buffer.from('hello')              // 字符串 → Buffer
Buffer.from([104, 101, 108, 108, 111]) // 字节数组 → Buffer
Buffer.from('hello', 'hex')       // 十六进制字符串
Buffer.alloc(1024)                // 分配 1KB (初始化为 0)
Buffer.allocUnsafe(1024)          // 分配 1KB (可能含旧数据，更快)

// ========== Buffer 读写 ==========
const buf = Buffer.from('hello');
buf[0]          // 104 (ASCII 'h')
buf.length      // 5
buf.byteOffset  // 0

// ========== 编码转换 ==========
buf.toString()           // 'hello' (默认 utf-8)
buf.toString('hex')      // '68656c6c6f'
buf.toString('base64')   // 'aGVsbG8='
buf.toString('utf-8')    // 'hello'
buf.toString('ascii')
buf.toString('latin1')   // 或 'binary'

// ========== 其他操作 ==========
Buffer.concat([buf1, buf2])       // 合并
buf.slice(0, 2)                   // 切片 (共享内存)
buf.copy(target, 0, 0, 2)        // 复制到目标 Buffer
Buffer.compare(buf1, buf2)        // 比较
buf.indexOf('ll')                 // 搜索
buf.includes('he')                // 是否包含

// ========== 静态属性 ==========
Buffer.poolSize        // 内存池大小 (8192)
Buffer.isBuffer(obj)   // 是否 Buffer
```


> **Note:** 💡 Buffer 是 Node.js 中处理二进制数据的核心。
>
>
> ⚠️ allocUnsafe 更快但不安全（含有旧数据）。
>
>
> ⚠️ slice 返回的 Buffer 与原 Buffer 共享内存。


## 演示：Buffer 操作

点击按钮查看


## Buffer 实战：文件哈希计算

```javascript
// ========== 使用 Buffer 计算文件哈希 ==========
const crypto = require('crypto');
const fs = require('fs');

function computeHash(filePath, algorithm = 'sha256') {
    return new Promise((resolve, reject) => {
        const hash = crypto.createHash(algorithm);
        const stream = fs.createReadStream(filePath);
        stream.on('data', (chunk) => hash.update(chunk));
        stream.on('end', () => resolve(hash.digest('hex')));
        stream.on('error', reject);
    });
}

// 使用
const hash = await computeHash('./package.json');
console.log(hash); // 'a1b2c3d4...'
```

## Buffer 编解码实战

```javascript
// ========== Base64 编解码 ==========
// 字符串 → Base64
const encoded = Buffer.from('Hello 你好').toString('base64');
console.log(encoded); // 'SGVsbG8g5L2g5aW9'

// Base64 → 字符串
const decoded = Buffer.from(encoded, 'base64').toString('utf-8');
console.log(decoded); // 'Hello 你好'

// ========== 图片转 Base64 (用于内联图片) ==========
const imageBuffer = fs.readFileSync('./logo.png');
const base64Image = `data:image/png;base64,${imageBuffer.toString('base64')}`;
// 可直接用于 <img src="data:image/png;base64,...">

// ========== URL 安全的 Base64 ==========
function toUrlSafeBase64(buffer) {
    return buffer.toString('base64')
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/, '');
}

function fromUrlSafeBase64(str) {
    str = str.replace(/-/g, '+').replace(/_/g, '/');
    while (str.length % 4) str += '=';
    return Buffer.from(str, 'base64');
}
```

## Buffer 与二进制协议

```javascript
// ========== 读写二进制数据 ==========
const buf = Buffer.alloc(16); // 分配16字节

// 写入不同类型数据
buf.writeUInt8(255, 0);          // 1字节无符号整数 (偏移0)
buf.writeUInt16LE(65535, 1);     // 2字节小端序 (偏移1)
buf.writeUInt32BE(4294967295, 3);// 4字节大端序 (偏移3)
buf.writeDoubleLE(3.14, 7);      // 8字节双精度 (偏移7)

// 读取
console.log(buf.readUInt8(0));        // 255
console.log(buf.readUInt16LE(1));     // 65535
console.log(buf.readUInt32BE(3));     // 4294967295
console.log(buf.readDoubleLE(7));     // 3.14

// ========== 文件头解析 (如解析 PNG) ==========
function parsePngHeader(buffer) {
    // PNG 签名: 89 50 4E 47 0D 0A 1A 0A
    const signature = buffer.slice(0, 8).toString('hex');
    if (signature !== '89504e470d0a1a0a') {
        throw new Error('不是有效的 PNG 文件');
    }
    return {
        width: buffer.readUInt32BE(16),
        height: buffer.readUInt32BE(20),
        bitDepth: buffer.readUInt8(24),
        colorType: buffer.readUInt8(25),
    };
}
```

## Buffer 性能优化

```javascript
// ========== Buffer 池 ==========
// Node.js 使用内存池管理小 Buffer (< 4KB)
// Buffer.allocUnsafe 配合 fill 可利用池化内存

function createBufferFast(size, fill = 0) {
    const buf = Buffer.allocUnsafe(size);
    buf.fill(fill);
    return buf;
}

// ========== 避免频繁创建/销毁 ==========
// 不好的做法
for (let i = 0; i < 10000; i++) {
    const buf = Buffer.alloc(1024); // 每次都分配新内存
}

// 好的做法: 复用 Buffer
const reusableBuf = Buffer.alloc(1024);
for (let i = 0; i < 10000; i++) {
    reusableBuf.fill(0);
    // 使用 reusableBuf...
}

// ========== 大 Buffer 注意事项 ==========
// Buffer 大小限制: ~2GB (v8 限制)
// 超大文件应使用 Stream 而非一次性读入 Buffer
```

<!-- Converted from: 9_Buffer.html -->
