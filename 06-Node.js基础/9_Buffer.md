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


<!-- Converted from: 9_Buffer.html -->
