# Web Crypto


## Web Crypto


crypto.getRandomValues、SubtleCrypto(digest/encrypt/sign)、hash 计算。


## Web Crypto API


```
// ========== 随机数 ==========
const arr = new Uint8Array(32);
crypto.getRandomValues(arr);

// ========== Hash 计算 ==========
const hash = await crypto.subtle.digest('SHA-256', data);
// SHA-1, SHA-256, SHA-384, SHA-512

// ========== 加密/解密 ==========
const key = await crypto.subtle.generateKey(
    { name: 'AES-GCM', length: 256 },
    true, ['encrypt', 'decrypt']
);
const encrypted = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv: iv },
    key, plaintext
);

// ========== 签名/验证 ==========
const signature = await crypto.subtle.sign(
    { name: 'HMAC', hash: 'SHA-256' },
    key, data
);
const verified = await crypto.subtle.verify(
    { name: 'HMAC', hash: 'SHA-256' },
    key, signature, data
);
```


## 演示：Web Crypto

点击按钮查看


<!-- Converted from: 66_Web Crypto.html -->
