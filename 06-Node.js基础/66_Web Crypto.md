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


## 完整加密工具

```javascript
// ========== AES-GCM 加密解密 ==========
async function generateAESKey() {
    return crypto.subtle.generateKey(
        { name: 'AES-GCM', length: 256 },
        true, ['encrypt', 'decrypt']
    );
}

async function encryptAES(plaintext, key) {
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const encoded = new TextEncoder().encode(plaintext);
    const ciphertext = await crypto.subtle.encrypt(
        { name: 'AES-GCM', iv },
        key,
        encoded
    );
    return { iv, ciphertext: new Uint8Array(ciphertext) };
}

async function decryptAES(encrypted, key) {
    const decrypted = await crypto.subtle.decrypt(
        { name: 'AES-GCM', iv: encrypted.iv },
        key,
        encrypted.ciphertext
    );
    return new TextDecoder().decode(decrypted);
}

// ========== 导出/导入密钥 ==========
async function exportKey(key) {
    const raw = await crypto.subtle.exportKey('raw', key);
    return btoa(String.fromCharCode(...new Uint8Array(raw)));
}

async function importKey(base64Key) {
    const raw = Uint8Array.from(atob(base64Key), c => c.charCodeAt(0));
    return crypto.subtle.importKey('raw', raw, { name: 'AES-GCM' }, true, ['encrypt', 'decrypt']);
}

// ========== RSA 签名验证 ==========
async function generateRSAKeyPair() {
    return crypto.subtle.generateKey(
        { name: 'RSASSA-PKCS1-v1_5', modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: 'SHA-256' },
        true, ['sign', 'verify']
    );
}

async function sign(data, privateKey) {
    const encoded = new TextEncoder().encode(data);
    const signature = await crypto.subtle.sign('RSASSA-PKCS1-v1_5', privateKey, encoded);
    return new Uint8Array(signature);
}

async function verify(data, signature, publicKey) {
    const encoded = new TextEncoder().encode(data);
    return crypto.subtle.verify('RSASSA-PKCS1-v1_5', publicKey, signature, encoded);
}

// ========== 密码派生 (PBKDF2) ==========
async function deriveKey(password, salt) {
    const enc = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
        'raw', enc.encode(password), 'PBKDF2', false, ['deriveKey']
    );
    return crypto.subtle.deriveKey(
        { name: 'PBKDF2', salt, iterations: 100000, hash: 'SHA-256' },
        keyMaterial,
        { name: 'AES-GCM', length: 256 },
        false,
        ['encrypt', 'decrypt']
    );
}
```

## Hash 与随机数

```javascript
// ========== 文件 Hash ==========
async function hashFile(file) {
    const buffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = new Uint8Array(hashBuffer);
    return Array.from(hashArray).map(b => b.toString(16).padStart(2, '0')).join('');
}

// ========== 安全随机数 ==========
function generateSecureRandom(length = 32) {
    const array = new Uint8Array(length);
    crypto.getRandomValues(array);
    return Array.from(array).map(b => b.toString(16).padStart(2, '0')).join('');
}

// ========== 生成 UUID ==========
function generateUUID() {
    return crypto.randomUUID();
}
```

<!-- Converted from: 66_Web Crypto.html -->
