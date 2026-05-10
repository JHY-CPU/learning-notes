# Python 安全工具


## 🔒 Python 安全工具


hashlib 哈希、hmac 消息认证、passlib 密码哈希、python-jose JWT、密码学基础 (Fernet/RSA)、OWASP 常见防护。


## hashlib 哈希


```
// ========== hashlib ==========
import hashlib

# hashlib: 提供常见哈希算法

# 1. SHA-256
data = b"hello world"
hash_obj = hashlib.sha256(data)
print(hash_obj.hexdigest())
# b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9

# 2. 大文件分块哈希
def hash_file(filename):
    sha256 = hashlib.sha256()
    with open(filename, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)  # 累积更新
    return sha256.hexdigest()

# 3. 不同算法
print(hashlib.md5(b"hello").hexdigest())    # 128位 (不安全,仅用于校验)
print(hashlib.sha1(b"hello").hexdigest())    # 160位 (不推荐)
print(hashlib.sha256(b"hello").hexdigest())  # 256位 (推荐)
print(hashlib.sha512(b"hello").hexdigest())  # 512位

# 4. sha3 系列 (Python 3.6+)
print(hashlib.sha3_256(b"hello").hexdigest())

# 5. PBKDF2 密码哈希 (带盐,迭代)
dk = hashlib.pbkdf2_hmac(
    "sha256",
    b"password",
    b"salt1234",
    100000,  # 迭代次数 (越高越安全)
)
print(dk.hex())

# 6. 一致性哈希算法
def consistent_hash(key, num_buckets):
    """简单的一致性哈希"""
    hash_val = int(hashlib.md5(str(key).encode()).hexdigest(), 16)
    return hash_val % num_buckets
```


## hmac 与 passlib


```
// ========== hmac ==========
import hmac
import hashlib

# hmac: 基于哈希的消息认证码
# 验证消息完整性和身份

def create_hmac(key, message):
    return hmac.new(
        key.encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()

def verify_hmac(key, message, signature):
    expected = create_hmac(key, message)
    return hmac.compare_digest(expected, signature)
    # compare_digest: 常量时间比较,防止时序攻击

# Webhook 签名验证:
def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)

# ========== passlib ==========
# pip install passlib[bcrypt]

from passlib.hash import bcrypt, argon2, pbkdf2_sha256

# 推荐: bcrypt (自动管理 salt 和迭代次数)

# 哈希密码:
hash_str = bcrypt.hash("my_password")
print(hash_str)
# $2b$12$LJ3m4ys3Lk...

# 验证密码:
is_valid = bcrypt.verify("my_password", hash_str)  # True
is_invalid = bcrypt.verify("wrong", hash_str)       # False

# Argon2 (更安全,但更慢):
hash_str = argon2.hash("password")
print(argon2.verify("password", hash_str))

# pbkdf2_sha256:
hash_str = pbkdf2_sha256.hash("password")
print(pbkdf2_sha256.verify("password", hash_str))
```


## JWT (python-jose)


```
// ========== python-jose ==========
# pip install python-jose[cryptography]

from jose import jwt, JWTError
from datetime import datetime, timedelta, UTC

# JWT: JSON Web Token
# 结构: header.payload.signature

SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE = 30  # 分钟

# ========== 创建 Token ==========
def create_access_token(data: dict):
    to_encode = data.copy()

    # 添加过期时间
    expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE)
    to_encode.update({"exp": expire})

    # 可选: 添加其他标准声明
    # "iss" (issuer): 签发者
    # "sub" (subject): 主题
    # "aud" (audience): 受众
    # "iat" (issued at): 签发时间
    # "nbf" (not before): 生效时间

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ========== 验证 Token ==========
def verify_token(token: str):
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            # options={"verify_exp": True},
        )
        return payload
    except JWTError as e:
        # Token 过期 / 签名无效 / 格式错误
        print(f"Token 无效: {e}")
        return None

# 使用:
token = create_access_token({"sub": "user123", "role": "admin"})
payload = verify_token(token)
print(payload["sub"])  # "user123"

# ========== RS256 (非对称) ==========
# 生成密钥: openssl genrsa -out private.pem 2048
#           openssl rsa -in private.pem -pubout -out public.pem

# 私钥签名:
with open("private.pem") as f:
    private_key = f.read()
token = jwt.encode({"sub": "user1"}, private_key, algorithm="RS256")

# 公钥验证:
with open("public.pem") as f:
    public_key = f.read()
payload = jwt.decode(token, public_key, algorithms=["RS256"])
```


## cryptography 密码学


```
// ========== cryptography ==========
# pip install cryptography

# cryptography: 底层密码学库
# 提供对称加密/非对称加密/证书管理

# ========== Fernet 对称加密 ==========
from cryptography.fernet import Fernet

# 生成密钥
key = Fernet.generate_key()  # b'...' (URL-safe base64)
print(f"密钥: {key.decode()}")

# 创建加密器
cipher = Fernet(key)

# 加密
message = b"敏感数据: 密码是 12345"
encrypted = cipher.encrypt(message)
print(f"加密后: {encrypted}")

# 解密
decrypted = cipher.decrypt(encrypted)
print(f"解密后: {decrypted}")

# 带过期时间的加密:
encrypted = cipher.encrypt_at_time(message, current_time=datetime.now())
# decrypt 会检查时间戳

# ========== RSA 非对称加密 ==========
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

# 生成密钥对
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)
public_key = private_key.public_key()

# 加密 (公钥)
message = b"秘密消息"
ciphertext = public_key.encrypt(
    message,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None,
    ),
)

# 解密 (私钥)
plaintext = private_key.decrypt(
    ciphertext,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None,
    ),
)
print(plaintext.decode())
```


## 安全最佳实践


```
// ========== Python 安全实践 ==========
import os
import secrets

# 1. 安全随机数
# ❌ 不安全的随机数
import random
token = random.randint(0, 999999)  # 可预测!

# ✅ 安全的随机数
token = secrets.randbelow(1000000)  # 密码学安全
token = secrets.token_hex(32)       # 32 字节随机十六进制
token = secrets.token_urlsafe(32)   # URL 安全 base64

# 重置密码令牌:
reset_token = secrets.token_urlsafe(48)
# "3x4mpl3_t0k3n_..._urls4f3"

# 2. 环境变量存储密钥
# ❌ 硬编码
# SECRET_KEY = "hardcoded-key"

# ✅ 环境变量
SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY 未设置")

# 3. SQL 注入防护
# ❌ 不安全
# cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# ✅ 参数化查询
# cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))

# 4. 命令注入防护
# ❌ 不安全
# os.system(f"ping {user_input}")

# ✅ 使用 subprocess 带参数列表
import subprocess
# subprocess.run(["ping", "-c", "4", user_input], check=True)

# 5. HTML 转义 (XSS 防护)
import html
safe_text = html.escape(user_input)  # <script> → <script>
```


> **Note:** 💡 安全要点: hashlib 哈希/校验; passlib bcrypt 密码存储; python-jose JWT 令牌; cryptography/Fernet 对称加密; secrets 安全随机数; 环境变量管理密钥; 参数化查询防注入。


## 练习


<!-- Converted from: 127_Python 安全工具.html -->
