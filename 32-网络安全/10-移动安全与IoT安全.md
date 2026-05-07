# 10-移动安全与IoT安全

## 一、Android安全

### 1.1 Android安全架构

```
Android安全层次：
┌─────────────────────────────┐
│        应用层                │
│  应用沙箱、权限管理、应用签名  │
├─────────────────────────────┤
│        框架层                │
│  Android Framework API      │
│  权限检查、内容提供者安全     │
├─────────────────────────────┤
│        运行时                │
│  ART/Dalvik虚拟机            │
│  内存安全、类型安全           │
├─────────────────────────────┤
│        原生层                │
│  系统库、HAL                 │
│  SELinux强制访问控制         │
├─────────────────────────────┤
│        内核层                │
│  Linux内核                  │
│  进程隔离、文件权限、能力模型  │
└─────────────────────────────┘
```

### 1.2 Android应用安全

#### 应用组件安全

| 组件 | 风险 | 防护 |
|------|------|------|
| Activity | 界面劫持、未授权启动 | exported=false、权限保护 |
| Service | 后台服务滥用 | 权限声明、前台服务 |
| BroadcastReceiver | 广播拦截、注入 | 本地广播、权限保护 |
| Content Provider | SQL注入、文件泄露 | 参数化查询、URI权限 |

#### 权限系统

```xml
<!-- AndroidManifest.xml -->
<!-- 声明权限 -->
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />

<!-- 保护组件 -->
<activity
    android:name=".SecretActivity"
    android:exported="false"
    android:permission="com.app.ACCESS_SECRET" />
```

#### 网络安全

```
网络安全配置（network_security_config.xml）：
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config cleartextTrafficPermitted="false">
        <domain includeSubdomains="true">api.example.com</domain>
        <pin-set expiration="2025-01-01">
            <pin digest="SHA-256">base64encodedpin=</pin>
        </pin-set>
    </domain-config>
</network-security-config>
```

### 1.3 Android应用安全测试

```bash
# 静态分析
apktool d app.apk           # 反编译APK
jadx -d output/ app.apk     # 反编译为Java源码
dex2jar app.apk             # 转换为JAR

# 动态分析
adb shell                   # 连接设备
frida -U -f com.app         # Frida动态插桩
objection -g com.app explore # Objection工具

# 常用工具
MobSF                      # 移动安全框架
Drozer                     # Android安全测试框架
APKTool                    # APK反编译
JEB                        # 商业反编译器
```

### 1.4 Android常见漏洞

| 漏洞类型 | 描述 | 工具检测 |
|----------|------|----------|
| 硬编码密钥 | API密钥、密码写在代码中 | 静态分析 |
| 不安全的数据存储 | 敏感数据明文存储在SharedPreferences | Drozer |
| 不安全的通信 | 未使用HTTPS或证书校验不严 | Burp Suite |
| 组件暴露 | 组件未设访问控制 | Drozer |
| WebView漏洞 | JavaScript接口暴露 | MobSF |
| 代码注入 | 动态加载、反序列化 | Frida |
| ROOT检测绕过 | 安全检测被绕过 | Frida |

---

## 二、iOS安全

### 2.1 iOS安全架构

```
iOS安全层次：
├── 硬件安全
│   ├── Secure Enclave（安全隔区）
│   ├── AES-256硬件加密引擎
│   ├── Touch ID/Face ID传感器
│   └── 设备唯一密钥（UID）
├── 系统安全
│   ├── Secure Boot Chain（安全启动链）
│   ├── 代码签名
│   ├── 沙箱机制
│   └── ASLR/DEP
├── 数据安全
│   ├── 文件数据保护（Data Protection）
│   ├── 钥匙串（Keychain）
│   └── 硬件加密
└── 网络安全
    ├── App Transport Security（ATS）
    ├── 证书固定
    └── VPN支持
```

### 2.2 iOS数据保护级别

| 级别 | 标记 | 可访问时间 |
|------|------|-----------|
| Complete Protection | NSFileProtectionComplete | 设备解锁时 |
| Protected Unless Open | NSFileProtectionCompleteUnlessOpen | 设备解锁后保持打开 |
| Protected Until First Auth | NSFileProtectionCompleteUntilFirstAuthentication | 首次认证后 |
| No Protection | NSFileProtectionNone | 始终可访问 |

### 2.3 iOS应用安全测试

```bash
# 越狱设备上
# 安装工具
apt install frida
pip install objection

# 应用数据提取
scp -r root@device:/var/mobile/Containers/Data/Application/UUID/ .

# SSL Pinning绕过
# 使用Frida脚本
frida -U -f com.app -l ssl-pinning-bypass.js

# 常用工具
objection - 运行时探索
class-dump - 导出头文件
Hopper/IDA - 反汇编
```

### 2.4 iOS常见漏洞

| 漏洞 | 描述 |
|------|------|
| Keychain数据泄露 | 未正确设置访问控制 |
| URL Scheme劫持 | 恶意应用注册相同URL Scheme |
| 剪贴板泄露 | 敏感数据留存在剪贴板 |
| 不安全的数据存储 | Plist、SQLite明文存储 |
| 证书校验不足 | 未实现证书固定 |
| 越狱检测绕过 | 安全检查不完善 |

---

## 三、移动应用安全测试

### 3.1 OWASP MASVS（移动应用安全验证标准）

```
安全级别：
├── L1 - 基础安全（所有应用）
├── L2 - 增强安全（处理敏感数据的应用）
├── R - 反篡改/逆向保护
└── R+R - 高安全要求

测试类别：
V1: 架构、设计和威胁建模
V2: 数据存储和隐私
V3: 加密验证
V4: 身份认证和会话管理
V5: 网络通信
V6: 平台交互
V7: 代码质量和构建设置
V8: 抗篡改和逆向
V9: 额外安全控制
```

### 3.2 移动应用安全编码

```
安全编码清单：
✅ 敏感数据加密存储
✅ HTTPS + 证书固定
✅ 服务器端认证和授权
✅ 输入验证（客户端+服务端）
✅ 代码混淆和加固
✅ 安全的会话管理
✅ 反调试和反篡改
✅ 不在日志中记录敏感信息
✅ 安全的第三方库
✅ ROOT/越狱检测
```

---

## 四、IoT安全

### 4.1 IoT安全架构

```
IoT安全层次：
┌─────────────────────────────────┐
│          应用层                  │
│  云平台安全、Web应用安全          │
├─────────────────────────────────┤
│          传输层                  │
│  TLS/DTLS、MQTT安全、CoAP安全   │
├─────────────────────────────────┤
│          网络层                  │
│  6LoWPAN安全、Zigbee安全         │
├─────────────────────────────────┤
│          感知层                  │
│  设备认证、固件安全、物理安全     │
└─────────────────────────────────┘
```

### 4.2 IoT安全威胁

| 威胁类别 | 具体威胁 |
|----------|----------|
| 设备安全 | 默认密码、固件漏洞、缺少安全启动 |
| 通信安全 | 明文传输、协议漏洞、中间人攻击 |
| 云平台安全 | API漏洞、认证缺陷、配置错误 |
| 物理安全 | 物理篡改、侧信道攻击、调试接口 |
| 隐私安全 | 数据采集过度、数据泄露 |
| DDoS | 僵尸网络（如Mirai） |

### 4.3 IoT常见协议安全

| 协议 | 安全机制 | 常见问题 |
|------|----------|----------|
| MQTT | TLS + 用户名/密码 | 默认无认证、明文传输 |
| CoAP | DTLS | 资源受限、DDoS放大 |
| Zigbee | AES-128-CCM | 密钥泄露、重放攻击 |
| Z-Wave | S2安全框架 | 早期版本无加密 |
| BLE | LE Secure Connections | 配对攻击、嗅探 |
| LoRaWAN | AES-128 | 密钥管理问题 |

### 4.4 IoT固件安全分析

```bash
# 固件提取
binwalk -e firmware.bin       # 提取固件文件系统
firmware-mod-kit              # 固件修改工具

# 固件分析
# 1. 文件系统分析
ls -la squashfs-root/
strings squashfs-root/bin/busybox

# 2. 查找硬编码凭据
grep -r "password" squashfs-root/
grep -r "admin" squashfs-root/etc/

# 3. 识别串口调试接口
# 通过串口连接获取shell

# 4. JTAG调试
OpenOCD - 连接JTAG接口

# 常用工具
├── Binwalk - 固件提取和分析
├── Firmwalker - 固件文件系统搜索
├── Firmadyne - 固件模拟
├── Ghidra - 固件逆向
├── FACT - 固件分析比较工具
└── Emba - 固件安全分析
```

### 4.5 IoT安全最佳实践

```
设备安全：
├── 安全启动（Secure Boot）
├── 固件签名验证
├── 安全OTA更新
├── 硬件安全模块（TPM/SE）
├── 唯一设备身份
├── 禁用调试接口
└── 最小化服务

通信安全：
├── TLS/DTLS加密
├── 证书双向认证
├── 安全密钥存储
├── 协议版本控制
└── 防重放攻击

云平台安全：
├── API认证（OAuth 2.0）
├── 速率限制
├── 设备认证
├── 异常检测
└── 安全日志

运维安全：
├── 固件安全更新机制
├── 漏洞管理
├── 设备监控
├── 安全退役（数据擦除）
└── 供应链安全
```

---

## 五、车联网安全

### 5.1 车联网攻击面

```
攻击面：
├── 远程攻击
│   ├── T-Box（远程通信模块）
│   ├── 云平台API
│   ├── 手机APP
│   └── V2X通信
├── 近场攻击
│   ├── 蓝牙
│   ├── Wi-Fi
│   ├── NFC
│   └── TPMS（胎压监测）
├── 物理攻击
│   ├── OBD-II接口
│   ├── USB接口
│   ├── CAN总线
│   └── ECU调试接口
└── 供应链攻击
    ├── 第三方组件
    ├── 软件供应链
    └── 维修渠道
```

### 5.2 CAN总线安全

```
CAN总线安全问题：
├── 无认证 - 任何设备可发送CAN帧
├── 无加密 - 数据明文传输
├── 无授权 - 无法限制访问
└── 广播 - 所有节点接收所有消息

攻击方式：
├── 嗅探 - 捕获CAN帧
├── 重放 - 重复发送捕获的帧
├── 注入 - 发送伪造的CAN帧
├── DoS - 发送高优先级帧
└── 模糊测试 - 发送随机数据

防御措施：
├── CAN加密（AUTOSAR SecOC）
├── 入侵检测系统（IDS）
├── 网关隔离
├── 安全CAN（CAN FD + 加密）
└── 车载防火墙
```

### 5.3 车联网安全标准

| 标准 | 说明 |
|------|------|
| ISO/SAE 21434 | 汽车网络安全工程 |
| UN R155 | 车辆网络安全管理体系 |
| UN R156 | 软件更新管理体系 |
| GB/T | 中国汽车信息安全标准 |

---

## 六、智能家居安全

### 6.1 智能家居威胁模型

```
威胁来源：
├── 外部攻击者 - 远程入侵
├── 邻近攻击者 - Wi-Fi/蓝牙攻击
├── 恶意设备 - 设备被篡改
├── 恶意应用 - APP权限滥用
└── 内部威胁 - 家庭成员

攻击目标：
├── 摄像头 - 隐私偷窥
├── 智能门锁 - 非法进入
├── 智能音箱 - 窃听
├── 智能路由器 - 流量劫持
└── 所有设备 - 僵尸网络
```

### 6.2 智能家居安全建议

```
消费者角度：
✅ 更改默认密码
✅ 定期更新固件
✅ 网络隔离（IoT专用VLAN）
✅ 禁用不需要的远程访问
✅ 关注安全公告

厂商角度：
✅ 安全开发生命周期
✅ 安全启动
✅ 加密通信
✅ 安全OTA更新
✅ 隐私设计
✅ 漏洞披露计划
```
