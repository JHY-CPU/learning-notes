# Crate推荐与选择

## 一、概念说明

Rust 生态有丰富的 crate，选择合适的 crate 对项目成功至关重要。

```toml
# 常用 crate 推荐
serde = "1.0"  # 序列化
tokio = "1"    # 异步运行时
reqwest = "0.11"  # HTTP 客户端
sqlx = "0.7"   # 数据库
clap = "4"     # CLI 参数解析
tracing = "0.1"  # 日志
```

## 二、具体用法

### 2.1 按场景选择

```toml
# Web 框架
axum = "0.7"  # 现代、类型安全
actix-web = "4"  # 成熟、高性能
warp = "0.3"  # 函数式风格

# 数据库
sqlx = "0.7"  # 异步、编译时检查
diesel = "2"  # ORM
sea-orm = "0.12"  # 异步 ORM

# 序列化
serde = "1.0"
serde_json = "1.0"
toml = "0.8"

# 异步
tokio = "1"
async-std = "1"
futures = "0.3"
```

### 2.2 性能考虑

```toml
# 替代标准库的高性能 crate
hashbrown = "0.14"  # HashMap 替代
parking_lot = "0.12"  # Mutex 替代
smallvec = "1"  # 小向量优化
bytes = "1"  # 高效字节缓冲
```

### 2.3 开发工具

```toml
[dev-dependencies]
criterion = "0.5"  # 基准测试
mockall = "0.12"  # Mock
proptest = "1"  # 属性测试
insta = "1"  # 快照测试

[build-dependencies]
tonic-build = "0.10"  # Proto 生成
```

### 2.4 测试 crate

```toml
[dev-dependencies]
# 单元测试
criterion = "0.5"        # 基准测试
mockall = "0.12"         # Mock
proptest = "1"           # 属性测试
insta = "1"              # 快照测试
assert_cmd = "2"         # CLI 测试
tempfile = "3"           # 临时文件
wiremock = "0.5"         # HTTP mock
```

### 2.5 按类别快速查找

```
Web 框架: axum, actix-web, warp
HTTP 客户端: reqwest, ureq
数据库: sqlx, diesel, sea-orm
序列化: serde, serde_json, toml
异步运行时: tokio, async-std
日志: tracing, log, env_logger
CLI: clap, structopt
配置: config, figment
错误处理: thiserror, anyhow
时间: chrono, time
正则: regex
加密: ring, rustls
JSON: serde_json, simd-json
```

## 四、Crate 质量评估标准

| 指标 | 权重 | 说明 |
|------|------|------|
| 下载量 | 中 | 反映流行度 |
| Star 数 | 低 | 参考价值有限 |
| 最近更新 | 高 | 活跃维护 |
| 文档质量 | 高 | 降低学习成本 |
| 依赖数量 | 中 | 影响编译时间 |
| 测试覆盖 | 高 | 代码质量 |
| MSRV | 中 | 最低 Rust 版本 |

## 五、注意事项与常见陷阱

1. **版本稳定性**：选择成熟稳定的 crate，避免 0.x 版本用于生产
2. **维护状态**：检查 crate 的维护状态，最后更新时间不应超过 1 年
3. **依赖树大小**：避免引入过多依赖，增加编译时间和攻击面
4. **许可证兼容**：检查许可证兼容性，注意 GPL 等传染性许可证
5. **文档质量**：优先选择文档完善的 crate，文档是最好的入门指南
6. **替代方案**：了解每个 crate 的替代方案，做出明智选择
7. **版本升级**：关注依赖的 breaking changes，及时升级
