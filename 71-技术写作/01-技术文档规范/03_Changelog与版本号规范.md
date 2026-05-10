# Changelog与版本号规范


## Changelog与版本号规范


技术写作版本管理Changelog


规范的版本号和Changelog是开源项目专业性的体现。


## 语义化版本号 (SemVer)


```
格式：MAJOR.MINOR.PATCH（主版本号.次版本号.修订号）

规则：
- MAJOR (主版本)：不兼容的API变更
- MINOR (次版本)：向后兼容的功能新增
- PATCH (修订号)：向后兼容的问题修复

先行版本号：
- 1.0.0-alpha.1：内部测试版
- 1.0.0-beta.1：公开测试版
- 1.0.0-rc.1：候选发布版 (Release Candidate)
- 1.0.0：正式发布版

构建元数据：
- 1.0.0+20240115.sha.abc1234

版本号示例：
┌────────────────────────────────────────────┐
│ 0.1.0 → 0.2.0：新功能，API可能变化         │
│ 0.9.0 → 1.0.0：API稳定，正式发布           │
│ 1.0.0 → 1.0.1：Bug修复                     │
│ 1.0.1 → 1.1.0：新功能，向后兼容            │
│ 1.1.0 → 2.0.0：破坏性变更                  │
└────────────────────────────────────────────┘

常见错误：
- 0.x.x阶段不要过度解读兼容性
- 不要跳过版本号（1.0直接到3.0）
- 不要把日期当版本号（20240115）
- Pre-1.0的项目API不稳定，谨慎使用
```


## Keep a Changelog 格式


```
原则：
1. Changelog是给人写的，不是给机器
2. 每个版本的所有显著变化都应记录
3. 相同类型的变化应分组
4. 版本和日期应清晰标注
5. 最新版本在最上面

标准分类（英文关键词）：
- Added (新增)：新功能
- Changed (变更)：已有功能的变更
- Deprecated (废弃)：即将移除的功能
- Removed (移除)：已移除的功能
- Fixed (修复)：Bug修复
- Security (安全)：安全漏洞修复

示例模板：
# Changelog

All notable changes to this project will be documented in this file.

## [2.1.0] - 2024-03-15

### Added
- 新增批量导出功能，支持CSV和JSON格式
- 新增用户偏好设置API (GET/PUT /api/preferences)

### Changed
- 优化搜索算法，响应时间降低40%
- 升级依赖 Spring Boot 3.1 → 3.2

### Fixed
- 修复高并发下连接池泄漏问题 (#1234)
- 修复时区转换导致的日期显示错误 (#1245)

### Security
- 升级 log4j 至 2.21.0 修复CVE-2024-XXXX

## [2.0.1] - 2024-02-28

### Fixed
- 修复迁移脚本中的数据丢失问题

## [2.0.0] - 2024-02-15

### Breaking Changes
- 移除已废弃的v1 API端点
- 最低Java版本提升至17

### Added
- 新增WebSocket实时通知功能
```


## Release Notes 最佳实践


```
Release Notes vs Changelog：
┌──────────┬──────────────────┬──────────────────┐
│ 维度       │ Changelog         │ Release Notes     │
├──────────┼──────────────────┼──────────────────┤
│ 受众       │ 开发者             │ 所有用户           │
│ 详细程度   │ 细节（PR级别）      │ 概述（功能级别）   │
│ 风格       │ 技术性             │ 营销+技术         │
│ 自动生成   │ 可以              │ 需人工润色         │
│ 格式       │ Markdown          │ 博客/邮件格式      │
└──────────┴──────────────────┴──────────────────┘

Release Notes 结构：
1. 发布亮点（What's New）
   - 3-5个最重要的变化
   - 用户视角描述价值

2. 功能详情
   - 按模块/领域分组
   - 配截图或GIF动画

3. 破坏性变更（Breaking Changes）
   - 明确列出不兼容变更
   - 提供迁移指南

4. 已知问题
   - 已发现但未修复的问题
   - 临时解决方案

5. 致谢
   - 感谢社区贡献者

自动化工具：
- standard-version：自动从conventional commits生成
- release-please：Google开源，自动PR
- semantic-release：全自动版本发布流程
- git-cliff：Rust编写，可配置的changelog生成

Conventional Commits格式：
feat: 新增批量导出功能 (#123)
fix: 修复连接池泄漏 (#124)
docs: 更新API文档
chore: 升级依赖版本
BREAKING CHANGE: 移除v1 API
```


> **Note:** 语义化版本号(SemVer)告诉用户升级是否安全，Changelog记录每个版本的变化细节，Release Notes用用户友好的语言讲述版本故事。三者配合，构成完整的版本沟通体系。


<!-- Converted from: 03_Changelog与版本号规范.html -->
