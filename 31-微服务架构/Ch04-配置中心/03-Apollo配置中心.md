# Apollo 配置中心

## 一、Apollo 核心功能

```
Apollo 功能:
├── 统一管理 - 多环境配置
├── 实时推送 - 配置变更秒级生效
├── 版本管理 - 配置回滚
├� 灰度发布 - 按比例发布配置
└── 权限管理 - 配置修改审批
```

## 二、Spring Boot 集成

```yaml
app:
  id: user-service
apollo:
  meta: http://apollo-config:8080
  bootstrap:
    enabled: true
    namespaces: application
```

## 三、注意事项

1. **Apollo 功能全面**，适合大型团队
2. **灰度发布是核心优势**
3. **配置修改需要审批**
4. **版本管理支持回滚**
5. **国内公司使用广泛**
