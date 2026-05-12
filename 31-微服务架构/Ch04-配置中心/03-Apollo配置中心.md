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

## 三、工作原理

Apollo 由 Config Service、Admin Service、Portal 三个核心组件组成。Config Service 对外提供配置读取接口，Admin Service 管理配置修改，Portal 是管理界面。客户端启动时从 Config Service 拉取全量配置并缓存到本地文件，运行期间通过 HTTP 长轮询监听变更。Apollo 的灰度发布支持将配置变更先推送给指定 IP 的实例，验证通过后再全量发布。配置每次修改都会生成版本记录，支持一键回滚。

## 四、优缺点

**优点：**
- 功能全面：灰度发布、权限审批、版本管理、回滚一应俱全
- 客户端本地缓存降级，配置中心不可用时服务仍可启动
- 支持多环境、多集群，适合大型分布式系统

**缺点：**
- 部署组件多（3 个服务 + MySQL + Portal），运维成本高
- 相比 Nacos 组件更重，小团队可能过度设计
- 仅专注配置管理，不包含注册中心功能

## 五、最佳实践

1. 配置修改必须经过审批流程，生产环境禁止直接修改
2. 灰度发布先推送给测试实例，确认无误再全量
3. 利用 Namespace 实现公共配置和私有配置分离
4. 开启客户端本地缓存，应对配置中心短暂不可用

## 六、常见陷阱

1. **Portal 和 Config Service 网络不通**，导致配置推送失败
2. **灰度发布忘记全量发布**，部分实例一直使用旧配置
3. **配置 key 命名不规范**，不同服务之间 key 冲突
4. **长轮询数量过多**，给 Config Service 带来压力
