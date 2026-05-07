# Jenkins概述

## 一、概念说明

Jenkins是开源的自动化服务器，用于CI/CD。支持流水线、插件扩展、分布式构建等功能。

## 二、安装

```bash
# Docker安装
docker run -d --name jenkins \
  -p 8080:8080 -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  jenkins/jenkins:lts

# 获取初始密码
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

## 三、核心概念

```bash
# Job（任务）
# 一个构建任务

# Pipeline（流水线）
# 用代码定义的CI/CD流程

# Node（节点）
# 执行构建的服务器

# Stage（阶段）
# 流水线的一个阶段

# Step（步骤）
# 单个构建步骤
```

## 四、基本配置

```bash
# 1. 安装推荐插件
# 2. 创建管理员用户
# 3. 配置Jenkins URL
# 4. 安装必要插件：Git、Pipeline、Docker
```

## 五、注意事项

1. **安全加固**：配置认证和授权
2. **插件管理**：定期更新插件
3. **备份**：定期备份JENKINS_HOME
4. **资源管理**：监控Jenkins服务器资源
