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

## 六、Docker Compose完整部署

```yaml
# docker-compose.yml
version: '3.8'
services:
  jenkins:
    image: jenkins/jenkins:lts-jdk17
    container_name: jenkins
    restart: always
    ports:
      - "8080:8080"
      - "50000:50000"
    volumes:
      - jenkins_home:/var/jenkins_home
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - JAVA_OPTS=-Djenkins.install.runSetupWizard=false
      - JENKINS_OPTS=--prefix=/jenkins
    networks:
      - jenkins-net

volumes:
  jenkins_home:

networks:
  jenkins-net:
    driver: bridge
```

```bash
# 初始化安装
docker-compose up -d
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword

# 配置As-Code（Jenkins Configuration as Code）
# 创建jenkins.yaml
docker run -d \
  -e CASC_JENKINS_CONFIG=/var/jenkins_home/jenkins.yaml \
  -v $(pwd)/jenkins.yaml:/var/jenkins_home/jenkins.yaml \
  -p 8080:8080 -p 50000:50000 \
  jenkins/jenkins:lts
```

## 七、Jenkins Configuration as Code (JCasC)

```yaml
# jenkins.yaml
jenkins:
  systemMessage: "Jenkins configured via JCasC"
  numExecutors: 0
  mode: EXCLUSIVE
  securityRealm:
    local:
      allowsSignup: false
      users:
        - id: admin
          password: "${JENKINS_ADMIN_PASSWORD}"
  authorizationStrategy:
    roleBased:
      roles:
        global:
          - name: admin
            permissions:
              - Overall/Administer
            entries:
              - user: admin
          - name: developer
            permissions:
              - Overall/Read
              - Job/Build
              - Job/Read

credentials:
  system:
    domainCredentials:
      - credentials:
          - string:
              scope: GLOBAL
              id: github-token
              secret: "${GITHUB_TOKEN}"

unclassified:
  location:
    url: http://jenkins.example.com/
```

## 八、Jenkins REST API

```bash
# 获取构建信息
curl -u admin:token http://jenkins-url/job/myjob/1/api/json

# 触发构建
curl -X POST -u admin:token http://jenkins-url/job/myjob/build

# 参数化构建
curl -X POST -u admin:token \
  "http://jenkins-url/job/myjob/buildWithParameters?VERSION=1.0.0&ENV=prod"

# 获取控制台输出
curl -u admin:token http://jenkins-url/job/myjob/1/consoleText

# 创建Job
curl -X POST -u admin:token \
  http://jenkins-url/createItem?name=newjob \
  -H "Content-Type: application/xml" \
  --data-binary @config.xml
```
