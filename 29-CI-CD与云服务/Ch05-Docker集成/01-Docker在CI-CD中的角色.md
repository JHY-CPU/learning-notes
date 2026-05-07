# Docker在CI-CD中的角色

## 一、概念说明

Docker为CI/CD提供一致的构建和运行环境，消除"在我机器上能运行"的问题。

## 二、Docker在CI/CD中的应用

```bash
# 1. 构建环境
# 使用Docker提供一致的构建环境

# 2. 测试环境
# 使用Docker运行测试环境

# 3. 制品管理
# Docker镜像作为部署制品

# 4. 运行环境
# 生产环境使用Docker运行应用
```

## 三、CI/CD流程

```
代码提交 → 构建Docker镜像 → 推送镜像仓库 → 部署到测试环境
    → 运行测试 → 部署到生产环境
```

## 四、基本Dockerfile

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]
```

## 五、CI中的Docker

```yaml
# GitHub Actions
- uses: docker/build-push-action@v5
  with:
    push: true
    tags: myapp:${{ github.sha }}
```

## 六、注意事项

1. **镜像大小**：使用多阶段构建减小镜像
2. **安全扫描**：构建后扫描镜像漏洞
3. **缓存利用**：利用Docker层缓存
4. **版本标签**：使用Git SHA作为镜像标签
