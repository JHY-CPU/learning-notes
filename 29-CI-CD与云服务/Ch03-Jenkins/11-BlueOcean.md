# BlueOcean

## 一、概念说明

BlueOcean是Jenkins的现代化UI，提供可视化的流水线编辑和监控界面。

## 二、安装

```bash
# 插件安装
# Manage Jenkins > Manage Plugins > Available
# 搜索 "Blue Ocean" 并安装

# 访问
http://jenkins-url/blue
```

## 三、功能特点

```bash
# 1. 可视化流水线
# 图形化展示流水线各阶段

# 2. 流水线编辑器
# 可视化创建Jenkinsfile

# 3. 实时日志
# 实时查看构建日志

# 4. 分支视图
# 查看各分支的构建状态
```

## 四、创建流水线

```bash
# 通过Blue Ocean创建
# New Pipeline > GitHub/GitLab/Bitbucket
# 选择仓库
# 创建Jenkinsfile
```

## 五、注意事项

1. **兼容性**：某些插件可能不兼容Blue Ocean
2. **回退**：传统UI仍然可用
3. **性能**：大型项目可能加载较慢
4. **移动端**：Blue Ocean支持移动端访问
