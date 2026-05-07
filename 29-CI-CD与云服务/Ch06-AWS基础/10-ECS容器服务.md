# ECS容器服务

## 一、概念说明

ECS（Elastic Container Service）是AWS的容器编排服务，支持Docker容器的部署、管理和扩展。提供两种启动类型：EC2和Fargate。

| 启动类型 | 说明 | 适用场景 |
|----------|------|----------|
| EC2 | 自管理EC2实例 | 需要控制底层基础设施 |
| Fargate | 无服务器容器 | 不想管理服务器 |

| 概念 | 说明 |
|------|------|
| Cluster | 容器集群 |
| Task Definition | 任务定义模板 |
| Task | 任务运行实例 |
| Service | 长期运行的服务 |
| Container Instance | EC2容器实例 |

## 二、具体用法

### 创建集群和服务

```bash
# 创建集群
aws ecs create-cluster --cluster-name my-cluster

# 注册任务定义
aws ecs register-task-definition --cli-input-json '{
    "family": "web-app",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "256",
    "memory": "512",
    "containerDefinitions": [{
        "name": "web",
        "image": "nginx:latest",
        "portMappings": [{
            "containerPort": 80,
            "protocol": "tcp"
        }],
        "essential": true,
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": "/ecs/web-app",
                "awslogs-region": "us-east-1",
                "awslogs-stream-prefix": "web"
            }
        }
    }]
}'

# 创建服务
aws ecs create-service \
    --cluster my-cluster \
    --service-name web-service \
    --task-definition web-app \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-12345678],securityGroups=[sg-12345678],assignPublicIp=ENABLED}"
```

### Task Definition（JSON）

```json
{
    "family": "web-app",
    "containerDefinitions": [{
        "name": "web",
        "image": "myrepo/webapp:latest",
        "cpu": 256,
        "memory": 512,
        "portMappings": [{"containerPort": 8080, "hostPort": 8080}],
        "environment": [
            {"name": "DB_HOST", "value": "db.example.com"}
        ],
        "secrets": [
            {"name": "DB_PASSWORD", "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:db-pass"}
        ]
    }]
}
```

### 扩缩容操作

```bash
# 手动扩展
aws ecs update-service \
    --cluster my-cluster \
    --service web-service \
    --desired-count 4

# 配置自动扩缩容
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --resource-id service/my-cluster/web-service \
    --scalable-dimension ecs:service:DesiredCount \
    --min-capacity 2 \
    --max-capacity 10
```

### 执行命令

```bash
# 在容器中执行命令
aws ecs execute-command \
    --cluster my-cluster \
    --task abc123def456 \
    --container web \
    --command "/bin/bash" \
    --interactive
```

## 三、注意事项与常见陷阱

1. **Fargate计费**：按vCPU和内存每秒计费，注意资源分配
2. **网络配置**：Fargate需要awsvpc网络模式
3. **健康检查**：配置容器和负载均衡器健康检查
4. **日志管理**：使用awslogs驱动将日志发送到CloudWatch
5. **镜像拉取**：确保ECS有权限拉取ECR/Docker Hub镜像
6. **任务角色**：为任务分配IAM角色，避免硬编码凭证
7. **滚动更新**：服务更新时配置最小健康百分比保证可用性
