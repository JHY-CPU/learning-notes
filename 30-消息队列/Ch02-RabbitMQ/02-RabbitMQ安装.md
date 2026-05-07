# RabbitMQ 安装

## 一、Docker 安装（推荐）

```yaml
# docker-compose.yml
version: '3.8'
services:
  rabbitmq:
    image: rabbitmq:3.13-management
    container_name: rabbitmq
    ports:
      - "5672:5672"     # AMQP 协议端口
      - "15672:15672"   # 管理界面端口
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: admin123
      RABBITMQ_DEFAULT_VHOST: /
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    restart: unless-stopped

  # 集群部署
  rabbitmq-node1:
    image: rabbitmq:3.13-management
    hostname: rabbitmq-node1
    environment:
      RABBITMQ_ERLANG_COOKIE: "secret_cookie"
    networks:
      - rabbitmq-net

  rabbitmq-node2:
    image: rabbitmq:3.13-management
    hostname: rabbitmq-node2
    environment:
      RABBITMQ_ERLANG_COOKIE: "secret_cookie"
    networks:
      - rabbitmq-net

volumes:
  rabbitmq_data:

networks:
  rabbitmq-net:
    driver: bridge
```

```bash
# 启动
docker-compose up -d

# 查看日志
docker logs -f rabbitmq

# 进入容器
docker exec -it rabbitmq bash
```

## 二、Linux 源码安装

```bash
# Ubuntu/Debian
# 1. 安装 Erlang
sudo apt-get install erlang

# 2. 添加 RabbitMQ 仓库
curl -1sLf 'https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-server/gpg.E495BB49CC4BBE5B.key' | sudo apt-key add -
curl -1sLf 'https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-server/config.deb.txt' | sudo tee /etc/apt/sources.list.d/rabbitmq.list

# 3. 安装 RabbitMQ
sudo apt-get update
sudo apt-get install rabbitmq-server

# 4. 启动服务
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server

# 5. 启用管理插件
sudo rabbitmq-plugins enable rabbitmq_management

# 6. 创建管理员用户
sudo rabbitmqctl add_user admin admin123
sudo rabbitmqctl set_user_tags admin administrator
sudo rabbitmqctl set_permissions -p / admin ".*" ".*" ".*"
```

## 三、验证安装

```bash
# 查看状态
rabbitmqctl status

# 查看队列
rabbitmqctl list_queues

# 查看交换机
rabbitmqctl list_exchanges

# 查看绑定
rabbitmqctl list_bindings
```

```java
// Java 客户端连接测试
public class RabbitMQTest {
    public static void main(String[] args) throws Exception {
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("localhost");
        factory.setPort(5672);
        factory.setUsername("admin");
        factory.setPassword("admin123");
        factory.setVirtualHost("/");

        try (Connection connection = factory.newConnection();
             Channel channel = connection.createChannel()) {
            // 声明队列
            channel.queueDeclare("test-queue", true, false, false, null);
            // 发送消息
            channel.basicPublish("", "test-queue", null, "Hello RabbitMQ!".getBytes());
            System.out.println("消息发送成功");
        }
    }
}
```

## 四、管理界面

安装后访问 `http://localhost:15672`：

```
管理界面功能:
├── Overview    - 服务器概览、节点状态
├── Connections - 连接管理
├── Channels    - 信道管理
├── Exchanges   - 交换机管理
├── Queues      - 队列管理
├── Admin       - 用户和权限管理
└── Streams     - Stream 队列管理
```

## 五、注意事项

1. **生产环境不要使用默认用户 guest/guest**，这是仅限本地访问的内置用户
2. **Docker 部署时务必挂载数据卷**，防止容器删除后数据丢失
3. **Erlang Cookie 在集群中必须一致**，位于 `~/.erlang.cookie`
4. **管理界面不要暴露到公网**，或至少设置防火墙规则
5. **磁盘空间不足会触发流控**，默认低于 50MB 时阻塞生产者
