# Serverless 部署

## 一、Knative

```yaml
# Knative Service
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: order-service
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "100"
        autoscaling.knative.dev/target: "100"
    spec:
      containers:
        - image: registry.example.com/order-service:v1
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          env:
            - name: SPRING_PROFILES_ACTIVE
              value: "knative"
```

## 二、AWS Lambda

```java
// Spring Cloud Function
@SpringBootApplication
public class OrderFunction {

    @Bean
    public Function<OrderRequest, Order> createOrder() {
        return request -> {
            Order order = new Order(request.getCustomerId(), request.getAmount());
            // 业务逻辑
            return order;
        };
    }

    @Bean
    public Function<OrderId, Order> getOrder() {
        return orderId -> orderRepository.findById(orderId).orElseThrow();
    };
}

# serverless.yml
service: order-service
provider:
  name: aws
  runtime: java17
  region: ap-east-1
  memorySize: 512
  timeout: 30

functions:
  createOrder:
    handler: org.springframework.cloud.function.adapter.aws.FunctionInvoker
    events:
      - http:
          path: /orders
          method: post
  getOrder:
    handler: org.springframework.cloud.function.adapter.aws.FunctionInvoker
    events:
      - http:
          path: /orders/{id}
          method: get
```

## 三、阿里云函数计算

```yaml
# Serverless Devs 配置
edition: 3.0.0
name: order-service
access: default

resources:
  order-function:
    component: fc3
    props:
      region: cn-hangzhou
      functionName: order-service
      runtime: java17
      handler: org.springframework.cloud.function.adapter.ali.AliSpringBootRequestHandler
      timeout: 30
      memorySize: 512
      code: ./target/order-service-1.0.0.jar
      triggers:
        - triggerName: http-trigger
          triggerType: http
          triggerConfig:
            authType: anonymous
            methods: [GET, POST]
```

## 四、注意事项

1. **冷启动是 Serverless 最大挑战**
2. **无状态设计是前提**
3. **执行时间有限制**
4. **成本按调用量计费**，注意成本控制
5. **适合事件驱动和间歇性任务**
