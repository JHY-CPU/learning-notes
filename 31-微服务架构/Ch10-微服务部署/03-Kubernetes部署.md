# Kubernetes 部署

## 一、Deployment 资源

```yaml
# order-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  namespace: microservices
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
    type: RollingUpdate
  template:
    metadata:
      labels:
        app: order-service
        version: v1
    spec:
      containers:
        - name: order-service
          image: registry.example.com/order-service:v1.2.3
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          livenessProbe:
            httpGet:
              path: /actuator/health/liveness
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /actuator/health/readiness
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
          env:
            - name: SPRING_PROFILES_ACTIVE
              value: "k8s"
            - name: NACOS_ADDR
              valueFrom:
                configMapKeyRef:
                  name: common-config
                  key: nacos-addr
```

## 二、Service 和 Ingress

```yaml
# Service
apiVersion: v1
kind: Service
metadata:
  name: order-service
spec:
  selector:
    app: order-service
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP

---
# Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: order-ingress
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /api/orders
            pathType: Prefix
            backend:
              service:
                name: order-service
                port:
                  number: 80
```

## 三、ConfigMap 和 Secret

```yaml
# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: order-config
data:
  application.yml: |
    server:
      port: 8080
    spring:
      datasource:
        url: jdbc:mysql://mysql:3306/order_db

---
# Secret
apiVersion: v1
kind: Secret
metadata:
  name: order-secret
type: Opaque
data:
  db-password: cGFzc3dvcmQxMjM=  # base64 encoded
  jwt-secret: c2VjcmV0a2V5
```

## 四、注意事项

1. **资源限制必须设置**，防止 OOM
2. **健康检查必须配置**
3. **滚动更新策略要合理**
4. **配置用 ConfigMap/Secret 管理**
5. **Namespace 隔离不同环境**
