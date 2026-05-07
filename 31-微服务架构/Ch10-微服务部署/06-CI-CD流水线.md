# CI/CD 流水线

## 一、Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent {
        kubernetes {
            yaml '''
                spec:
                  containers:
                    - name: maven
                      image: maven:3.9-eclipse-temurin-17
                      command: sleep
                      args: ["infinity"]
                    - name: docker
                      image: docker:24-dind
                      securityContext:
                        privileged: true
            '''
        }
    }

    stages {
        stage('Build') {
            steps {
                container('maven') {
                    sh 'mvn clean package -DskipTests'
                }
            }
        }

        stage('Test') {
            parallel {
                stage('Unit Test') {
                    steps {
                        container('maven') {
                            sh 'mvn test'
                        }
                    }
                    post {
                        always {
                            junit 'target/surefire-reports/*.xml'
                        }
                    }
                }
                stage('Integration Test') {
                    steps {
                        container('maven') {
                            sh 'mvn verify -P integration-test'
                        }
                    }
                }
            }
        }

        stage('Build Image') {
            steps {
                container('docker') {
                    sh """
                        docker build -t registry.example.com/order-service:${env.BUILD_NUMBER} .
                        docker push registry.example.com/order-service:${env.BUILD_NUMBER}
                    """
                }
            }
        }

        stage('Deploy Staging') {
            steps {
                sh """
                    helm upgrade --install order-service ./helm \
                        --set image.tag=${env.BUILD_NUMBER} \
                        -n staging
                """
            }
        }

        stage('Deploy Production') {
            when {
                branch 'main'
            }
            input {
                message "确认部署到生产环境?"
                ok "确认"
            }
            steps {
                sh """
                    helm upgrade --install order-service ./helm \
                        --set image.tag=${env.BUILD_NUMBER} \
                        -n production
                """
            }
        }
    }
}
```

## 二、GitOps (ArgoCD)

```yaml
# ArgoCD Application
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: order-service
spec:
  source:
    repoURL: https://github.com/org/k8s-manifests.git
    path: order-service
    targetRevision: main
  destination:
    server: https://kubernetes.default.svc
    namespace: microservices
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

## 三、流水线阶段

```yaml
CI 阶段:
  - 代码质量检查 (SonarQube)
  - 单元测试 + 覆盖率
  - 集成测试
  - 安全扫描 (Trivy)
  - 镜像构建与推送

CD 阶段:
  - Dev 环境自动部署
  - Staging 环境自动部署
  - 生产环境审批后部署
  - 灰度发布
  - 全量发布
```

## 四、注意事项

1. **流水线即代码**，版本化管理
2. **构建产物不可变**，同一镜像各环境复用
3. **安全扫描集成到流水线**
4. **GitOps 模式保证环境一致性**
5. **失败要有通知机制**
