# DaemonSet与Job

## 一、概念说明

DaemonSet 确保每个节点运行一个 Pod，Job 运行一次性任务，CronJob 运行定时任务。

## 二、具体用法

### 2.1 DaemonSet

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      tolerations:
      - key: node-role.kubernetes.io/control-plane
        effect: NoSchedule
      containers:
      - name: fluentd
        image: fluent/fluentd:v1.16
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: containers
          mountPath: /var/lib/docker/containers
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: containers
        hostPath:
          path: /var/lib/docker/containers
```

### 2.2 Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-migration
spec:
  completions: 1            # 需要成功完成的次数
  parallelism: 1            # 并行运行的 Pod 数
  backoffLimit: 3           # 失败重试次数
  activeDeadlineSeconds: 600  # 最大运行时间
  template:
    spec:
      restartPolicy: Never    # Job 必须用 Never 或 OnFailure
      containers:
      - name: migration
        image: my-migration:v1
        command: ["python", "migrate.py"]
```

### 2.3 并行 Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: parallel-job
spec:
  completions: 10           # 需要完成 10 次
  parallelism: 3            # 每次运行 3 个 Pod
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: worker
        image: my-worker:v1
        env:
        - name: JOB_COMPLETION_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['batch.kubernetes.io/job-completion-index']
```

### 2.4 CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: backup
spec:
  schedule: "0 2 * * *"     # 每天凌晨 2 点
  concurrencyPolicy: Forbid  # 不允许并发
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: Never
          containers:
          - name: backup
            image: my-backup:v1
            command: ["sh", "-c", "pg_dump > /backup/dump.sql"]
```

### 2.5 CronJob 调度选项

```yaml
spec:
  schedule: "*/5 * * * *"     # 每 5 分钟
  schedule: "0 9-17 * * MON-FRI"  # 工作日 9-17 点每小时
  schedule: "@every 1h"       # 每小时
  schedule: "@daily"           # 每天
  concurrencyPolicy: Allow     # 允许并发（默认）
  concurrencyPolicy: Forbid    # 跳过并发任务
  concurrencyPolicy: Replace   # 替换正在运行的任务
```

## 三、注意事项与常见陷阱

1. **DaemonSet 节点选择**：注意 master/worker 节点的 taint/toleration
2. **Job 重启策略**：Job 不能用 Always，必须用 Never 或 OnFailure
3. **CronJob 并发**：注意并发策略，避免重复执行
4. **日志保留**：completed 的 Job/Pod 会保留，注意清理
5. **时区**：CronJob 使用 kube-controller-manager 时区
