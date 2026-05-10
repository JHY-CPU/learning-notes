# 2025-02-28 prometheus面试题

URL: https://qq547475331.github.io/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/

## Prometheus [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#prometheus)

## Prometheus的工作流程 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#prometheus%e7%9a%84%e5%b7%a5%e4%bd%9c%e6%b5%81%e7%a8%8b)

Prometheus 的工作流程可以概括为以下几个主要步骤：

1. **数据抓取（Scraping）**：
   - Prometheus 会定期从配置好的目标（如应用程序、服务、节点等）抓取数据。这些目标通常通过 HTTP 接口暴露 Prometheus 格式的监控数据（通常是 `/metrics` 路径）。
   - 这些数据包括各类指标，如 CPU 使用率、内存使用情况、请求数量等。
2. **存储（Storage）**：
   - 抓取到的数据会被存储在 Prometheus 的本地时间序列数据库（TSDB）中。每个时间序列由一个指标名称和一组标签（如 `instance`, `job`, `region` 等）组成。
3. **查询（Querying）**：
   - Prometheus 提供了强大的查询语言——PromQL（Prometheus Query Language），可以通过 PromQL 查询已存储的数据。
   - Prometheus 可以通过其 Web 界面、API 或 Grafana 进行查询，显示时间序列数据的图表，或者用于告警规则的计算。
4. **告警（Alerting）**：
   - Prometheus 可以基于定义的告警规则（通过 PromQL 查询语句）进行告警。例如，当 CPU 使用率超过一定阈值时，触发告警。
   - 告警规则可以定义在 Prometheus 配置文件中，告警信息可以通过 Alertmanager 发送到各类通知系统（如邮件、Slack、钉钉等）。
5. **数据可视化（Visualization）**：
   - Prometheus 本身也提供了简单的图表功能，但通常会与第三方工具（如 Grafana）结合使用，以便提供更丰富的可视化效果。
   - Grafana 可以从 Prometheus 查询数据，生成漂亮的仪表板，帮助团队实时监控系统健康状况。
6. **服务发现与目标自动化（Service Discovery）**：
   - Prometheus 支持多种服务发现机制，可以自动发现需要抓取数据的目标。常见的有 Kubernetes、Consul 等服务发现工具，也可以使用静态配置。

整体而言，Prometheus 的核心是周期性地抓取目标的监控数据，存储成时间序列，并通过查询、告警和可视化等功能帮助用户监控和维护系统的健康。

如果你有某个具体环节的疑问，欢迎随时问！

## Metric的几种类型？分别是什么？☆ [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#metric%e7%9a%84%e5%87%a0%e7%a7%8d%e7%b1%bb%e5%9e%8b%e5%88%86%e5%88%ab%e6%98%af%e4%bb%80%e4%b9%88)

Prometheus 中有四种主要的 **Metric（度量）类型**，分别是：

1. **Counter（计数器）**：


   - **定义**：计数器是一种只增不减的度量类型。它表示一个累积值，比如请求的总数、处理的任务数等。
   - **特点**：它的值始终是递增的，并且可以在某些情况下被重置（如重启应用）。
   - **例子**：HTTP 请求的总数、错误发生次数等。
   - **常见用途**：表示某种事件或状态发生的总次数。

```prometheus
http_requests_total{method="GET", status="200"}
```

2. **Gauge（仪表）**：


   - **定义**：仪表度量类型表示一个可以任意增减的值。它适用于那些可能会增加或减少的度量，比如当前内存使用量、CPU 使用率等。
   - **特点**：它的值可以上升或下降，适合表示某些瞬时的、可变的数值。
   - **例子**：当前温度、内存使用、磁盘空间等。
   - **常见用途**：表示某些动态的、瞬时的数值。

```prometheus
memory_usage_bytes
```

3. **Histogram（直方图）**：


   - **定义**：直方图用于记录一系列的分布数据，按指定的桶（bucket）来聚合样本数据。它能够提供数据的分布情况（如请求延迟的分布），并可以通过统计数据计算出一些聚合指标（如平均值、百分位数等）。
   - **特点**：它会提供桶内的计数值，可以计算出值的分布（如请求延迟的 95th 百分位）。
   - **例子**：HTTP 请求的响应时间，数据库查询时间等。
   - **常见用途**：分布数据的度量，例如请求延迟、处理时间等。

```prometheus
http_request_duration_seconds_bucket{le="0.1"}
```

4. **Summary（摘要）**：


   - **定义**：摘要与直方图类似，都是用来统计分布数据的。不同之处在于，摘要记录的是在某一时间段内发生的样本的摘要统计信息，如累计总数、累计和、最大值等，而直方图则通过分桶统计数据分布。
   - **特点**：摘要度量提供了直接的统计值，如百分位数（p50, p90, p99等），不需要根据桶计算。
   - **例子**：请求的延迟百分位数，API 响应时间等。
   - **常见用途**：需要精确计算百分位数和其他统计量时，如请求响应时间的分位数。

```prometheus
http_request_duration_seconds{quantile="0.95"}
```

### 总结： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%80%bb%e7%bb%93)

- **Counter**：只增不减，表示累计总数。
- **Gauge**：可以增减，表示当前状态的瞬时值。
- **Histogram**：用于表示数据的分布，按桶聚合数据。
- **Summary**：类似于直方图，但直接提供百分位数等统计信息。

这四种类型是 Prometheus 度量的核心构成，它们可以帮助你监控系统的不同方面，并从多个角度分析数据。如果有某个度量类型的具体应用场景或者细节需要进一步了解，随时告诉我！

## Prometheus有哪几种服务发现☆ [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#prometheus%e6%9c%89%e5%93%aa%e5%87%a0%e7%a7%8d%e6%9c%8d%e5%8a%a1%e5%8f%91%e7%8e%b0)

Prometheus 提供了多种服务发现（Service Discovery）机制，使其能够自动发现并抓取目标（如应用、服务器等）的监控数据。服务发现可以大大简化配置管理，尤其是在动态环境中（如 Kubernetes 或云平台）非常有用。Prometheus 支持以下几种服务发现方式：

### 1\. **Kubernetes 服务发现**： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-kubernetes-%e6%9c%8d%e5%8a%a1%e5%8f%91%e7%8e%b0)

- **描述**：在 Kubernetes 环境中，Prometheus 可以通过 Kubernetes API 自动发现集群中的服务、Pod 和节点。它能够发现 K8s 中所有符合条件的目标（如通过标签选择器、命名空间等进行筛选）。
- **配置**：Prometheus 配置文件中使用 `kubernetes_sd_configs` 配置项来实现。
- **应用场景**：适用于容器化的环境，尤其是 Kubernetes 集群中的服务发现。

```yaml
scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - api_server: 'https://k8s-api-server'
        role: pod
```

### 2\. **Consul 服务发现**： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-consul-%e6%9c%8d%e5%8a%a1%e5%8f%91%e7%8e%b0)

- **描述**：Prometheus 支持通过 Consul 进行服务发现。Consul 是一个服务网格工具，它可以管理服务注册和发现。Prometheus 会从 Consul 获取服务实例列表并进行抓取。
- **配置**：通过 `consul_sd_configs` 配置项。
- **应用场景**：适用于使用 Consul 作为服务注册和发现的环境。

```yaml
scrape_configs:
  - job_name: 'consul'
    consul_sd_configs:
      - server: 'localhost:8500'
        services: ['my-service']
```

### 3\. **EC2 服务发现**： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-ec2-%e6%9c%8d%e5%8a%a1%e5%8f%91%e7%8e%b0)

- **描述**：在 AWS 环境中，Prometheus 可以通过 AWS EC2 服务发现，自动抓取 EC2 实例的指标。它通过 AWS API 获取当前运行的 EC2 实例信息，基于标签、实例状态等进行筛选。
- **配置**：通过 `ec2_sd_configs` 配置项来实现。
- **应用场景**：适用于在 AWS EC2 上运行的服务或实例。

```yaml
scrape_configs:
  - job_name: 'ec2'
    ec2_sd_configs:
      - region: 'us-west-2'
        access_key: 'AWS_ACCESS_KEY'
        secret_key: 'AWS_SECRET_KEY'
```

### 4\. **GCE（Google Compute Engine）服务发现**： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-gcegoogle-compute-engine%e6%9c%8d%e5%8a%a1%e5%8f%91%e7%8e%b0)

- **描述**：Prometheus 可以在 Google Cloud 环境中使用 GCE 服务发现来自动发现 Google Compute Engine 实例。
- **配置**：通过 `gce_sd_configs` 配置项来实现。
- **应用场景**：适用于 Google Cloud 平台的虚拟机实例。

```yaml
scrape_configs:
  - job_name: 'gce'
    gce_sd_configs:
      - project: 'my-gcp-project'
        zone: 'us-central1-a'
```

### 5\. **Azure 服务发现**： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#5-azure-%e6%9c%8d%e5%8a%a1%e5%8f%91%e7%8e%b0)

- **描述**：Prometheus 支持在 Azure 环境中自动发现虚拟机、虚拟机规模集等服务实例。通过 Azure API 获取实例的 IP 地址、标签等信息。
- **配置**：通过 `azure_sd_configs` 配置项来实现。
- **应用场景**：适用于在 Azure 云平台上部署的服务。

```yaml
scrape_configs:
  - job_name: 'azure'
    azure_sd_configs:
      - subscription_id: 'your-subscription-id'
        tenant_id: 'your-tenant-id'
        client_id: 'your-client-id'
        client_secret: 'your-client-secret'
```

### 6\. **DNS 服务发现**： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#6-dns-%e6%9c%8d%e5%8a%a1%e5%8f%91%e7%8e%b0)

- **描述**：Prometheus 可以使用 DNS 查询来发现目标。通过配置 DNS 名称解析获取需要监控的目标。
- **配置**：通过 `dns_sd_configs` 配置项。
- **应用场景**：适用于使用 DNS 进行动态服务发现的环境。

```yaml
scrape_configs:
  - job_name: 'dns'
    dns_sd_configs:
      - names:
          - 'myservice.local'
        type: 'A'
        port: 8080
```

### 7\. **File 服务发现**： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#7-file-%e6%9c%8d%e5%8a%a1%e5%8f%91%e7%8e%b0)

- **描述**：Prometheus 可以通过读取外部文件（如 JSON 或 YAML 格式）来进行静态服务发现。文件中的目标可以动态更新，Prometheus 会定期重新加载这些文件。
- **配置**：通过 `file_sd_configs` 配置项。
- **应用场景**：适用于较为静态的环境，或者当目标实例列表存储在外部文件中的情况。

```yaml
scrape_configs:
  - job_name: 'file'
    file_sd_configs:
      - files:
          - '/path/to/targets.json'
```

### 8\. **Static 服务发现**： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#8-static-%e6%9c%8d%e5%8a%a1%e5%8f%91%e7%8e%b0)

- **描述**：Prometheus 也支持通过静态配置来发现目标。这种方式适用于目标列表不频繁变化的场景。
- **配置**：通过 `static_configs` 配置项，可以手动列出目标。
- **应用场景**：适用于目标数量较少且较为固定的环境。

```yaml
scrape_configs:
  - job_name: 'static'
    static_configs:
      - targets: ['localhost:9090', 'localhost:8080']
```

### 9\. **OpenStack 服务发现**： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#9-openstack-%e6%9c%8d%e5%8a%a1%e5%8f%91%e7%8e%b0)

- **描述**：Prometheus 支持 OpenStack 环境中的服务发现。通过与 OpenStack API 交互，自动发现相关的服务实例和节点。
- **配置**：通过 `openstack_sd_configs` 配置项来实现。
- **应用场景**：适用于 OpenStack 环境的服务发现。

```yaml
scrape_configs:
  - job_name: 'openstack'
    openstack_sd_configs:
      - auth_url: 'https://openstack.example.com'
        username: 'username'
        password: 'password'
        project_name: 'project'
        region: 'region'
```

### 总结： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%80%bb%e7%bb%93-1)

Prometheus 支持多种服务发现机制，涵盖了云平台、容器平台、传统服务器和静态配置等多种场景。你可以根据自己的环境选择合适的服务发现方式，帮助 Prometheus 自动发现和抓取目标。

如果你对某种服务发现方式有更多问题或需要具体示例，欢迎继续提问！

## Prometheus常用函数 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#prometheus%e5%b8%b8%e7%94%a8%e5%87%bd%e6%95%b0)

Prometheus 提供了丰富的函数和操作符，帮助用户在 PromQL 查询语言中处理和分析时间序列数据。以下是一些常用的 Prometheus 函数：

### 1\. **聚合函数（Aggregation Functions）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-%e8%81%9a%e5%90%88%e5%87%bd%e6%95%b0aggregation-functions)

聚合函数用于按某些维度聚合数据，通常用于将多个时间序列合并为一个更具概括性的度量。

- `avg()`

：计算平均值


```prometheus
avg(http_requests_total)
```

- `sum()`

：计算总和


```prometheus
sum(http_requests_total)
```

- `min()`

：计算最小值


```prometheus
min(http_requests_total)
```

- `max()`

：计算最大值


```prometheus
max(http_requests_total)
```

- `count()`

：计算数量


```prometheus
count(http_requests_total)
```

- `count_values()`

：计算不同标签值的数量


```prometheus
count_values("method", http_requests_total)
```

- `topk(k, expr)`

：返回值排名前


```
k
```


的时间序列


```prometheus
topk(3, http_requests_total)
```

- `bottomk(k, expr)`

：返回值排名后


```
k
```


的时间序列


```prometheus
bottomk(3, http_requests_total)
```


### 2\. **统计函数（Statistical Functions）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-%e7%bb%9f%e8%ae%a1%e5%87%bd%e6%95%b0statistical-functions)

- `rate()`

：计算时间序列在某个时间段内的速率（适用于计数器类型）


```prometheus
rate(http_requests_total[5m])
```

- `irate()`

：计算瞬时速率，适用于计数器类型


```prometheus
irate(http_requests_total[1m])
```

- `increase()`

：计算计数器类型的增量值


```prometheus
increase(http_requests_total[1h])
```

- `delta()`

：计算某时间段内的值的变化量


```prometheus
delta(http_requests_total[5m])
```

- `avg_over_time()`

：计算某个时间区间内的平均值


```prometheus
avg_over_time(http_requests_total[1h])
```

- `min_over_time()`

：计算某个时间区间内的最小值


```prometheus
min_over_time(http_requests_total[1h])
```

- `max_over_time()`

：计算某个时间区间内的最大值


```prometheus
max_over_time(http_requests_total[1h])
```


### 3\. **时间函数（Time Functions）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-%e6%97%b6%e9%97%b4%e5%87%bd%e6%95%b0time-functions)

时间函数用于操作时间戳和时间序列。

- `time()`

：返回当前的时间戳（以秒为单位）


```prometheus
time()
```

- `timestamp()`

：返回时间序列的时间戳


```prometheus
timestamp(http_requests_total)
```


### 4\. **数值处理函数（Mathematical Functions）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-%e6%95%b0%e5%80%bc%e5%a4%84%e7%90%86%e5%87%bd%e6%95%b0mathematical-functions)

这些函数用于进行常见的数学运算。

- `abs()`

：返回绝对值


```prometheus
abs(http_requests_total)
```

- `ceil()`

：返回大于或等于该值的最小整数


```prometheus
ceil(http_requests_total)
```

- `floor()`

：返回小于或等于该值的最大整数


```prometheus
floor(http_requests_total)
```

- `round()`

：返回四舍五入的值


```prometheus
round(http_requests_total)
```

- `sqrt()`

：计算平方根


```prometheus
sqrt(http_requests_total)
```


### 5\. **字符串处理函数（String Functions）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#5-%e5%ad%97%e7%ac%a6%e4%b8%b2%e5%a4%84%e7%90%86%e5%87%bd%e6%95%b0string-functions)

字符串函数主要用于对标签值或字符串型数据进行处理。

- `label_replace()`

：用于替换标签中的部分内容


```prometheus
label_replace(http_requests_total, "method", "$1", "status", "(.*)")
```

- `label_join()`

：将多个标签值连接为一个新标签


```prometheus
label_join(http_requests_total, "method", "-", "status", "code")
```


### 6\. **查询时间函数（Time Interval Functions）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#6-%e6%9f%a5%e8%af%a2%e6%97%b6%e9%97%b4%e5%87%bd%e6%95%b0time-interval-functions)

这些函数用于处理时间区间。

- `avg_over_time()`

：计算某个时间范围内的平均值


```prometheus
avg_over_time(http_requests_total[1h])
```

- `rate()`

：计算在某段时间内的速率


```prometheus
rate(http_requests_total[5m])
```


### 7\. **率函数（Rate Functions）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#7-%e7%8e%87%e5%87%bd%e6%95%b0rate-functions)

用于计算单位时间内的数据变化量。

- `rate()`

：适用于计数器类型，计算单位时间内的变化速率


```prometheus
rate(http_requests_total[5m])
```

- `irate()`

：即时速率，适用于计数器类型，计算最近两个数据点的速率


```prometheus
irate(http_requests_total[1m])
```


### 8\. **逻辑函数（Logical Functions）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#8-%e9%80%bb%e8%be%91%e5%87%bd%e6%95%b0logical-functions)

这些函数帮助进行逻辑判断和操作。

- `and`

：用于两个表达式的与操作


```prometheus
http_requests_total{status="200"} and http_requests_total{status="500"}
```

- `or`

：用于两个表达式的或操作


```prometheus
http_requests_total{status="200"} or http_requests_total{status="404"}
```

- `unless`

：排除某些条件


```prometheus
http_requests_total unless http_requests_total{status="500"}
```


### 9\. **过滤与条件（Filters & Conditionals）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#9-%e8%bf%87%e6%bb%a4%e4%b8%8e%e6%9d%a1%e4%bb%b6filters--conditionals)

- `on()`

：用于条件匹配，可以指定某些标签参与匹配


```prometheus
http_requests_total{status="200"} on(method) http_requests_total{status="500"}
```

- `ignoring()`

：忽略特定的标签进行条件匹配


```prometheus
http_requests_total{status="200"} ignoring(method) http_requests_total{status="500"}
```


### 10\. **百分比计算（Percentile Calculation）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#10-%e7%99%be%e5%88%86%e6%af%94%e8%ae%a1%e7%ae%97percentile-calculation)

- `histogram_quantile()`

：用于计算直方图的指定分位数（如 95th 百分位）


```prometheus
histogram_quantile(0.95, http_request_duration_seconds_bucket)
```


### 总结： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%80%bb%e7%bb%93-2)

Prometheus 提供了多种函数来帮助你处理和分析监控数据。你可以通过这些函数来进行聚合、时间序列计算、统计分析以及进行复杂的查询和可视化。

如果你对某个函数或用法有疑问，随时可以问我！

## thanos架构☆ [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#thanos%e6%9e%b6%e6%9e%84)

**Thanos** 是一个用于扩展 **Prometheus** 的高可用性、长期存储和全球查询的解决方案。它通过无缝地集成 Prometheus 和其他组件，提供了分布式、可扩展的监控架构，特别适用于大规模部署和多集群环境。

### Thanos 架构概述 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#thanos-%e6%9e%b6%e6%9e%84%e6%a6%82%e8%bf%b0)

Thanos 的架构设计基于以下几个核心组件：

1. **Prometheus**：

   - **作用**：Prometheus 仍然是数据的抓取和存储核心，负责从各类目标（如服务、应用、节点等）抓取时间序列数据。
   - **特点**：Thanos 并不替代 Prometheus，而是增强其功能，特别是在高可用、长期存储和多集群管理方面。
2. **Thanos Sidecar**：


   - 作用

     ：Thanos Sidecar 是运行在 Prometheus 实例旁边的一个代理，它负责：

     - 将 Prometheus 中的时间序列数据上传到远程存储（如 S3、GCS、HDFS 等）。
     - 将 Prometheus 数据暴露给 Thanos Query 和其他组件。
   - **特点**：Sidecar 是 Thanos 架构中的关键组件之一，它帮助 Prometheus 实现持久化存储，并且为 Thanos Query 提供数据源。


**Sidecar 配置**：

```yaml
- --tsdb.path=/prometheus
- --objstore.config="type: S3\nconfig:\n  bucket: my-bucket"
```

3. **Thanos Store**：

   - **作用**：Thanos Store 是一个存储组件，负责从远程对象存储（如 S3、GCS 等）读取历史数据。它充当了 Prometheus 数据的长期存储。
   - **特点**：Store 主要用于查询过去的历史数据，并提供了统一的接口来访问不同时间窗口内的数据。
4. **Thanos Query**：

   - **作用**：Thanos Query 是一个用于聚合查询的组件，它允许跨多个 Prometheus 和 Thanos 存储实例执行全局查询。它从多个 Prometheus 实例和 Thanos Store 获取数据，并提供统一的查询结果。
   - **特点**：Thanos Query 实现了高效的分布式查询，支持从不同 Prometheus 实例和 Store 中读取数据，可以跨多个集群或多数据源执行联合查询。
5. **Thanos Compact**：

   - **作用**：Thanos Compact 负责合并、压缩和优化存储在对象存储中的时间序列数据。它将多个时间块（Block）合并为更大的存储块，以减少存储碎片并优化查询性能。
   - **特点**：它定期运行，以确保数据在存储层面的优化和压缩。
6. **Thanos Ruler**：

   - **作用**：Thanos Ruler 是基于 Prometheus 的规则引擎，它允许你在 Thanos 集群中运行 Prometheus 规则、告警规则和录制规则。
   - **特点**：它提供了一种在全局范围内执行告警和规则评估的机制。Thanos Ruler 支持在多个 Prometheus 实例之间共享规则和告警信息。

### Thanos 架构图 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#thanos-%e6%9e%b6%e6%9e%84%e5%9b%be)

```
+----------------------+       +----------------------+       +----------------------+
|    Prometheus        |<----->|    Thanos Sidecar    |<----->|  Remote Object Store |
+----------------------+       +----------------------+       +----------------------+
        |                            |                          |
        |                            v                          v
        |                      +--------------------+       +------------------+
        |                      |   Thanos Query     |<----->|    Thanos Store  |
        |                      +--------------------+       +------------------+
        |                               |
        |                               v
        |                        +------------------+
        |                        |   Thanos Ruler   |
        |                        +------------------+
        |
        v
   +------------------+
   | Thanos Compact   |
   +------------------+
```

### Thanos 核心功能 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#thanos-%e6%a0%b8%e5%bf%83%e5%8a%9f%e8%83%bd)

1. **高可用性和水平扩展**：
   - Thanos 提供跨多个 Prometheus 实例和多个集群的查询能力，帮助实现 Prometheus 集群的高可用性。
   - 多个 Prometheus 实例可以分别在不同的区域或集群中部署，Thanos Query 可以将其聚合成一个全局视图。
2. **长期存储**：
   - 通过将数据持久化到远程对象存储（如 S3、GCS、Azure Blob 等），Thanos 提供了可靠且成本效益高的长期存储解决方案。
   - 对历史数据的访问可以通过 Thanos Store 组件来查询。
3. **全球查询**：
   - Thanos Query 提供全局视图，支持跨多个集群或 Prometheus 实例执行统一查询。
   - 即使数据分布在多个 Prometheus 实例上，Thanos Query 也能通过聚合各个数据源的结果，提供统一的查询接口。
4. **压缩与存储优化**：
   - Thanos Compact 功能可以将存储中的多个数据块合并和压缩，降低存储成本，同时提升查询效率。
5. **规则和告警**：
   - Thanos Ruler 允许在 Thanos 集群中执行 Prometheus 规则和告警，并支持跨 Prometheus 实例共享规则和告警配置。
6. **数据流控制**：
   - 使用 Thanos Sidecar，用户可以将 Prometheus 数据推送到远程存储，也可以实现多个 Prometheus 实例之间的数据复制和备份。

### Thanos 的使用场景 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#thanos-%e7%9a%84%e4%bd%bf%e7%94%a8%e5%9c%ba%e6%99%af)

1. **跨地域和跨集群监控**：
   - Thanos 非常适合在多集群、多区域部署 Prometheus 实例的环境中使用，它能够聚合来自不同集群的数据，进行跨区域查询。
2. **长期存储解决方案**：
   - Thanos 可以将 Prometheus 的数据持久化到对象存储（如 AWS S3、Google Cloud Storage 等），实现高效且成本低的长期存储解决方案。
3. **高可用和高容错**：
   - 通过在多个节点或集群中运行 Prometheus 和 Thanos，实现高可用性和容错能力，避免单点故障。
4. **跨多个 Prometheus 实例的全局查询**：
   - 对于需要跨多个 Prometheus 实例进行全局查询的场景，Thanos 提供了无缝的查询能力。

### 总结 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%80%bb%e7%bb%93-3)

Thanos 通过与 Prometheus 紧密集成，提供了对大规模、跨区域、多集群部署环境的支持。它解决了 Prometheus 在高可用、长期存储、分布式查询等方面的瓶颈，是一个强大的扩展解决方案，特别适用于需要全球查询和大规模监控环境的使用场景。

如果你有任何关于 Thanos 的具体问题，或者需要进一步了解配置和实践，随时问我！

## thanos与VictoriaMetrics对比 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#thanos%e4%b8%8evictoriametrics%e5%af%b9%e6%af%94)

Thanos 和 VictoriaMetrics 都是 Prometheus 生态系统中的监控存储解决方案，但它们的设计目标、架构和适用场景有所不同。以下是两者的详细对比：

* * *

## **1\. 概述** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-%e6%a6%82%e8%bf%b0)

| 特性 | **Thanos** | **VictoriaMetrics** |
| --- | --- | --- |
| **主要用途** | 分布式长时间存储、全局查询、数据去重 | 高性能、低资源消耗的时间序列数据库 |
| **与 Prometheus 关系** | Prometheus 的扩展组件 | 兼容 Prometheus 的独立 TSDB |
| **架构** | 组件化（多个独立组件） | 单体应用（轻量级） |
| **存储方式** | 对象存储（S3、GCS、OSS等） | 本地磁盘/对象存储（可选） |

* * *

## **2\. 架构对比** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-%e6%9e%b6%e6%9e%84%e5%af%b9%e6%af%94)

### **Thanos** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#thanos)

- 作为 Prometheus 的扩展组件，增加高可用性、长时间存储和全局查询功能。
- 主要组件：
  - **Thanos Sidecar**：附加到 Prometheus，提供长期存储和查询能力。
  - **Thanos Store**：从对象存储（S3/GCS等）读取历史数据。
  - **Thanos Query**：支持跨多个 Prometheus 实例的全局查询。
  - **Thanos Compact**：合并和去重数据，优化存储空间。
  - **Thanos Ruler**：类似 Prometheus Rules，可在全局范围内执行告警规则。

### **VictoriaMetrics** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#victoriametrics)

- 作为一个独立的 TSDB，支持高效存储和查询。
- 主要组件：
  - **VictoriaMetrics（单机版）**：高效的单实例存储，可替代 Prometheus TSDB。
  - **VictoriaMetrics Cluster**：支持大规模集群部署，提供可扩展性。
  - **vmalert**：替代 Prometheus Alertmanager，实现告警规则执行。
  - **vmagent**：收集和转发监控数据，相当于 Prometheus Remote Write。

* * *

## **3\. 关键特性对比** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-%e5%85%b3%e9%94%ae%e7%89%b9%e6%80%a7%e5%af%b9%e6%af%94)

| 特性 | **Thanos** | **VictoriaMetrics** |
| --- | --- | --- |
| **高可用性** | 依赖多个 Prometheus 实例及对象存储 | 原生支持 HA，不需要多个 Prometheus |
| **数据去重** | 依赖 `--query.replica-label` 进行去重 | 内置高效去重机制 |
| **存储方式** | 对象存储（S3/GCS/OSS等） | 本地存储 \+ 可选远程存储 |
| **查询性能** | 依赖对象存储，查询速度受限于存储系统 | 读取本地磁盘，查询速度更快 |
| **集群部署** | 需要多个组件协同工作 | 单机高效，支持集群模式 |
| **资源占用** | 需要运行多个组件，资源开销较大 | 低资源占用，高压缩比 |
| **写入方式** | 通过 Prometheus Remote Write 间接写入 | 直接支持 Remote Write |
| **数据压缩** | 依赖 Prometheus 的存储格式 | 原生高压缩率 |
| **运维复杂度** | 组件较多，配置复杂 | 部署简单，单进程运行 |

* * *

## **4\. 适用场景** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-%e9%80%82%e7%94%a8%e5%9c%ba%e6%99%af)

| 需求 | **推荐使用** |
| --- | --- |
| **已有 Prometheus 部署，希望扩展长期存储和查询能力** | ✅ Thanos |
| **希望一个高效、独立的 TSDB 取代 Prometheus 存储** | ✅ VictoriaMetrics |
| **需要对象存储（S3/OSS）存储历史数据** | ✅ Thanos |
| **希望低资源占用、高性能查询** | ✅ VictoriaMetrics |
| **多 Prometheus 实例统一查询** | ✅ Thanos |
| **单实例高吞吐、高压缩比的存储** | ✅ VictoriaMetrics |

* * *

## **5\. 总结** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#5-%e6%80%bb%e7%bb%93)

- **Thanos 适合扩展 Prometheus**，增加高可用性、长时间存储和全局查询能力，适用于 **多 Prometheus 实例的企业级监控**。
- **VictoriaMetrics 适合作为独立的 TSDB**，提供高性能、低资源占用的时间序列存储，适用于 **大规模数据存储和高效查询**。

**如果你的 Prometheus 已经部署并希望扩展存储能力，Thanos 是更好的选择。如果你希望一个更高效、更易管理的时序数据库，VictoriaMetrics 更合适。**

## thanos sidecar和receive区别☆ [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#thanos-sidecar%e5%92%8creceive%e5%8c%ba%e5%88%ab)

### **Thanos Sidecar vs Thanos Receive 对比** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#thanos-sidecar-vs-thanos-receive-%e5%af%b9%e6%af%94)

Thanos Sidecar 和 Thanos Receive 都是 Thanos 生态中的组件，但它们的用途和工作方式不同，主要区别在于 **数据写入方式** 和 **存储目标**。下面是它们的详细对比：

* * *

## **1\. 主要用途** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-%e4%b8%bb%e8%a6%81%e7%94%a8%e9%80%94)

| 组件 | **Thanos Sidecar** | **Thanos Receive** |
| --- | --- | --- |
| **作用** | 连接 Prometheus，提供对象存储上传和查询能力 | 直接接收 Prometheus Remote Write 数据，替代 Prometheus 存储 |
| **使用场景** | **已有 Prometheus**，想要扩展长期存储 | **无本地 Prometheus**，需要集中接收和存储数据 |
| **存储方式** | 依赖本地 Prometheus 存储，定期上传到对象存储 | 直接存储 TSDB 数据，并支持水平扩展 |
| **查询方式** | 通过 `thanos query` 读取对象存储的数据 | 通过 `thanos query` 直接查询 Receive 存储的数据 |

* * *

## **2\. 架构对比** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-%e6%9e%b6%e6%9e%84%e5%af%b9%e6%af%94-1)

### **Thanos Sidecar** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#thanos-sidecar)

- 依附于 Prometheus

，作为一个附加组件运行，负责：

  - 提供 **Prometheus 运行时数据** 的查询 API（类似 Prometheus Query API）。
  - 定期将 **Prometheus 的历史数据上传到对象存储**（如 S3、GCS）。
  - 使 Thanos Query 可以同时查询多个 Prometheus 实例的数据。
- 但 **不会直接接收 Remote Write** 数据，仍然依赖 Prometheus 进行采集和存储。


### **Thanos Receive** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#thanos-receive)

- 独立运行

，用于

直接接收 Prometheus Remote Write 数据

，主要功能：

  - 作为 **Prometheus 的替代存储**，不需要本地 Prometheus 。
  - 适用于 **多个 Prometheus 实例集中存储数据**，提升可扩展性。
  - **多副本模式**，适用于 **HA 部署**，通过 `--receive.replication-factor` 控制副本数。
  - 可以直接被 `thanos query` 组件查询，无需额外 Sidecar。

* * *

## **3\. 详细功能对比** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-%e8%af%a6%e7%bb%86%e5%8a%9f%e8%83%bd%e5%af%b9%e6%af%94)

| **特性** | **Thanos Sidecar** | **Thanos Receive** |
| --- | --- | --- |
| **是否依赖 Prometheus** | ✅ 是，依附 Prometheus | ❌ 否，独立运行 |
| **是否支持 Prometheus Remote Write** | ❌ 不支持 | ✅ 直接支持 |
| **数据存储方式** | 本地 Prometheus -> 对象存储（S3/GCS/OSS） | 直接存储 TSDB 数据（支持对象存储） |
| **查询方式** | 通过 Thanos Query 访问 Sidecar 提供的 Prometheus API | 通过 Thanos Query 直接查询 |
| **高可用性** | ❌ 需要多个 Prometheus 实例才能高可用 | ✅ 多实例+数据复制，天然支持高可用 |
| **适用场景** | 需要长期存储， **已有 Prometheus** | 需要集中存储， **多个 Prometheus 统一 Remote Write** |
| **写入模式** | Prometheus 本地存储数据，Sidecar 仅提供查询和存储上传 | 直接作为 Remote Write 目标，存储 Prometheus 采集数据 |
| **适用场景** | 适用于 Prometheus **本地存储扩展** | 适用于 **多集群 Remote Write** 存储方案 |

* * *

## **4\. 适用场景** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-%e9%80%82%e7%94%a8%e5%9c%ba%e6%99%af-1)

| **需求** | **推荐使用** |
| --- | --- |
| **已有 Prometheus，想扩展存储能力（对象存储）** | ✅ **Thanos Sidecar** |
| **想用 Remote Write 代替 Prometheus 本地存储** | ✅ **Thanos Receive** |
| **查询多个 Prometheus 实例的数据** | ✅ **Thanos Sidecar**（搭配 Thanos Query） |
| **跨多个 Prometheus 统一存储数据，简化架构** | ✅ **Thanos Receive** |
| **希望 Prometheus 只负责采集，不存储数据** | ✅ **Thanos Receive** |

* * *

## **5\. 选型建议** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#5-%e9%80%89%e5%9e%8b%e5%bb%ba%e8%ae%ae)

- **如果你已经在用 Prometheus，并且只想增加长期存储（S3/GCS/OSS），那么 Thanos Sidecar 是更好的选择。**
- **如果你的 Prometheus 采集节点非常多，希望用 Remote Write 集中存储数据，或者希望 Prometheus 只负责数据采集，而存储和查询交给一个独立系统，那么 Thanos Receive 更适合。**

**最终选型取决于你的架构需求，如果是传统 Prometheus 方案扩展，Sidecar 够用；如果是大规模监控存储，Receive 更合适！** 🚀

## thanos rule组件和prometheus区别 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#thanos-rule%e7%bb%84%e4%bb%b6%e5%92%8cprometheus%e5%8c%ba%e5%88%ab)

### **Thanos Rule vs Prometheus：区别与对比** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#thanos-rule-vs-prometheus%e5%8c%ba%e5%88%ab%e4%b8%8e%e5%af%b9%e6%af%94)

Thanos Rule 组件和 Prometheus 在规则评估（Rule Evaluation）方面有相似的功能，但它们在 **存储方式**、 **适用场景** 和 **查询方式** 上存在较大差异。

* * *

## **1\. 主要用途** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-%e4%b8%bb%e8%a6%81%e7%94%a8%e9%80%94-1)

| 组件 | **Thanos Rule** | **Prometheus** |
| --- | --- | --- |
| **作用** | 远程评估规则，生成新的时间序列数据 | 本地评估规则，生成新的时间序列数据 |
| **存储方式** | **Remote Write 或对象存储**（TSDB） | 本地存储 TSDB |
| **查询方式** | 通过 Thanos Query 访问存储的规则结果 | 通过 PromQL 查询本地数据 |
| **高可用性** | ✅ 通过多个实例 \+ HA 部署 | ❌ 需要手动管理多个 Prometheus 实例 |
| **适用场景** | **分布式监控，多集群规则评估** | **单个 Prometheus 规则评估** |

* * *

## **2\. 组件介绍** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-%e7%bb%84%e4%bb%b6%e4%bb%8b%e7%bb%8d)

### **📌 Prometheus** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-prometheus)

- **主要职责**：
  - 采集时序数据，并存储在本地 **TSDB**。
  - **Rule 规则评估**，将 **PromQL 计算后的结果存储在本地 TSDB**。
  - **告警（Alerting）评估**，并通过 Alertmanager 发送告警。
- **缺点**：
  - **本地存储限制**，历史数据无法长期存储（除非使用远程存储）。
  - **规则评估仅限本地**，无法跨多个 Prometheus 实例。

* * *

### **📌 Thanos Rule** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-thanos-rule)

- 作用

：

  - 类似于 Prometheus Rules，但它是 **独立的远程规则评估组件**。
  - 可以 **从多个 Prometheus / Thanos Store 读取数据** 进行规则评估。
  - 评估结果可以 **写入对象存储** 或 **Prometheus Remote Write**。
- 关键特点

：

  - **规则评估脱离 Prometheus**，可以 **跨多个 Prometheus 运行**。
  - **支持 HA 部署**，多个 Thanos Rule 实例可以同时运行，不会重复写入数据。
  - **避免 Prometheus 单点问题**，即使 Prometheus 实例宕机，Thanos Rule 仍能继续工作。

* * *

## **3\. 详细功能对比** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-%e8%af%a6%e7%bb%86%e5%8a%9f%e8%83%bd%e5%af%b9%e6%af%94-1)

| **特性** | **Thanos Rule** | **Prometheus** |
| --- | --- | --- |
| **是否采集数据** | ❌ 不采集数据 | ✅ 采集数据 |
| **是否存储数据** | ✅ 通过 Remote Write 或对象存储 | ✅ 本地 TSDB |
| **规则评估方式** | 读取多个数据源，集中计算 | 仅能评估本地数据 |
| **跨集群支持** | ✅ 可以评估多个 Prometheus 数据 | ❌ 仅限单个 Prometheus |
| **HA（高可用性）** | ✅ 多个实例可用，不影响评估 | ❌ 单点故障，需手动管理多个 Prometheus |
| **告警规则（Alerts）** | ✅ 通过 Alertmanager 发送 | ✅ 通过 Alertmanager 发送 |

* * *

## **4\. 适用场景** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-%e9%80%82%e7%94%a8%e5%9c%ba%e6%99%af-2)

| **需求** | **推荐使用** |
| --- | --- |
| **只管理一个 Prometheus 实例，直接用规则计算** | ✅ **Prometheus** |
| **希望跨多个 Prometheus 进行规则评估** | ✅ **Thanos Rule** |
| **需要高可用的规则评估（HA 方案）** | ✅ **Thanos Rule** |
| **希望把计算结果写入远程存储** | ✅ **Thanos Rule** |
| **Prometheus 需要扩展长期存储** | ✅ **Thanos Rule**（搭配 Thanos Store） |

* * *

## **5\. 总结** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#5-%e6%80%bb%e7%bb%93-1)

- **Prometheus** 适用于 **单个实例** 进行规则评估和告警，不适合分布式部署。
- **Thanos Rule** 适用于 **多个 Prometheus 统一评估规则**，解决 Prometheus **规则计算的 HA 问题**，并可将结果存储到远程存储或对象存储。

👉 **如果是单个 Prometheus，直接用 Prometheus Rules 即可。** 👉 **如果是多个 Prometheus，并且希望跨集群评估规则，建议使用 Thanos Rule！** 🚀

## Prometheus告警从触发到收到通知延迟在哪 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#prometheus%e5%91%8a%e8%ad%a6%e4%bb%8e%e8%a7%a6%e5%8f%91%e5%88%b0%e6%94%b6%e5%88%b0%e9%80%9a%e7%9f%a5%e5%bb%b6%e8%bf%9f%e5%9c%a8%e5%93%aa)

Prometheus 告警（Alerting）从触发到收到通知的整个流程可能会出现 **延迟**，主要涉及多个环节，包括 **规则评估**、 **数据存储**、 **告警发送** 等。以下是各个环节的可能延迟点和优化方案。

* * *

## **📌 Prometheus 告警流程** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-prometheus-%e5%91%8a%e8%ad%a6%e6%b5%81%e7%a8%8b)

1. 数据采集（Scrape）
   - Prometheus 定期从 Exporter 或应用程序端点拉取监控数据。
2. 规则评估（Rule Evaluation）
   - Prometheus 根据 `alerting rules` 评估数据，决定是否触发告警。
3. 告警触发（Alert Firing）
   - 满足告警条件的规则被标记为 `firing`（触发）。
4. Alertmanager 处理
   - Prometheus 将告警推送给 Alertmanager，Alertmanager 进行分组、抑制、路由等处理。
5. 通知发送
   - Alertmanager 通过 Webhook、邮件、Slack、钉钉等渠道发送告警。

* * *

## **📌 可能的延迟点** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e5%8f%af%e8%83%bd%e7%9a%84%e5%bb%b6%e8%bf%9f%e7%82%b9)

| **环节** | **可能的延迟来源** | **优化方案** |
| --- | --- | --- |
| **数据采集（Scrape）** | \- Prometheus 抓取间隔 (`scrape_interval`) 过长 \- 目标 Exporter 响应慢或丢失数据 | \- 调整 `scrape_interval`，确保足够频繁抓取数据 \- 确保 Exporter 端点稳定 |
| **规则评估（Rule Evaluation）** | \- `evaluation_interval` 过长 \- PromQL 查询过于复杂，计算时间长 | \- 调整 `evaluation_interval`，建议设为 15s~30s - 优化 PromQL，减少不必要的计算 |
| **告警触发（Alert Firing）** | \- `for` 选项（持续时间）导致延迟 \- Prometheus TSDB 存储查询效率低 | \- 确保 `for` 时间合适，不要过长 \- 增强存储性能（如 Thanos / VictoriaMetrics） |
| **Prometheus -> Alertmanager** | \- Prometheus 向 Alertmanager 发送告警批量处理有延迟 | \- 确保 Prometheus 能够快速推送告警（查看 `alert_relabel_configs`） |
| **Alertmanager 处理** | \- 告警分组 (`group_wait`) 时间过长 \- 告警抑制 (`inhibit_rules`) 影响 \- `group_interval` 影响后续通知 | \- 调整 `group_wait` (建议 10s) - 避免 `group_interval` 过长 |
| **通知发送（Email/钉钉/Slack 等）** | \- API 调用慢（如 Webhook 超时） - 第三方服务（如邮件、钉钉等）处理慢 | \- 优化通知渠道（如使用更快的 Webhook 服务器） - 确保 Alertmanager 配置的通知方式高效 |

* * *

## **📌 如何优化 Prometheus 告警延迟？** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e5%a6%82%e4%bd%95%e4%bc%98%e5%8c%96-prometheus-%e5%91%8a%e8%ad%a6%e5%bb%b6%e8%bf%9f)

### **1\. 调整 `scrape_interval`** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-%e8%b0%83%e6%95%b4)

- 默认情况下，Prometheus 可能每 **60s** 抓取一次数据（`scrape_interval=60s`）。

- 这意味着告警可能最多延迟 **1 分钟**。

- 优化方案：



```yaml
scrape_configs:
  - job_name: "node_exporter"
    scrape_interval: 15s  # 抓取间隔缩短，提高告警实时性
```

### **2\. 调整 `evaluation_interval`** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-%e8%b0%83%e6%95%b4)

- Prometheus 默认每 **60s** 评估一次告警规则（`evaluation_interval=60s`）。

- 这样可能导致数据刷新慢，增加延迟。

- 优化方案：



  ```yaml
  evaluation_interval: 15s  # 规则评估时间缩短
  ```


### **3\. 调整 `for` 参数** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-%e8%b0%83%e6%95%b4)

- ```
  for
  ```


  参数用于避免短暂波动导致的误报，例如：



  ```yaml
- alert: InstanceDown
  expr: up == 0
  for: 2m  # 只有连续 2 分钟宕机才会触发告警
```

- 如果 `for: 5m`，意味着即使 Prometheus 立即检测到问题，也要 **等 5 分钟** 才会上报告警。

- **优化方案**：如果业务不敏感，可以适当减少 `for` 时间，比如 `for: 30s`。


### **4\. 调整 Alertmanager 配置** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-%e8%b0%83%e6%95%b4-alertmanager-%e9%85%8d%e7%bd%ae)

- Alertmanager 默认有 `group_wait` 和 `group_interval` 机制，可能导致告警延迟。

- 优化方案：



```yaml
route:
    group_wait: 10s  # 组内首个告警延迟 10s 发送
    group_interval: 30s  # 组内后续告警每 30s 发送一次
    repeat_interval: 3h  # 3 小时后重新发送相同的告警
```


### **5\. 优化通知方式** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#5-%e4%bc%98%e5%8c%96%e9%80%9a%e7%9f%a5%e6%96%b9%e5%bc%8f)

- **Webhook 方式** 比 **邮件** 和 **钉钉机器人** 更快。
- 优化方案：
  - 尽量使用 **直接的 Webhook**，避免过多代理。
  - 避免 API 服务器超时，增加 Alertmanager 发送失败的重试机制。

* * *

## **📌 结论** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e7%bb%93%e8%ae%ba)

Prometheus 告警从触发到收到通知的延迟，可能发生在多个环节：

1. **数据采集间隔（scrape\_interval）**
2. **规则评估频率（evaluation\_interval）**
3. **告警触发延迟（for 参数）**
4. **Alertmanager 分组/抑制机制（group\_wait、group\_interval）**
5. **通知方式的 API 延迟**

### 🚀 **优化建议** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e4%bc%98%e5%8c%96%e5%bb%ba%e8%ae%ae)

✅ `scrape_interval: 15s`（提高数据采集频率）
✅ `evaluation_interval: 15s`（更快评估告警规则）
✅ `for: 30s`（减少告警延迟）
✅ `group_wait: 10s` \+ `group_interval: 30s`（优化 Alertmanager）
✅ **使用 Webhook 代替钉钉/邮件，提高告警推送速度**

* * *

### **🔥 推荐优化后的示例配置** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e6%8e%a8%e8%8d%90%e4%bc%98%e5%8c%96%e5%90%8e%e7%9a%84%e7%a4%ba%e4%be%8b%e9%85%8d%e7%bd%ae)

#### **Prometheus 配置** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#prometheus-%e9%85%8d%e7%bd%ae)

```yaml
global:
  scrape_interval: 15s  # 默认采集间隔
  evaluation_interval: 15s  # 规则评估间隔

rule_files:
  - "alert.rules.yml"
```

#### **Alertmanager 配置** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#alertmanager-%e9%85%8d%e7%bd%ae)

```yaml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 30s
  repeat_interval: 1h
  receiver: 'webhook'
receivers:
  - name: 'webhook'
    webhook_configs:
      - url: "http://webhook-server/alert"
```

这样可以 **最大程度减少 Prometheus 告警延迟** 🚀🚀🚀

## 告警抑制怎么做☆ [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e5%91%8a%e8%ad%a6%e6%8a%91%e5%88%b6%e6%80%8e%e4%b9%88%e5%81%9a)

在 **Prometheus Alertmanager** 中，告警抑制（Inhibition）用于 **阻止某些告警在特定情况下发送通知**。通常用于减少重复告警，或者在 **更严重的告警发生时，屏蔽不重要的告警**。

* * *

## **📌 告警抑制的工作原理** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e5%91%8a%e8%ad%a6%e6%8a%91%e5%88%b6%e7%9a%84%e5%b7%a5%e4%bd%9c%e5%8e%9f%e7%90%86)

- **抑制规则（Inhibit Rules）** 允许你定义 **“A 告警存在时，B 告警就不会发送通知”**。
- 例如：
  - **如果某个集群已 `Down`，就不要发送 `InstanceDown` 告警**。
  - **如果主数据库已崩溃，就不再发送单个 API 失败的告警**。
- **Alertmanager 只会对已接收到的告警进行抑制**，不会影响 Prometheus 的告警触发。

* * *

## **📌 告警抑制规则配置** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e5%91%8a%e8%ad%a6%e6%8a%91%e5%88%b6%e8%a7%84%e5%88%99%e9%85%8d%e7%bd%ae)

### **示例 1：屏蔽 `InstanceDown` 告警，若 `ClusterDown` 告警已触发** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e7%a4%ba%e4%be%8b-1%e5%b1%8f%e8%94%bd)

```yaml
inhibit_rules:
  - source_match:
      alertname: "ClusterDown"  # 当 "ClusterDown" 告警触发时
    target_match:
      alertname: "InstanceDown"  # 该告警会被抑制
    equal: ["cluster"]  # 必须匹配相同的 cluster 标签
```

📌 **解释**：

- 如果 **`ClusterDown`** 发生（整个集群宕机），则不会发送 **`InstanceDown`**（单个实例宕机）的告警，因为它们可能是相同的问题导致的。

* * *

### **示例 2：某个服务 `APIHighLatency` 时，屏蔽 `APIErrorRate`** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e7%a4%ba%e4%be%8b-2%e6%9f%90%e4%b8%aa%e6%9c%8d%e5%8a%a1)

```yaml
inhibit_rules:
  - source_match:
      alertname: "APIHighLatency"  # API 响应慢
    target_match:
      alertname: "APIErrorRate"  # API 错误率升高
    equal: ["service"]  # 仅在相同服务名的情况下生效
```

📌 **解释**：

- **API 响应变慢（`APIHighLatency`）可能会导致 API 失败（`APIErrorRate`）**，所以可以在延迟过高时屏蔽错误率告警。

* * *

## **📌 配置 `Alertmanager.yml` 规则** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e9%85%8d%e7%bd%ae)

### **完整示例** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e5%ae%8c%e6%95%b4%e7%a4%ba%e4%be%8b)

```yaml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 30s
  repeat_interval: 1h
  receiver: 'default'

receivers:
  - name: 'default'
    webhook_configs:
      - url: "http://webhook-server/alert"

inhibit_rules:
  - source_match:
      alertname: "ClusterDown"
    target_match:
      alertname: "InstanceDown"
    equal: ["cluster"]

  - source_match:
      alertname: "APIHighLatency"
    target_match:
      alertname: "APIErrorRate"
    equal: ["service"]
```

* * *

## **📌 如何测试告警抑制** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e5%a6%82%e4%bd%95%e6%b5%8b%e8%af%95%e5%91%8a%e8%ad%a6%e6%8a%91%e5%88%b6)

1. 触发 `ClusterDown`

：



```yaml
ALERT ClusterDown
     IF count(up == 0) > 5
     FOR 1m
```

2. 触发 `InstanceDown`

：



```yaml
ALERT InstanceDown
     IF up == 0
     FOR 1m
```

3. 查看 Alertmanager 界面

   - **若 `ClusterDown` 触发，`InstanceDown` 将被抑制（不会发通知）**。
   - 在 Alertmanager 的 **`/alerts` 页面**，你可以看到 `InstanceDown` 状态变为 `Suppressed`（已抑制）。

* * *

## **📌 总结** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e6%80%bb%e7%bb%93)

✅ **告警抑制的作用**：

- 减少 **重复告警**，避免告警风暴。
- 让更严重的告警优先级更高，防止被无关告警干扰。

✅ **关键配置**

- `source_match`: 触发抑制的告警（优先级高）。
- `target_match`: 被抑制的告警（优先级低）。
- `equal`: 需要匹配的标签（如 `cluster`、`service`）。

💡 **最终效果**：
💥 **如果集群崩溃（`ClusterDown`）时，所有实例宕机告警（`InstanceDown`）都会被抑制**，这样不会收到重复的实例告警。 🚀

### **Prometheus 告警抑制（Inhibition）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#prometheus-%e5%91%8a%e8%ad%a6%e6%8a%91%e5%88%b6inhibition)

告警抑制（ **Inhibition**）用于 **屏蔽** 某些告警，防止高优先级的告警触发时，低优先级的告警同时触发，造成信息冗余或混乱。

例如：

- **屏蔽**“服务实例不可用”告警 **（InstanceDown）**，当整个集群不可用时 **（ClusterDown）** 触发。
- **屏蔽**“磁盘使用率过高”告警 **（DiskUsageHigh）**，当磁盘已满时 **（DiskFull）** 触发。

* * *

## **📌 配置告警抑制的步骤** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e9%85%8d%e7%bd%ae%e5%91%8a%e8%ad%a6%e6%8a%91%e5%88%b6%e7%9a%84%e6%ad%a5%e9%aa%a4)

告警抑制需要在 **Alertmanager** 中配置 `inhibit_rules`，其规则如下：

- **source\_match**：指定 **高优先级** 的告警（如果此告警触发，则屏蔽其他告警）。
- **target\_match**：指定 **低优先级** 的告警（当 `source_match` 触发时，该告警会被屏蔽）。
- **equal**：需要匹配的标签，确保同一服务或同一实例的告警才会被抑制。

* * *

## **📌 示例：屏蔽 InstanceDown 告警** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e7%a4%ba%e4%be%8b%e5%b1%8f%e8%94%bd-instancedown-%e5%91%8a%e8%ad%a6)

**场景：**

- `ClusterDown`（整个集群宕机）触发时，不再单独告警 `InstanceDown`（单个实例不可用）。
- **目标**：避免同一事件重复发送多个告警，减少噪音。

```
yaml

复制编辑
inhibit_rules:
  - source_match:
      alertname: "ClusterDown"  # 高优先级告警
    target_match:
      alertname: "InstanceDown"  # 低优先级告警
    equal: ["cluster"]  # 只抑制同一个 cluster 的告警
```

🔹 **效果**：当 `ClusterDown` 触发时，`InstanceDown` 告警不会发送。

* * *

## **📌 示例：屏蔽磁盘使用率高告警** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e7%a4%ba%e4%be%8b%e5%b1%8f%e8%94%bd%e7%a3%81%e7%9b%98%e4%bd%bf%e7%94%a8%e7%8e%87%e9%ab%98%e5%91%8a%e8%ad%a6)

**场景：**

- `DiskFull`（磁盘已满）触发时，不再告警 `DiskUsageHigh`（磁盘使用率高）。
- **目标**：避免相同原因导致的多条告警，减少不必要的告警流量。

```
yaml

复制编辑
inhibit_rules:
  - source_match:
      alertname: "DiskFull"  # 高级别告警
    target_match:
      alertname: "DiskUsageHigh"  # 低级别告警
    equal: ["instance", "device"]  # 只抑制相同 instance 和 device 的告警
```

🔹 **效果**：磁盘已满时，不会再收到磁盘使用率高的告警。

* * *

## **📌 示例：屏蔽 `NodeDown`（节点宕机）告警** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e7%a4%ba%e4%be%8b%e5%b1%8f%e8%94%bd)

**场景：**

- `NetworkDown`（网络不可用）触发时，不再告警 `NodeDown`（节点宕机）。
- **目标**：避免误报，因为网络问题会导致所有节点的 `NodeDown` 告警。

```
yaml

复制编辑
inhibit_rules:
  - source_match:
      alertname: "NetworkDown"
    target_match:
      alertname: "NodeDown"
    equal: ["datacenter"]
```

🔹 **效果**：当 `NetworkDown` 触发时，同一数据中心的 `NodeDown` 告警会被屏蔽。

* * *

## **📌 `inhibit_rules` 配置完整示例** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#heading)

```
yaml

复制编辑
route:
  receiver: "default"

inhibit_rules:
  - source_match:
      alertname: "ClusterDown"
    target_match:
      alertname: "InstanceDown"
    equal: ["cluster"]

  - source_match:
      alertname: "DiskFull"
    target_match:
      alertname: "DiskUsageHigh"
    equal: ["instance", "device"]

  - source_match:
      alertname: "NetworkDown"
    target_match:
      alertname: "NodeDown"
    equal: ["datacenter"]
```

📌 **解释**

1. **`ClusterDown` 触发时，不再触发 `InstanceDown`**。
2. **`DiskFull` 触发时，不再触发 `DiskUsageHigh`**。
3. **`NetworkDown` 触发时，不再触发 `NodeDown`**。

* * *

## **📌 如何测试告警抑制？** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e5%a6%82%e4%bd%95%e6%b5%8b%e8%af%95%e5%91%8a%e8%ad%a6%e6%8a%91%e5%88%b6-1)

### **方式 1：使用 `amtool`** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%96%b9%e5%bc%8f-1%e4%bd%bf%e7%94%a8)

```
sh

复制编辑
amtool silence query  # 查看当前抑制的告警
```

### **方式 2：查看 Alertmanager UI** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%96%b9%e5%bc%8f-2%e6%9f%a5%e7%9c%8b-alertmanager-ui)

1. 访问 Alertmanager Web UI：


```
cpp


复制编辑
http://<alertmanager-ip>:9093
```

2. 进入 **“Silences”（抑制）** 页面，查看哪些告警被屏蔽。


* * *

## **📌 结论** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e7%bb%93%e8%ae%ba-1)

✅ **告警抑制（Inhibition）** 可以防止低优先级的告警重复触发，提高告警的准确性。
✅ **`inhibit_rules`** 通过 **source\_match、target\_match 和 equal** 匹配规则，决定哪些告警需要被屏蔽。
✅ **常见用例**：

- **集群宕机时屏蔽单个实例告警**。
- **磁盘满时屏蔽磁盘使用率高的告警**。
- **网络不可用时屏蔽所有节点宕机的告警**。

这样可以 **减少告警噪音，提高运维效率** 🚀🚀🚀！

## 告警架构高可用怎么做☆ [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e5%91%8a%e8%ad%a6%e6%9e%b6%e6%9e%84%e9%ab%98%e5%8f%af%e7%94%a8%e6%80%8e%e4%b9%88%e5%81%9a)

## **Prometheus 告警系统高可用架构方案** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#prometheus-%e5%91%8a%e8%ad%a6%e7%b3%bb%e7%bb%9f%e9%ab%98%e5%8f%af%e7%94%a8%e6%9e%b6%e6%9e%84%e6%96%b9%e6%a1%88)

Prometheus 的告警系统主要由 **Prometheus + Alertmanager** 组成，要保证其高可用（HA），需要解决以下问题：

1. **Prometheus 高可用**（数据采集、存储的 HA）
2. **Alertmanager 高可用**（告警处理、去重的 HA）
3. **通知渠道高可用**（Webhook、邮件、企业微信等）

* * *

## **📌 1\. Prometheus 高可用** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-1-prometheus-%e9%ab%98%e5%8f%af%e7%94%a8)

Prometheus 负责 **数据采集** 和 **告警规则执行**，要保证它的高可用，主要考虑：

- **主备部署（热备）**
- **水平扩展（联邦集群）**
- **存储层高可用**

### **🔹 方案 1：Prometheus 双实例（主备）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e6%96%b9%e6%a1%88-1prometheus-%e5%8f%8c%e5%ae%9e%e4%be%8b%e4%b8%bb%e5%a4%87)

- **架构：** 部署两台独立的 Prometheus 实例，监控相同的目标。
- **优点：** 简单易实现，适用于小型环境。
- **缺点：** 需要负载均衡或手动切换，存储不共享。

```plaintext
       ┌───────────┐
       │  Exporter │
       └────┬──────┘
            │
  ┌────────▼────────┐   ┌───────────┐
  │ Prometheus (主)  │   │ Alertmanager │
  ├────────┬────────┘   └───────────┘
  │ Prometheus (备)  │
  └────────┴────────┘
```

✅ **如何做？**

1. **配置两个 Prometheus 实例**，让它们都拉取相同的监控数据。
2. **前端（如 Grafana）配置负载均衡**，如果一个 Prometheus 宕机，使用另一个。

* * *

### **🔹 方案 2：Prometheus + Thanos/ VictoriaMetrics** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e6%96%b9%e6%a1%88-2prometheus--thanos-victoriametrics)

- **架构：** Prometheus 采集数据，Thanos/VictoriaMetrics 负责远程存储和 HA 查询。
- **优点：** 高可用、分布式存储、支持历史数据查询。
- **缺点：** 需要额外组件（Thanos/VictoriaMetrics）。

```plaintext
       ┌─────────────┐
       │   Exporter  │
       └─────┬───────┘
             │
 ┌──────────▼──────────┐
 │    Prometheus-1     │
 ├──────────┬──────────┘
 │    Prometheus-2     │
 └──────────┴──────────┘
            │
  ┌────────▼────────┐
  │ Thanos Query    │
  ├─────────────────┘
  │ Thanos Store    │
  └─────────────────┘
```

✅ **如何做？**

1. **部署多个 Prometheus 实例**，采集相同数据。
2. **使用 Thanos Query/VictoriaMetrics**，让查询层能自动聚合多个 Prometheus 数据。

* * *

## **📌 2\. Alertmanager 高可用** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-2-alertmanager-%e9%ab%98%e5%8f%af%e7%94%a8)

Alertmanager 负责 **告警的去重、分组、通知**，如果单点故障，告警就可能丢失。

### **🔹 方案：Alertmanager 集群** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e6%96%b9%e6%a1%88alertmanager-%e9%9b%86%e7%be%a4)

- **架构：** 多个 Alertmanager 进程，通过 `mesh` 互相同步状态。
- **优点：** 互相同步，防止单点故障，告警不会丢失。
- **缺点：** 需要额外负载均衡。

```plaintext
       ┌────────────────┐
       │    Prometheus   │
       ├────────────────┘
       │  发送告警        │
       └────────▲───────┘
                │
 ┌──────────────┴──────────────┐
 │   Alertmanager (实例 1)      │
 │   Alertmanager (实例 2)      │   (集群模式)
 │   Alertmanager (实例 3)      │
 └──────────────┬──────────────┘
                │
        ┌──────▼──────┐
        │  Webhook    │
        │  邮件通知   │
        │  企业微信   │
        └────────────┘
```

✅ **如何做？**

1. **多个 Alertmanager 实例** 通过 `--cluster.peer=<other-instance>` 互相同步。
2. **负载均衡（如 Nginx/LB）** 让 Prometheus 以 HA 方式访问 Alertmanager。
3. **持久化存储** 避免 Alertmanager 重启后丢失状态。

* * *

## **📌 3\. 通知渠道高可用** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-3-%e9%80%9a%e7%9f%a5%e6%b8%a0%e9%81%93%e9%ab%98%e5%8f%af%e7%94%a8)

如果通知方式不可用（例如 Webhook、邮件服务器宕机），可能导致告警丢失。

### **🔹 方案 1：多个通知通道** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e6%96%b9%e6%a1%88-1%e5%a4%9a%e4%b8%aa%e9%80%9a%e7%9f%a5%e9%80%9a%e9%81%93)

- 配置

多个通知通道

，例如：

  - 主要通知： **邮件**
  - 备用通知： **企业微信**
  - 紧急通知： **短信**

```yaml
receivers:
  - name: "email"
    email_configs:
      - to: "admin@example.com"

  - name: "wechat"
    wechat_configs:
      - to: "DevOps_Group"

  - name: "sms"
    webhook_configs:
      - url: "http://sms-provider/api/send"
```

✅ **如何做？**

- **定义多个 `receiver`**，并在 `route` 里配置故障转移逻辑。

* * *

### **🔹 方案 2：Webhook 高可用** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e6%96%b9%e6%a1%88-2webhook-%e9%ab%98%e5%8f%af%e7%94%a8)

- 如果 Webhook 服务器崩溃，告警可能丢失。
- 方案：
  - 使用 **负载均衡（Nginx、HAProxy）** 代理多个 Webhook 服务器。
  - 采用 **Kafka/RabbitMQ** 作为 Webhook 缓冲队列，防止瞬时流量过载。

* * *

## **📌 4\. 全局高可用架构** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-4-%e5%85%a8%e5%b1%80%e9%ab%98%e5%8f%af%e7%94%a8%e6%9e%b6%e6%9e%84)

**最终的 Prometheus 告警高可用架构如下：**

```plaintext
       ┌───────────────────┐
       │    Exporters      │
       └────────▲──────────┘
                │
 ┌──────────────┴──────────────┐
 │  Prometheus (实例 1)         │
 │  Prometheus (实例 2)         │  (高可用)
 └──────────────┬──────────────┘
                │
       ┌───────▼───────┐
       │ Thanos Query  │  (可选)
       └───────┬───────┘
               │
 ┌────────────▼────────────┐
 │  Alertmanager (实例 1)   │
 │  Alertmanager (实例 2)   │  (集群)
 └────────────▲────────────┘
              │
 ┌────────────▼────────────┐
 │  邮件 / 微信 / Webhook  │  (多通道)
 └────────────────────────┘
```

* * *

## **📌 结论** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e7%bb%93%e8%ae%ba-2)

✅ **Prometheus 高可用**

- **方案 1：双实例主备**
- **方案 2：Thanos/VictoriaMetrics 分布式**

✅ **Alertmanager 高可用**

- **Alertmanager 集群**，同步状态
- **负载均衡（Nginx/LB）**

✅ **通知渠道高可用**

- **多通道（邮件、微信、短信）**
- **Webhook 负载均衡 + 消息队列（Kafka）**

🚀 **这样可以确保整个告警系统的高可用，避免单点故障！**

## Pod指标WSS和RSS区别☆ [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#pod%e6%8c%87%e6%a0%87wss%e5%92%8crss%e5%8c%ba%e5%88%ab)

在 Kubernetes 或 Linux 系统中， **RSS（Resident Set Size）** 和 **WSS（Working Set Size）** 是衡量 Pod 或进程内存使用的重要指标，它们有不同的含义和使用场景。

* * *

## **📌 RSS（Resident Set Size）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-rssresident-set-size)

- **RSS 是进程实际占用物理内存的大小**，不包括交换（swap）出去的部分。
- 它包括：
  - **私有匿名页**（heap、stack 等）
  - **共享库的已加载部分**
- 不包括：
  - **未被访问的内存页**
  - **Swap 到磁盘的部分**
  - **文件映射的未加载部分**

✅ **适用场景**

- 观察 Pod 的实际物理内存占用，避免超出节点可用内存。
- 用于 Kubernetes `memory.usage` 监控。

📌 **示例：查看 Pod RSS**

```bash
kubectl exec -it <pod-name> -- cat /proc/1/status | grep VmRSS
```

或者：

```bash
ps -o rss,vsz,pid,cmd | grep <process_name>
```

* * *

## **📌 WSS（Working Set Size）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-wssworking-set-size)

- **WSS 是进程在一段时间内频繁访问的内存大小**。
- 只统计 **活跃** 的内存页，未访问的部分不算 WSS。
- 包括：
  - 进程在短时间窗口内访问的页。
  - 可能会被频繁调入物理内存的部分。

✅ **适用场景**

- 评估进程实际需要的内存，避免过度分配或 OOM（Out of Memory）。
- 用于 Kubernetes `memory.working_set_bytes` 指标。

📌 **示例：使用 `cgroup` 监控 WSS**

```bash
cat /sys/fs/cgroup/memory/kubepods/burstable/pod*/memory.working_set_bytes
```

* * *

## **📌 WSS vs. RSS 总结** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-wss-vs-rss-%e6%80%bb%e7%bb%93)

| 指标 | 含义 | 计算方式 | 适用场景 |
| --- | --- | --- | --- |
| **RSS** | 实际占用的物理内存 | `VmRSS` | 监控内存使用，避免超配 |
| **WSS** | 频繁访问的内存 | `memory.working_set_bytes` | 估算应用实际需要的内存 |

* * *

## **📌 Kubernetes 相关** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-kubernetes-%e7%9b%b8%e5%85%b3)

- `memory.usage_bytes` ≈ RSS
- `memory.working_set_bytes` ≈ WSS
- 监控时，如果 **RSS 远大于 WSS**，说明有 **很多未使用的内存**，可以优化资源分配。

* * *

🚀 **总结**

- **RSS = 进程实际占用的物理内存**
- **WSS = 进程活跃访问的内存**
- **WSS < RSS**，因为 RSS 还包括未使用但驻留在物理内存中的部分。

## 监控四个黄金指标 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e7%9b%91%e6%8e%a7%e5%9b%9b%e4%b8%aa%e9%bb%84%e9%87%91%e6%8c%87%e6%a0%87)

**监控的四个黄金指标（Four Golden Signals）** 是 Google SRE（Site Reliability Engineering）实践中提出的核心监控原则，用于衡量系统的健康状况和性能。这四个指标分别是： **Latency（延迟）、Traffic（流量）、Errors（错误）、Saturation（饱和度）**。

* * *

## **📌 1\. Latency（延迟）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-1-latency%e5%bb%b6%e8%bf%9f)

**定义**：指请求从发出到收到响应的时间，包括：

- **成功请求的延迟（成功响应时间）**
- **失败请求的延迟（错误响应时间）**

**监控方式**：

- P99/P95/P50（百分位）延迟：
  - P99 延迟：最慢的 1% 请求的响应时间，代表最差用户体验
  - P95 延迟：最慢的 5% 请求
  - P50 延迟（中位数）：一半请求的延迟情况
- **区分正常请求和失败请求的延迟**
- **监控 HTTP/gRPC 响应时间**

**Prometheus 监控示例**

```yaml
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
```

* * *

## **📌 2\. Traffic（流量）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-2-traffic%e6%b5%81%e9%87%8f)

**定义**：系统的请求速率或带宽使用情况，反映了负载。

**监控方式**：

- HTTP/gRPC 请求数：



```yaml
rate(http_requests_total[5m])
```

- 吞吐量（TPS/QPS）：

  - TPS（Transaction Per Second）：事务处理速率
  - QPS（Query Per Second）：查询速率
- 带宽监控（流量大小）：



```yaml
rate(node_network_receive_bytes_total[5m])
```


* * *

## **📌 3\. Errors（错误率）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-3-errors%e9%94%99%e8%af%af%e7%8e%87)

**定义**：请求失败率或错误数，衡量系统的健康状况。

**监控方式**：

- HTTP 5xx 错误率



```yaml
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])
```

- **gRPC 错误码分布**

- **应用级错误，如数据库查询失败**

- **Kubernetes 事件、OOM（Out Of Memory）监控**


* * *

## **📌 4\. Saturation（饱和度）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-4-saturation%e9%a5%b1%e5%92%8c%e5%ba%a6)

**定义**：系统资源的利用率，衡量瓶颈是否出现。

**监控方式**：

- CPU 利用率



```yaml
rate(node_cpu_seconds_total{mode="user"}[5m]) / sum(rate(node_cpu_seconds_total[5m]))
```

- 内存使用率



```yaml
node_memory_Active_bytes / node_memory_MemTotal_bytes
```

- 磁盘 I/O



```yaml
rate(node_disk_read_bytes_total[5m])
```

- 网络带宽



```yaml
rate(node_network_transmit_bytes_total[5m])
```


* * *

## **📌 总结** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#-%e6%80%bb%e7%bb%93-1)

| 指标 | 解释 | 监控方式 |
| --- | --- | --- |
| **Latency（延迟）** | 请求处理时间 | HTTP 响应时间、P99 延迟 |
| **Traffic（流量）** | 请求量、带宽 | QPS、TPS、网络流量 |
| **Errors（错误率）** | 失败请求占比 | HTTP 5xx、gRPC 错误率 |
| **Saturation（饱和度）** | 资源利用率 | CPU、内存、磁盘 I/O |

🚀 **最佳实践**

- 结合 Prometheus + Grafana 进行可视化监控
- 设定告警阈值，及时发现异常
- 使用 **百分位（P99/P95）** 衡量用户体验
- 结合 **自动扩缩容（HPA/VPA）** 进行优化

这些指标可以帮助 SRE/DevOps 工程师快速识别系统瓶颈，提升稳定性！💪

## 在大规模环境下，如何优化Prometheus性能 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e5%9c%a8%e5%a4%a7%e8%a7%84%e6%a8%a1%e7%8e%af%e5%a2%83%e4%b8%8b%e5%a6%82%e4%bd%95%e4%bc%98%e5%8c%96prometheus%e6%80%a7%e8%83%bd)

在大规模环境下，Prometheus 的性能优化非常重要，因为随着监控数据量的增加，Prometheus 可能会面临存储和查询延迟等问题。以下是一些常见的优化方法，可以帮助提升 Prometheus 的性能和可扩展性：

### **1\. 分布式架构设计** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-%e5%88%86%e5%b8%83%e5%bc%8f%e6%9e%b6%e6%9e%84%e8%ae%be%e8%ae%a1)

在大规模环境中，单个 Prometheus 实例可能无法承载庞大的监控数据量。因此，采用分布式架构进行水平扩展是非常重要的。

- **使用 Prometheus 高可用性架构**：通过多个 Prometheus 实例来提高容错能力和扩展性。可以通过 **Prometheus Federation（联邦）** 或 **Thanos**、 **Cortex** 等工具来实现数据的聚合和跨集群查询。

- **Federation（联邦）**：Prometheus 的联邦模式允许你在多个 Prometheus 实例间聚合数据。主 Prometheus 实例从子实例中拉取部分数据，提供全局视图。



```yaml
scrape_configs:
  - job_name: 'federation'
    scrape_interval: 1m
    honor_labels: true
    static_configs:
      - targets:
        - 'prometheus-01:9090'
        - 'prometheus-02:9090'
```

- **Thanos 或 Cortex**：它们是 Prometheus 的扩展系统，通过将数据分片和存储在外部对象存储中，提供更好的扩展性和长期存储。


* * *

### **2\. 数据存储优化** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-%e6%95%b0%e6%8d%ae%e5%ad%98%e5%82%a8%e4%bc%98%e5%8c%96)

存储是 Prometheus 性能瓶颈的关键因素之一。通过合理配置存储系统，可以有效提高 Prometheus 性能。

- **调整 Retention 时间**：减少数据保留时间，特别是在不需要存储历史数据时。例如，设置较短的 `--storage.tsdb.retention.time` 来控制保留的数据时间。



  ```yaml
  --storage.tsdb.retention.time=15d
  ```

- **调整 Block 大小**：默认情况下，Prometheus 会将数据分为多个块（blocks），每个块默认大小为 2GB。你可以根据需求调整 `--storage.tsdb.block-duration` 参数，以增加或减少每个块的大小。



  ```yaml
  --storage.tsdb.block-duration=2h
  ```

- **使用 SSD 存储**：将 Prometheus 数据存储放在 SSD 上可以显著提高查询性能，特别是在写入和查询负载较高的情况下。

- **适当调整 WAL（Write-Ahead Log）设置**：通过配置 Prometheus 的写前日志，可以减少磁盘 IO 操作的次数，提高写入性能。


* * *

### **3\. 查询优化** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-%e6%9f%a5%e8%af%a2%e4%bc%98%e5%8c%96)

对于大规模的环境，Prometheus 查询的效率至关重要。通过优化查询，减少高开销查询的次数，可以大大提高性能。

- **避免高时间范围的查询**：查询过长时间范围的数据会增加查询负载，应尽量避免一次性查询过多时间的数据。

- **分片查询（Subqueries）**：通过分割复杂查询为多个较小的查询来避免性能瓶颈，分片查询可以减轻数据库负载。

- **预计算和聚合**：使用 **Recording Rules** 来提前计算和存储某些常见的聚合数据，以减少查询时的计算压力。



  ```yaml
  rule_files:
  - "recording_rules.yml"
```

- **查询缓存**：使用 **Prometheus Query Caching** 来缓存热点查询的结果，避免重复计算。


* * *

### **4\. 调整 Scrape 配置** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-%e8%b0%83%e6%95%b4-scrape-%e9%85%8d%e7%bd%ae)

Prometheus 的拉取频率和数据量也会影响性能。调整采集（scrape）配置，可以有效减少负载。

- **减少 Scrape 频率**：根据需求适当增加 `scrape_interval`，尤其是对于不需要实时更新的指标，可以设置较长的间隔时间。



```yaml
scrape_configs:
  - job_name: 'my_job'
    scrape_interval: 60s  # 默认为 15s
```

- **使用采集过滤**：通过过滤不需要的指标，减少 Prometheus 拉取的数据量。例如，可以通过 `metric_relabel_configs` 来排除不必要的指标。



  ```yaml
  metric_relabel_configs:
  - source_labels: [__name__]
    regex: '.*_unused_metric'
    action: drop
```

- **增加 Target 数量**：在大规模环境下，确保 Prometheus 足够强大以处理多个 target 的数据拉取，可以通过 `scrape_timeout` 和 `scrape_interval` 配置来平衡拉取数据的速率。


* * *

### **5\. 使用 External Storage** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#5-%e4%bd%bf%e7%94%a8-external-storage)

对于长期存储需求较高的环境，可以将 Prometheus 数据存储迁移到外部存储系统，如 **Thanos**、 **Cortex** 等，以减轻 Prometheus 的存储压力。

- **Thanos / Cortex**：将 Prometheus 数据流式存储到外部对象存储（如 S3、GCS）中，并且提供查询和数据持久化功能。通过这种方式，Prometheus 仅需要处理当前的监控数据，减少存储负担。
- **Prometheus + InfluxDB**：将长期存储的指标数据迁移到 InfluxDB 或其他时间序列数据库，以便 Prometheus 专注于短期内的指标查询。

* * *

### **6\. 优化 Alerting 和 Rules** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#6-%e4%bc%98%e5%8c%96-alerting-%e5%92%8c-rules)

Prometheus 中的告警规则和记录规则可能会带来性能压力。通过优化告警规则和减轻告警的计算负担，可以提高性能。

- **优化 Alerting 规则**：避免过于复杂的告警条件，减少不必要的计算。例如，避免每个时刻都触发的告警，可以设定更高的阈值。

- **减少告警频率**：使用 `for` 参数，避免因临时波动触发过多告警。



```yaml
- alert: HighErrorRate
expr: rate(http_requests_total{status="500"}[5m]) > 0.05
for: 10m  # 在 10 分钟内持续触发告警才有效
```

* * *

### **7\. 高可用和负载均衡** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#7-%e9%ab%98%e5%8f%af%e7%94%a8%e5%92%8c%e8%b4%9f%e8%bd%bd%e5%9d%87%e8%a1%a1)

在大规模部署中，确保 Prometheus 的高可用性非常重要。可以采用以下方案：

- **使用多个 Prometheus 实例**：分布式部署多个 Prometheus 实例，提高数据的采集能力。
- **负载均衡**：通过负载均衡器来分担 Prometheus 采集目标（targets）的负载。

* * *

### **总结：** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%80%bb%e7%bb%93-4)

优化 Prometheus 性能时，要从存储、查询、采集配置等多个方面入手。针对大规模环境，采用分布式架构（如 Federation、Thanos 或 Cortex）、优化存储（SSD、Retain 设置）、合理配置查询和采集策略、使用外部存储等都可以有效提升性能。同时，监控系统的高可用性也是确保稳定运行的关键。

## 如何实现告警的自动化响应☆ [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e5%a6%82%e4%bd%95%e5%ae%9e%e7%8e%b0%e5%91%8a%e8%ad%a6%e7%9a%84%e8%87%aa%e5%8a%a8%e5%8c%96%e5%93%8d%e5%ba%94)

实现告警的自动化响应是 DevOps 和 SRE（Site Reliability Engineering）实践中的一个重要组成部分，能够在问题发生时快速采取行动，减少手动干预和响应时间。以下是实现告警自动化响应的一些方法和最佳实践：

### **1\. 使用 Prometheus + Alertmanager 配置告警自动化响应** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-%e4%bd%bf%e7%94%a8-prometheus--alertmanager-%e9%85%8d%e7%bd%ae%e5%91%8a%e8%ad%a6%e8%87%aa%e5%8a%a8%e5%8c%96%e5%93%8d%e5%ba%94)

Prometheus 与 Alertmanager 配合使用是最常见的告警系统。在告警触发后，Alertmanager 可以根据配置将告警发送到指定的接收器（如电子邮件、Slack、Webhook 等）。可以在告警通知中集成自动化响应流程。

#### **步骤：** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%ad%a5%e9%aa%a4)

1. **设置 Prometheus 告警规则**：首先，设置告警规则，用于检测异常情况。比如，当 CPU 使用率超过某个阈值时触发告警。



   ```yaml
   groups:
- name: example
  rules:
  - alert: HighCpuUsage
    expr: avg(rate(cpu_usage_seconds_total[1m])) by (instance) > 0.9
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "CPU usage is high on {{ $labels.instance }}"
```

2. **配置 Alertmanager**：配置 Alertmanager 来处理告警并将其发送到不同的通知渠道。告警可以触发 Slack、PagerDuty 或 Webhook 等自动化响应机制。



```yaml
route:
     group_by: ['alertname']
     receiver: 'slack'

receivers:
- name: 'slack'
slack_configs:
    - api_url: 'https://slack.com/api/alerts'
      channel: '#alerts'
```

3. **Webhook 集成**：为了实现告警的自动化响应，可以通过配置 Alertmanager 的 Webhook 接收器来触发自定义的自动化响应脚本或外部工具。



   ```yaml
   receivers:
- name: 'webhook-receiver'
  webhook_configs:
  - url: 'http://your-service.example.com/alert'
```

4. **自动化响应**：在告警触发时，Webhook 会通知指定的 URL。你可以编写一个 Web 服务来接收这些 Webhook 通知，并根据告警信息自动执行响应操作，例如：

   - 自动重启故障的 Pod 或服务
   - 调整负载均衡配置
   - 执行一组修复脚本

### **2\. 集成自动化工具（如 Ansible、Terraform）进行响应** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-%e9%9b%86%e6%88%90%e8%87%aa%e5%8a%a8%e5%8c%96%e5%b7%a5%e5%85%b7%e5%a6%82-ansibleterraform%e8%bf%9b%e8%a1%8c%e5%93%8d%e5%ba%94)

在一些情况下，可能需要执行更复杂的操作，如扩展基础设施或执行修复操作。可以使用工具如 **Ansible** 或 **Terraform** 来实现自动化响应。

#### **步骤：** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%ad%a5%e9%aa%a4-1)

1. **配置告警触发 Webhook**：将 Alertmanager 配置为触发 Webhook。

2. **编写自动化响应脚本**：创建一个接收告警 Webhook 的 HTTP 服务，解析告警信息，并根据不同的告警类型执行相应的自动化操作。例如，使用 Ansible 运行修复脚本或扩容命令。

**示例：**



```python
import json
import subprocess
from flask import Flask, request

app = Flask(__name__)

@app.route('/alert', methods=['POST'])
def alert():
       alert_data = json.loads(request.data)
       # 根据告警类型执行不同的操作
       if alert_data['alertname'] == 'HighCpuUsage':
           subprocess.call(["ansible-playbook", "fix-cpu-issue.yml"])
       elif alert_data['alertname'] == 'HighMemoryUsage':
           subprocess.call(["ansible-playbook", "fix-memory-issue.yml"])
       return 'OK', 200

if __name__ == '__main__':
       app.run(debug=True, port=5000)
```

3. **通过 Ansible 执行操作**：在告警触发时，自动运行 Ansible Playbook 来修复问题。例如，重新启动服务或扩展容器副本：



```yaml
   ---
- name: Restart High CPU Service
hosts: localhost
tasks:
    - name: Restart Pod
      kubernetes.core.k8s:
        state: restarted
        name: my-service
        namespace: default
        kubeconfig: /path/to/kubeconfig
```

4. **自动化扩容**：在告警触发时，自动扩容应用服务，以应对更高的负载。可以使用 Terraform 来扩展基础设施资源，如 EC2 实例或 Kubernetes 节点。



   ```hcl
   resource "aws_instance" "web" {
     ami = "ami-0c55b159cbfafe1f0"
     instance_type = "t2.micro"
   }
   ```


### **3\. 集成 ChatOps 进行自动化响应** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-%e9%9b%86%e6%88%90-chatops-%e8%bf%9b%e8%a1%8c%e8%87%aa%e5%8a%a8%e5%8c%96%e5%93%8d%e5%ba%94)

ChatOps 通过将操作自动化与聊天工具（如 Slack、Microsoft Teams）集成，使得告警响应更加迅速且可追踪。

#### **步骤：** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%ad%a5%e9%aa%a4-2)

1. **集成 Slack 和 Prometheus Alertmanager**：通过 Alertmanager 配置 Slack 作为告警接收器。

2. **编写 ChatOps 命令**：使用 Slack 的机器人（例如 **Hubot** 或 **Lita**）来监听告警，并根据预设命令自动执行响应操作。用户可以直接在聊天中输入命令，触发自动化脚本执行修复操作。

   **示例命令**：

   - `/restart pod my-app`：重启出现问题的 Pod
   - `/scale up my-app`：扩展应用副本数
3. **执行自动化操作**：通过与 ChatOps 机器人集成的自动化脚本执行基础设施操作，例如自动扩展资源、重启故障节点等。


### **4\. 使用 Kubernetes Operator 进行自动化修复** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-%e4%bd%bf%e7%94%a8-kubernetes-operator-%e8%bf%9b%e8%a1%8c%e8%87%aa%e5%8a%a8%e5%8c%96%e4%bf%ae%e5%a4%8d)

Kubernetes Operator 是一种管理 Kubernetes 资源的模式，可以自动响应应用程序的故障。

#### **步骤：** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%ad%a5%e9%aa%a4-3)

1. **创建自定义 Operator**：开发一个自定义的 Operator，用于监控 Prometheus 告警并自动响应。Operator 监听特定的指标或告警，当某个阈值被触发时，自动采取行动。

2. **自动修复应用**：例如，在某些指标超过阈值时，Operator 可以自动重启 Pod，或者增加副本数量，进行资源调度。

   **示例：**



   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: my-app
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: my-app
     template:
       metadata:
         labels:
           app: my-app
       spec:
         containers:
      - name: my-app
        image: my-app:latest
```

### **5\. 使用 PagerDuty、OpsGenie 等工具的自动化响应** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#5-%e4%bd%bf%e7%94%a8-pagerdutyopsgenie-%e7%ad%89%e5%b7%a5%e5%85%b7%e7%9a%84%e8%87%aa%e5%8a%a8%e5%8c%96%e5%93%8d%e5%ba%94)

PagerDuty 和 OpsGenie 是常用的告警响应工具，可以配置自动化响应策略，如自动重启服务、扩容等。

#### **步骤：** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%ad%a5%e9%aa%a4-4)

1. **集成告警通知**：配置 Prometheus 或其他监控工具将告警发送到 PagerDuty、OpsGenie 等工具。
2. **自动化响应配置**：这些工具支持通过 API 或自动化规则触发响应。可以配置自动创建事件并根据告警类型自动执行操作，如发送确认、执行故障修复等。

* * *

### **总结：** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%80%bb%e7%bb%93-5)

实现告警的自动化响应需要根据不同的需求和工具链来设计。常见的方法包括：

- 使用 **Prometheus + Alertmanager** 配置告警并集成 Webhook、Ansible、Terraform 等自动化工具。
- 通过 **ChatOps** 将告警响应与聊天工具集成。
- 使用 **Kubernetes Operator** 进行自定义的自动化修复。
- 使用告警管理工具（如 **PagerDuty** 或 **OpsGenie**）来响应告警。

自动化响应不仅能够提高运维效率，还能确保在发生故障时迅速做出反应，减少业务影响。

## Prometheus数据压缩和持久化实现原理 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#prometheus%e6%95%b0%e6%8d%ae%e5%8e%8b%e7%bc%a9%e5%92%8c%e6%8c%81%e4%b9%85%e5%8c%96%e5%ae%9e%e7%8e%b0%e5%8e%9f%e7%90%86)

Prometheus 是一个开源的监控系统，能够通过时序数据的方式进行数据采集、存储和查询。在 Prometheus 中，数据压缩和持久化是两个关键的设计问题，保证了系统能够高效地存储大量的时序数据。

### **1\. Prometheus 数据存储架构** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-prometheus-%e6%95%b0%e6%8d%ae%e5%ad%98%e5%82%a8%e6%9e%b6%e6%9e%84)

Prometheus 主要使用本地磁盘存储时序数据，并且在存储和压缩数据时采用了一些技术来确保其性能和可扩展性。其数据存储分为以下几个部分：

- **Time Series Database（TSDB）**：Prometheus 使用时序数据库存储时序数据。每个时间序列由一个唯一的标识符（即指标名称和标签组合）和一系列时间戳与数值对组成。
- **块（Blocks）**：Prometheus 的存储采用了块（Block）的方式，将数据按时间分块存储，每个块通常保存一个时间段的数据（例如两小时的数据）。块存储是压缩存储和持久化的基础。

### **2\. 数据存储与持久化原理** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-%e6%95%b0%e6%8d%ae%e5%ad%98%e5%82%a8%e4%b8%8e%e6%8c%81%e4%b9%85%e5%8c%96%e5%8e%9f%e7%90%86)

#### **2.1. 数据模型** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#21-%e6%95%b0%e6%8d%ae%e6%a8%a1%e5%9e%8b)

在 Prometheus 中，数据以时间序列（Time Series）的形式进行存储。每个时间序列由以下几个部分组成：

- **Metric Name（指标名）**：唯一标识一个时间序列的名称。
- **Labels（标签）**：用于标识该时间序列的额外维度（如 `instance`、`job` 等），标签是一个键值对（Key-Value），例如 `job="node_exporter"`。
- **Timestamp（时间戳）**：时间戳指示该数据点的时间，Prometheus 中的数据精度为毫秒级。
- **Value（值）**：每个时间戳对应的度量值。

#### **2.2. 数据写入过程** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#22-%e6%95%b0%e6%8d%ae%e5%86%99%e5%85%a5%e8%bf%87%e7%a8%8b)

当 Prometheus 从目标端点抓取数据时，数据会按照时间序列的形式进行存储。每个时间序列的每个数据点都会包含一个时间戳和相应的数值。这些数据会被按块进行存储。

Prometheus 采用 **时间分区（Time Partitioning）** 和 **块文件（Block Files）** 方式进行存储。数据会被分为多个 **块（Block）**，每个块包含一定时间范围（如两个小时或更长时间）的数据。

#### **2.3. 块存储结构** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#23-%e5%9d%97%e5%ad%98%e5%82%a8%e7%bb%93%e6%9e%84)

Prometheus 存储的每个块（Block）都有一个固定的时间跨度，通常为 **2小时**。每个块都包含以下信息：

- **索引文件**：记录了时间序列的元数据，例如标签和指标信息。
- **数据文件**：存储了每个时间序列的实际数据点，数据点按时间顺序排列。
- **压缩文件**：Prometheus 会对数据进行压缩，以减小存储空间。

每个块在磁盘上表现为一个目录，包含多个压缩后的数据文件（如 `.tsdb` 文件）。这些块会按时间顺序依次存储，并且在一段时间后进行合并。

#### **2.4. 数据压缩** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#24-%e6%95%b0%e6%8d%ae%e5%8e%8b%e7%bc%a9)

Prometheus 通过 **WAL（Write Ahead Log）** 和 **TSDB（Time Series Database）** 数据结构来实现高效的压缩和持久化。

- **WAL（Write-Ahead Log）**：在写入数据时，Prometheus 会先将数据写入 WAL 文件，确保数据不丢失。WAL 文件通常存储在磁盘上，当达到一定的大小后，会将 WAL 中的数据合并到 TSDB 的块中。
- **TSDB 数据结构**：TSDB 是 Prometheus 的核心数据存储引擎，它将时间序列数据按块（Block）存储，并使用一种基于 **LZ4 压缩算法** 的方法对时间序列数据进行压缩。每个数据块会按照一定的时间间隔（如 2 小时）来创建，并且每个数据块会进行压缩存储。压缩后的数据占用的磁盘空间远小于原始数据。

Prometheus 使用 **chunk encoding**（块编码）对时间序列数据进行存储和压缩，具体方法包括：

- **Delta Encoding**（增量编码）：通过记录相邻两个值之间的差异来减少存储空间。
- **Run-Length Encoding (RLE)**：对连续相同的值进行编码，以节省存储。
- **LZ4 Compression**：在块级别上使用 LZ4 压缩算法对数据进行压缩。

#### **2.5. 数据块合并** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#25-%e6%95%b0%e6%8d%ae%e5%9d%97%e5%90%88%e5%b9%b6)

Prometheus 会定期进行块的合并（Compaction）操作。块合并的目的是减少存储占用和提高查询效率。合并过程会把小的时间段合并成一个大的时间段，压缩率进一步提高，同时减少了存储碎片。

Prometheus 会定期清理过期数据，可以通过 `--storage.tsdb.retention.time` 参数来配置数据的保留时间。

### **3\. 数据压缩与持久化的实现原理总结** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-%e6%95%b0%e6%8d%ae%e5%8e%8b%e7%bc%a9%e4%b8%8e%e6%8c%81%e4%b9%85%e5%8c%96%e7%9a%84%e5%ae%9e%e7%8e%b0%e5%8e%9f%e7%90%86%e6%80%bb%e7%bb%93)

- **时间序列数据结构**：Prometheus 存储时序数据时，每个时间序列有唯一的标签组合和数值。
- **块存储机制**：数据按时间切分成多个块，每个块存储一定时间段的数据，通常为 2 小时。
- **压缩算法**：使用增量编码（Delta Encoding）、游程长度编码（Run-Length Encoding）以及 LZ4 压缩算法对数据进行压缩，减少存储空间。
- **合并与清理**：定期合并小的块并清理过期数据，确保存储空间的高效利用。
- **WAL 日志**：写操作首先记录到 WAL，确保数据不会丢失，然后再写入 TSDB 块中。

### **4\. 持久化存储** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-%e6%8c%81%e4%b9%85%e5%8c%96%e5%ad%98%e5%82%a8)

Prometheus 的数据持久化是基于本地磁盘的，数据会以块的形式进行存储，并且会进行定期的压缩和合并。数据的持久化时间可以通过配置文件进行设置。通常，Prometheus 的存储引擎是有限制的，适合中小型监控集群。

对于大规模环境，可能需要使用外部存储系统（如 **Thanos** 或 **Cortex**）来实现分布式存储和长期持久化。

## kubectl top输出与Linux free命令不一致原因☆ [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#kubectl-top%e8%be%93%e5%87%ba%e4%b8%8elinux-free%e5%91%bd%e4%bb%a4%e4%b8%8d%e4%b8%80%e8%87%b4%e5%8e%9f%e5%9b%a0)

`kubectl top` 和 Linux `free` 命令在输出上可能会有所不同，主要原因在于它们获取和显示的资源数据来源、计算方式和覆盖范围不同。具体来说：

### **1\. 数据来源的差异** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-%e6%95%b0%e6%8d%ae%e6%9d%a5%e6%ba%90%e7%9a%84%e5%b7%ae%e5%bc%82)

- **`kubectl top`**:
  - `kubectl top` 是 Kubernetes 的监控命令，主要通过 **metrics-server** 或 **Prometheus** 等监控工具提供集群的资源使用情况。它显示的是每个 Pod、节点、容器等在 Kubernetes 集群中的资源使用情况（如 CPU、内存等）。
  - **CPU 使用情况** 是基于 Kubernetes 中的容器实际请求和分配的资源数据。
  - **内存使用情况** 是基于容器在运行时的实际内存使用量（容器分配的内存可能不完全等于其实际使用的内存）。
- **`free` 命令**:
  - `free` 是 Linux 系统的标准命令，用于显示操作系统层面的内存使用情况。它显示的是整个操作系统级别的内存使用状态，包括系统内存（RAM）、交换内存（Swap）等。
  - `free` 命令报告的是整个机器上的物理内存和虚拟内存的使用情况，而不考虑容器或 Kubernetes 环境中的资源隔离。

### **2\. 资源的计算方式** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-%e8%b5%84%e6%ba%90%e7%9a%84%e8%ae%a1%e7%ae%97%e6%96%b9%e5%bc%8f)

- **`kubectl top`**:

  - **CPU**：`kubectl top` 显示的是容器实际消耗的 CPU 资源，通常以 **millicores**（mCPU）为单位（例如 500m 表示 0.5 核）。
  - **内存**：它显示的是容器实际使用的内存量，不包括系统内存中的缓存和缓冲区。容器的内存消耗可能与它所请求的内存有所不同。
- **`free` 命令**：

  - 物理内存：


    ```
    free
    ```


    显示的是操作系统级别的内存使用情况，通常会报告


    ```
    used
    ```


    、


    ```
    free
    ```


    、


    ```
    buffers
    ```


    、


    ```
    cached
    ```


    等字段。

    - `used`：已使用的内存，包括系统缓存和缓冲区的内存。
    - `free`：空闲的内存。
    - `buffers/cache`：用于缓存文件和 I/O 操作的内存。
  - **交换空间（Swap）**：`free` 还显示了交换空间的使用情况，这与容器的内存使用无关。

### **3\. 资源隔离和容器化** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-%e8%b5%84%e6%ba%90%e9%9a%94%e7%a6%bb%e5%92%8c%e5%ae%b9%e5%99%a8%e5%8c%96)

在 Kubernetes 环境中，容器的资源限制是与主机系统资源进行隔离的。每个容器（或 Pod）可以有自己的 CPU、内存限制和请求。`kubectl top` 是基于这些限制和容器实际消耗的资源来报告的。

而 `free` 命令显示的是整个系统级别的内存使用情况，不会区分容器之间的资源消耗。所以，`kubectl top` 中显示的内存和 CPU 使用量是基于容器内的使用情况，而 `free` 命令显示的是操作系统层面的资源使用，不区分容器之间的使用。

### **4\. 汇总与显示的不同** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-%e6%b1%87%e6%80%bb%e4%b8%8e%e6%98%be%e7%a4%ba%e7%9a%84%e4%b8%8d%e5%90%8c)

- **`kubectl top`**：是以 Pod、Node 或 Container 为单位显示资源使用情况。如果你查询的是 Pod 的资源使用情况，它将报告该 Pod 内所有容器的资源使用合计值，而这些容器的实际资源消耗和内存使用量是通过 Kubernetes 的监控组件（如 metrics-server）来汇报的。
- **`free` 命令**：显示的是整个操作系统（包括主机、所有容器和所有进程）的内存使用情况，它的输出包括操作系统级别的内存总量和已使用、空闲、缓存等的详细情况。

### **总结** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%80%bb%e7%bb%93-6)

`kubectl top` 和 `free` 命令显示的数据来源和计算方式不同：

- `kubectl top` 主要显示 Kubernetes 环境下的资源使用情况，依据容器的资源消耗。
- `free` 显示的是 Linux 系统层级的内存情况，包括操作系统的所有内存使用情况、缓存、缓冲区等。

因此，`kubectl top` 和 `free` 输出不一致是因为它们的数据来源不同、计算方式不同以及所显示的资源范围不同。

## 用到了哪些exporter，功能是什么在 Prometheus 监控中， **Exporter** 是一个重要的组件，它用于从不同的应用或系统中收集指标，并将这些指标以 Prometheus 能够理解的格式暴露出来。以下是一些常用的 Prometheus Exporter 及其功能： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e7%94%a8%e5%88%b0%e4%ba%86%e5%93%aa%e4%ba%9bexporter%e5%8a%9f%e8%83%bd%e6%98%af%e4%bb%80%e4%b9%88%e5%9c%a8-prometheus-%e7%9b%91%e6%8e%a7%e4%b8%adexporter-%e6%98%af%e4%b8%80%e4%b8%aa%e9%87%8d%e8%a6%81%e7%9a%84%e7%bb%84%e4%bb%b6%e5%ae%83%e7%94%a8%e4%ba%8e%e4%bb%8e%e4%b8%8d%e5%90%8c%e7%9a%84%e5%ba%94%e7%94%a8%e6%88%96%e7%b3%bb%e7%bb%9f%e4%b8%ad%e6%94%b6%e9%9b%86%e6%8c%87%e6%a0%87%e5%b9%b6%e5%b0%86%e8%bf%99%e4%ba%9b%e6%8c%87%e6%a0%87%e4%bb%a5-prometheus-%e8%83%bd%e5%a4%9f%e7%90%86%e8%a7%a3%e7%9a%84%e6%a0%bc%e5%bc%8f%e6%9a%b4%e9%9c%b2%e5%87%ba%e6%9d%a5%e4%bb%a5%e4%b8%8b%e6%98%af%e4%b8%80%e4%ba%9b%e5%b8%b8%e7%94%a8%e7%9a%84-prometheus-exporter-%e5%8f%8a%e5%85%b6%e5%8a%9f%e8%83%bd)

### 1\. **Node Exporter** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-node-exporter)

- **功能**：用于收集操作系统级别的硬件和操作系统指标，涵盖 CPU、内存、磁盘、网络等基本资源的使用情况。

- 监控内容

：

  - CPU 使用率
  - 内存使用情况
  - 磁盘使用情况（包括 I/O 速率、磁盘空间等）
  - 网络流量
  - 系统负载
- **适用场景**：监控 Linux/Unix 系统的资源使用情况。


### 2\. **kube-state-metrics** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-kube-state-metrics)

- **功能**：用于从 Kubernetes 集群中收集关于 Kubernetes 资源的状态指标。不同于 `Node Exporter` 采集操作系统的指标，`kube-state-metrics` 采集的是 Kubernetes 资源（如 Pod、Deployment、ReplicaSet、StatefulSet、Node 等）的状态信息。

- 监控内容

：

  - Pod 状态（运行、待启动、失败等）
  - Deployment 和 ReplicaSet 的副本数
  - 节点状态（Ready、NotReady）
  - 资源请求与限制
- **适用场景**：监控 Kubernetes 集群的资源状态和健康状况。


### 3\. **cAdvisor** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-cadvisor)

- **功能**：用于收集 Docker 容器的资源使用情况，提供 CPU、内存、磁盘 I/O、网络使用等容器级别的指标。

- 监控内容

：

  - 每个 Docker 容器的 CPU、内存使用情况
  - 容器的网络流量
  - 容器的磁盘 I/O
  - 容器的生命周期事件（启动、停止等）
- **适用场景**：监控 Docker 容器的资源消耗和性能。


### 4\. **Blackbox Exporter** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-blackbox-exporter)

- **功能**：用于通过黑盒方式（模拟用户行为）检查服务的可用性和响应时间。它支持 HTTP、HTTPS、DNS、TCP 等协议的健康检查。

- 监控内容

：

  - HTTP/HTTPS 状态码监控
  - DNS 查询响应时间
  - TCP 端口的可达性
- **适用场景**：检测外部服务的可用性，检查网站或其他网络服务的健康状态。


### 5\. **MySQL Exporter** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#5-mysql-exporter)

- **功能**：专门用于收集 MySQL 数据库的指标，监控 MySQL 数据库的性能和健康状况。

- 监控内容

：

  - 数据库的查询性能
  - 慢查询日志
  - 数据库的连接数
  - 数据库的缓存使用
  - 磁盘 I/O 等
- **适用场景**：监控 MySQL 数据库的性能，帮助诊断数据库性能瓶颈。


### 6\. **PostgreSQL Exporter** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#6-postgresql-exporter)

- **功能**：用于从 PostgreSQL 数据库中收集指标，类似于 MySQL Exporter，但专为 PostgreSQL 定制。

- 监控内容

：

  - 数据库的连接数
  - 缓存和缓存命中率
  - 活跃查询数量
  - 数据库大小
- **适用场景**：监控 PostgreSQL 数据库的性能和健康。


### 7\. **JMX Exporter** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#7-jmx-exporter)

- **功能**：用于从 Java 应用中收集 JMX（Java Management Extensions）暴露的指标，适用于基于 Java 的应用，如 Kafka、Tomcat、JVM 等。

- 监控内容

：

  - JVM 内存使用情况
  - 垃圾回收统计
  - 线程池使用情况
  - Kafka、Tomcat 等 Java 服务的性能指标
- **适用场景**：监控 Java 应用和 JVM 的性能，尤其是在大规模生产环境中。


### 8\. **Redis Exporter** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#8-redis-exporter)

- **功能**：用于监控 Redis 数据库的性能和健康。

- 监控内容

：

  - Redis 内存使用
  - 命中率
  - 锁和连接
  - 键空间使用情况
- **适用场景**：监控 Redis 数据库的性能，确保 Redis 运行健康。


### 9\. **MongoDB Exporter** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#9-mongodb-exporter)

- **功能**：用于监控 MongoDB 数据库的性能和状态。

- 监控内容

：

  - 数据库的连接数
  - 查询性能
  - 内存使用情况
  - 复制集状态
- **适用场景**：监控 MongoDB 数据库，诊断数据库性能问题。


### 10\. **Elasticsearch Exporter** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#10-elasticsearch-exporter)

- **功能**：用于从 Elasticsearch 集群中收集指标，监控 Elasticsearch 的健康和性能。

- 监控内容

：

  - 集群的健康状态
  - 节点和索引的性能
  - 文档数、查询性能等
- **适用场景**：监控 Elasticsearch 集群，确保其高效运行。


### 11\. **Nginx Exporter** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#11-nginx-exporter)

- **功能**：用于从 Nginx 服务器收集指标，帮助监控 Nginx 的性能。

- 监控内容

：

  - Nginx 请求和响应数
  - Nginx 错误日志
  - Nginx 的连接数
- **适用场景**：监控 Nginx 的负载和性能，诊断问题。


### 12\. **Kafka Exporter** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#12-kafka-exporter)

- **功能**：用于监控 Kafka 集群的健康和性能。

- 监控内容

：

  - 消费者组和主题的偏移量
  - 请求和响应延迟
  - 磁盘和内存使用
- **适用场景**：监控 Kafka 集群的健康状况，确保消息传递的高效性。


### **总结**： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%80%bb%e7%bb%93-7)

这些 Exporter 的共同目标是将各个系统、服务或应用的性能指标暴露给 Prometheus，以便对其进行监控和告警。通过合适的 Exporter，我们可以详细了解每个服务或系统的运行状态、资源使用情况和潜在问题，帮助运维人员做出及时响应和优化决策。

## 是否自己开发过exporter☆ [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%98%af%e5%90%a6%e8%87%aa%e5%b7%b1%e5%bc%80%e5%8f%91%e8%bf%87exporter)

是的，我有开发过自定义 **Exporter** 的经验。在某些场景下，现有的 Prometheus Exporter 可能无法满足特定的监控需求，这时就需要开发一个自定义 Exporter 来暴露特定服务或应用的指标。

### 自定义 Exporter 开发的一般步骤： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e8%87%aa%e5%ae%9a%e4%b9%89-exporter-%e5%bc%80%e5%8f%91%e7%9a%84%e4%b8%80%e8%88%ac%e6%ad%a5%e9%aa%a4)

1. **选择语言和框架**：常见的开发语言有 Go、Python、Java 等。Go 是官方推荐的语言，因为 Prometheus 本身就是用 Go 编写的，并且提供了官方的客户端库（`prometheus/client_golang`）。
2. **创建指标**：
   - 使用 Prometheus 的客户端库来定义所需的指标（例如，计数器、直方图、摘要等）。
   - 通过 Prometheus 提供的 `http.Handler` 向 Prometheus 暴露这些指标。
3. **收集数据**：
   - 通过代码收集系统或应用的特定数据（如应用的健康状态、性能指标等）。这通常涉及访问 API、日志文件或数据库等。
4. **启动 HTTP 服务器**：
   - 启动一个 HTTP 服务器，并监听 Prometheus 进行指标抓取的端口。
5. **定期更新指标**：
   - 可以通过定时任务（如 `time.Ticker`）定期更新或重新抓取需要监控的数据。

### 示例：用 Go 开发一个简单的自定义 Exporter [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e7%a4%ba%e4%be%8b%e7%94%a8-go-%e5%bc%80%e5%8f%91%e4%b8%80%e4%b8%aa%e7%ae%80%e5%8d%95%e7%9a%84%e8%87%aa%e5%ae%9a%e4%b9%89-exporter)

这是一个简单的 Go 语言示例，创建了一个自定义的 **Exporter** 来监控一个假设的服务。

```go
package main

import (
	"fmt"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"net/http"
	"time"
)

// 创建一个自定义的指标
var (
	upGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "my_service_up",
			Help: "Indicates whether my service is up (1) or down (0).",
		},
		[]string{"service_name"},
	)
)

func init() {
	// 注册自定义指标
	prometheus.MustRegister(upGauge)
}

func recordMetrics() {
	// 模拟服务状态更新
	go func() {
		for {
			upGauge.WithLabelValues("my_service").Set(1) // 服务状态为 up
			time.Sleep(5 * time.Second)
			upGauge.WithLabelValues("my_service").Set(0) // 服务状态为 down
			time.Sleep(5 * time.Second)
		}
	}()
}

func main() {
	// 启动一个 HTTP 服务，暴露 Prometheus 的指标
	http.Handle("/metrics", promhttp.Handler())

	// 开启服务状态的模拟
	recordMetrics()

	// 启动 HTTP 服务器
	fmt.Println("Starting custom exporter on :8080")
	http.ListenAndServe(":8080", nil)
}
```

### 关键点： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e5%85%b3%e9%94%ae%e7%82%b9)

- **upGauge** 是一个自定义的 `GaugeVec` 类型指标，用来表示服务是否可用。
- `recordMetrics` 函数模拟每 5 秒更新一次服务状态：服务处于 “up” 或 “down” 状态。
- `/metrics` 路径用于暴露 Prometheus 所需的指标。

### 开发过程中可能遇到的挑战： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e5%bc%80%e5%8f%91%e8%bf%87%e7%a8%8b%e4%b8%ad%e5%8f%af%e8%83%bd%e9%81%87%e5%88%b0%e7%9a%84%e6%8c%91%e6%88%98)

1. **数据源的访问问题**：在自定义 Exporter 中，数据的收集源可能是数据库、API 或其他外部系统，访问这些数据源可能需要额外的权限或安全配置。
2. **性能问题**：当需要频繁收集大量数据时，确保 Exporter 性能高效是非常重要的，否则可能对被监控服务造成影响。
3. **监控指标设计**：设计好的指标对于后续的监控和告警非常关键，要确保指标具有可理解性、足够的细粒度，且符合 Prometheus 的最佳实践。

开发自定义 Exporter 的关键是根据具体需求暴露合适的指标，并确保能够高效、稳定地运行。

## target down的情况如何进行故障排除？ [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#target-down%e7%9a%84%e6%83%85%e5%86%b5%e5%a6%82%e4%bd%95%e8%bf%9b%e8%a1%8c%e6%95%85%e9%9a%9c%e6%8e%92%e9%99%a4)

当 Prometheus 报告某个 **target down**（目标不可达）时，意味着 Prometheus 无法从该目标抓取到数据。这种情况的故障排除可以通过以下几个步骤进行：

### 1\. **检查 Prometheus 日志** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-%e6%a3%80%e6%9f%a5-prometheus-%e6%97%a5%e5%bf%97)

- 查看 Prometheus 的日志文件，以便识别是否有任何网络错误、身份验证失败或超时等信息。
- 日志中通常会给出详细的错误信息，例如：
  - “connection refused”（拒绝连接）
  - “timeout”（超时）
  - “authentication failure”（认证失败）

### 2\. **检查网络连接** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-%e6%a3%80%e6%9f%a5%e7%bd%91%e7%bb%9c%e8%bf%9e%e6%8e%a5)

- Ping 目标服务

：首先确认 Prometheus 是否能够通过网络连接到目标机器或服务。使用


```
ping
```


或


```
telnet
```


测试与目标端口的连接。



```bash
ping <target_host>
telnet <target_host> <target_port>
```

- **防火墙和网络策略**：确保 Prometheus 服务器与目标服务之间没有防火墙规则或网络策略阻止通信。检查防火墙设置，确保相应端口（通常是 80/443 或 Prometheus 的抓取端口）是开放的。


### 3\. **检查目标服务状态** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-%e6%a3%80%e6%9f%a5%e7%9b%ae%e6%a0%87%e6%9c%8d%e5%8a%a1%e7%8a%b6%e6%80%81)

- 如果目标服务是一个 Web 服务或应用程序，确保它正在运行并且能够正常响应请求。

  - Web 服务

    ：可以直接在浏览器中或使用


    ```
    curl
    ```


    请求目标服务的


    ```
    /metrics
    ```


    端点：



    ```bash
    curl http://<target_host>:<port>/metrics
    ```

  - **服务健康检查**：如果目标服务提供健康检查 API，先检查健康状态是否正常。

### 4\. **检查目标端点的配置** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-%e6%a3%80%e6%9f%a5%e7%9b%ae%e6%a0%87%e7%ab%af%e7%82%b9%e7%9a%84%e9%85%8d%e7%bd%ae)

- 确保 Prometheus 配置文件中的抓取端点是正确的，特别是 URL 和端口。如果 URL 或端口配置错误，Prometheus 将无法抓取数据。
- 检查 `scrape_configs` 部分，确保目标服务的地址和端口正确，并且没有拼写错误或格式问题。

### 5\. **目标服务的负载问题** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#5-%e7%9b%ae%e6%a0%87%e6%9c%8d%e5%8a%a1%e7%9a%84%e8%b4%9f%e8%bd%bd%e9%97%ae%e9%a2%98)

- 如果目标服务在高负载下，可能会导致响应时间过长，从而导致 Prometheus 无法及时抓取数据。检查目标服务的负载和性能指标（如 CPU、内存使用率等），并进行优化。
- **资源不足**：检查目标机器的资源使用情况（例如 CPU、内存、磁盘 I/O），如果负载过高，可能会导致响应超时或服务中断。

### 6\. **身份验证问题** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#6-%e8%ba%ab%e4%bb%bd%e9%aa%8c%e8%af%81%e9%97%ae%e9%a2%98)

- 如果目标服务需要身份验证（如 HTTP 基本认证、OAuth 等），确保 Prometheus 配置文件中正确设置了身份验证信息。
- 在 Prometheus 配置文件的 `scrape_configs` 部分中，检查是否需要添加身份验证的 `basic_auth` 或 `bearer_token` 等参数。

### 7\. **检查 Prometheus 配置文件** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#7-%e6%a3%80%e6%9f%a5-prometheus-%e9%85%8d%e7%bd%ae%e6%96%87%e4%bb%b6)

- 确保 `scrape_interval` 配置合理。如果抓取间隔太短，目标可能无法及时响应，导致抓取失败。适当增大抓取间隔，以避免过度的压力。
- 还要检查是否设置了 **scrape\_timeout**，如果目标响应超时，可能会导致 `target down`。

### 8\. **查看 Prometheus 目标状态** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#8-%e6%9f%a5%e7%9c%8b-prometheus-%e7%9b%ae%e6%a0%87%e7%8a%b6%e6%80%81)

- 在 Prometheus 的 Web UI 中，访问 `Status -> Targets` 页面，查看目标的状态、最后一次抓取的时间和任何错误信息。
- 目标页面通常会显示一些详细的错误信息，帮助你进一步诊断问题。例如，是否存在 “http\_errors” 或 “timeouts”。

### 9\. **重启 Prometheus** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#9-%e9%87%8d%e5%90%af-prometheus)

- 如果所有设置看起来都正确，但 Prometheus 仍然无法连接到目标服务，可以尝试重启 Prometheus。这有时可以解决与网络连接或服务状态相关的临时问题。

### 10\. **目标端服务日志** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#10-%e7%9b%ae%e6%a0%87%e7%ab%af%e6%9c%8d%e5%8a%a1%e6%97%a5%e5%bf%97)

- 查看目标服务的日志文件，检查是否有任何与 Prometheus 抓取相关的错误或警告信息。
- 如果目标服务暴露在容器或虚拟机中，确保该容器或虚拟机运行正常且没有资源瓶颈。

### 故障排除示例： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%95%85%e9%9a%9c%e6%8e%92%e9%99%a4%e7%a4%ba%e4%be%8b)

假设 Prometheus 显示一个目标为 **down**，你可以通过以下步骤进行排查：

1. 在 Prometheus UI 中查看目标的详细信息，发现错误信息为 `connection refused`。
2. 使用 `telnet` 测试与目标端口的连接，发现连接不上。
3. 检查目标服务的状态，发现目标服务未启动，尝试重启目标服务。
4. 再次查看 Prometheus UI，发现目标状态变为 **up**，问题解决。

### 总结 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%80%bb%e7%bb%93-8)

解决 `target down` 问题通常涉及以下几个方面的检查：

1. 网络连接和防火墙。
2. 目标服务的可用性。
3. Prometheus 配置和抓取端点。
4. 目标服务的负载和身份验证。

通过有序的排查，你应该能够找出并解决目标服务无法抓取的根本原因。

## Exporter 停止工作，如何监控？ [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#exporter-%e5%81%9c%e6%ad%a2%e5%b7%a5%e4%bd%9c%e5%a6%82%e4%bd%95%e7%9b%91%e6%8e%a7)

监控 **Exporter** 停止工作是确保系统可靠性和健康的关键步骤。Exporter 停止工作时，Prometheus 无法抓取指标，这会影响你的监控和报警系统。因此，监控 Exporter 本身的健康状态也是至关重要的。以下是一些监控 Exporter 停止工作的方法：

### 1\. **Prometheus 自身的监控** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-prometheus-%e8%87%aa%e8%ba%ab%e7%9a%84%e7%9b%91%e6%8e%a7)

- Target 状态

：Prometheus 会在 Web UI 上显示抓取目标的健康状态。你可以监控

Prometheus Targets

页面的


```
last scrape
```


时间，看看目标是否仍然处于正常抓取状态。

  - 通过 `Status -> Targets` 页面，你可以查看每个目标的最后抓取时间、抓取状态以及相关的错误信息。通过监控这些信息，可以确保 Exporter 是否正常工作。
- Prometheus 任务失败报警

：Prometheus 可以设置报警规则，当某个 Exporter 长时间没有抓取到数据时发出报警。例如，


```
up
```


指标为 0 表示目标不可达，可以设置告警规则。



```yaml
groups:
  - name: exporters
    rules:
      - alert: ExporterDown
        expr: up{job="your_exporter_job"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Exporter is down for more than 5 minutes"
          description: "The exporter for {{ $labels.instance }} is not reachable for the last 5 minutes."
```

这种方式可以确保在 Exporter 停止工作时及时通知你。

### 2\. **Exporter 的健康检查** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-exporter-%e7%9a%84%e5%81%a5%e5%ba%b7%e6%a3%80%e6%9f%a5)

- **HTTP 健康检查端点**：如果你的 Exporter 支持健康检查端点（如 `/health` 或 `/metrics`），你可以通过 Prometheus 或其他监控工具定期检查该端点的响应状态。

- 自定义 Exporter 健康指标

  ：许多 Exporter 提供了健康状态或内部指标，如


  ```
  up
  ```


  、


  ```
  health
  ```


  、


  ```
  status
  ```


  等。如果没有，可以考虑自定义实现，暴露一个指标来显示 Exporter 是否正常工作。例如：



  ```bash
  # 在你的 Exporter 中加入 /metrics 端点监控服务状态
  # 例如：up{job="my_exporter"} 1
  ```



  Prometheus 可以抓取并在不正常时报警。


### 3\. **Exporter 进程监控** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-exporter-%e8%bf%9b%e7%a8%8b%e7%9b%91%e6%8e%a7)

- **进程监控**：使用 **node\_exporter** 或 **prometheus-node-exporter** 来监控 Exporter 的进程状态。你可以查看进程是否仍在运行，并监控 Exporter 的资源使用情况（CPU、内存、磁盘等）。

- Linux 进程监控

  ：你可以使用


  ```
  ps
  ```


  、


  ```
  top
  ```


  或


  ```
  systemd
  ```


  等工具查看 Exporter 进程是否正常运行。比如：



  ```bash
  ps aux | grep <exporter_name>
  ```



  如果进程停止，你可以设置自动重启进程的机制（如 systemd 自动重启）。


### 4\. **外部服务监控** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-%e5%a4%96%e9%83%a8%e6%9c%8d%e5%8a%a1%e7%9b%91%e6%8e%a7)

- 容器化 Exporter

  ：如果 Exporter 运行在容器中，可以使用容器监控工具（如 Kubernetes 或 Docker）来监控容器的健康状态。


  - 在 Kubernetes 中，你可以配置健康检查（liveness 和 readiness probe）来检查 Exporter 是否可用。这样，当 Exporter 停止时，Kubernetes 会自动重启容器。
  - 在 Docker 中，你可以设置容器的健康检查，监控容器内的服务状态。

```bash
docker inspect --format '{{json .State.Health}}' <container_id>
```

### 5\. **日志监控** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#5-%e6%97%a5%e5%bf%97%e7%9b%91%e6%8e%a7)

- **Exporter 日志**：很多 Exporter 都有日志输出，监控日志是排查问题的好方法。通过 **log shipper**（如 **Fluentd**、 **Logstash**、 **Filebeat** 等）将日志收集到 ELK 或其他日志管理系统中，检测是否存在错误或崩溃的日志。
- **错误日志分析**：当 Exporter 停止工作时，通常会有相关错误日志。可以通过分析日志内容来发现故障的根本原因。

### 6\. **自动化修复措施** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#6-%e8%87%aa%e5%8a%a8%e5%8c%96%e4%bf%ae%e5%a4%8d%e6%8e%aa%e6%96%bd)

- 自动重启 Exporter

  ：如果 Exporter 停止工作，可以配置自动重启机制。通过

  systemd

  或

  Supervisor

  等工具，确保 Exporter 在停止后能够自动重启。

  - 在 systemd 中，你可以配置如下：



    ```ini
    [Unit]
    Description=Exporter

    [Service]
    ExecStart=/path/to/exporter
    Restart=always
    RestartSec=3

    [Install]
    WantedBy=multi-user.target
    ```

  - 通过设置 `Restart=always`，可以确保 Exporter 在任何意外停止后被自动重启。

### 7\. **外部监控工具** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#7-%e5%a4%96%e9%83%a8%e7%9b%91%e6%8e%a7%e5%b7%a5%e5%85%b7)

- **黑盒监控**：你还可以使用外部监控工具（如 **Pingdom** 或 **UptimeRobot**）对 Exporter 的 `/metrics` 端点进行黑盒监控，确保服务是可达的。
- **Alertmanager 集成**：通过 Prometheus 的 **Alertmanager** 集成，可以设置复杂的报警规则，进行多渠道通知（如 Slack、邮件、短信等），确保在 Exporter 停止工作时能够及时通知运维人员。

### 8\. **分析指标丢失** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#8-%e5%88%86%e6%9e%90%e6%8c%87%e6%a0%87%e4%b8%a2%e5%a4%b1)

- 如果 Exporter 停止工作，Prometheus 会在抓取该目标时丢失指标。通过监控 Prometheus 的 **scrape\_latency** 和 **scrape\_errors** 指标，可以分析抓取时的延迟和错误，发现 Exporter 是否有问题。

### 总结 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%80%bb%e7%bb%93-9)

监控 Exporter 停止工作主要是确保其 **可达性**、 **健康状态**、 **进程运行状态**，并通过 Prometheus、外部工具以及日志监控等手段确保及时发现问题并进行自动恢复。通过配置合理的告警机制，当 Exporter 停止工作时，能够及时采取应对措施，确保监控系统的高可用性。

## Prometheus的拉取模式与zabbix推送模式有何区别？各有什么优缺点？ [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#prometheus%e7%9a%84%e6%8b%89%e5%8f%96%e6%a8%a1%e5%bc%8f%e4%b8%8ezabbix%e6%8e%a8%e9%80%81%e6%a8%a1%e5%bc%8f%e6%9c%89%e4%bd%95%e5%8c%ba%e5%88%ab%e5%90%84%e6%9c%89%e4%bb%80%e4%b9%88%e4%bc%98%e7%bc%ba%e7%82%b9)

Prometheus 的拉取模式（Pull Model）和 Zabbix 的推送模式（Push Model）在数据收集和监控架构上有很大的区别。以下是这两种模式的比较，包括它们的优缺点：

### 1\. **Prometheus 拉取模式（Pull Model）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-prometheus-%e6%8b%89%e5%8f%96%e6%a8%a1%e5%bc%8fpull-model)

**原理**：Prometheus 作为监控系统，定期主动从被监控的目标（如应用程序、数据库、服务器等）中拉取（scrape）指标数据。

**工作方式**：

- Prometheus 配置了多个 `targets`（目标），每个目标都有一个暴露 `/metrics` 端点（通常是 HTTP）。
- Prometheus 定期访问这些端点拉取数据。
- 拉取数据的间隔、超时等都可以在 Prometheus 配置文件中进行配置。

**优点**：

- **去中心化管理**：Prometheus 主动从多个目标拉取数据，不需要在被监控的系统上安装任何推送的客户端或服务。
- **无客户端配置**：由于拉取数据是由 Prometheus 自己发起的，被监控的目标不需要知道 Prometheus 的存在，配置较为简单。
- **灵活性高**：可以方便地配置抓取间隔、拉取超时等参数，灵活应对不同目标的监控需求。
- **支持服务发现**：Prometheus 支持多种服务发现机制（如 Kubernetes、Consul、DNS），能够动态发现并监控新的目标。

**缺点**：

- **依赖目标可达性**：拉取模式要求 Prometheus 能够访问所有目标，如果目标不可达，Prometheus 就无法获取数据。
- **网络带宽消耗较大**：由于 Prometheus 定期拉取数据，可能会产生大量的网络请求，尤其是在大规模部署时，对网络带宽有较大的消耗。
- **延迟较高**：拉取模式下，Prometheus 的数据采集间隔较为固定，可能会出现数据有一定延迟（例如每 15 秒或 1 分钟才抓取一次）。

* * *

### 2\. **Zabbix 推送模式（Push Model）** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-zabbix-%e6%8e%a8%e9%80%81%e6%a8%a1%e5%bc%8fpush-model)

**原理**：Zabbix 通过监控代理或客户端，将监控数据主动推送到 Zabbix 服务器。

**工作方式**：

- Zabbix 在被监控目标上安装了监控代理（Zabbix Agent），该代理负责收集本机的各种指标。
- 代理将数据推送到 Zabbix Server 或 Zabbix Proxy，Zabbix 服务器定期处理这些数据并生成报警、报告等。

**优点**：

- **即刻收集数据**：数据推送到服务器后，几乎实时可以看到监控数据，延迟较低。
- **适应动态环境**：在一些目标较为动态（如容器化环境）时，推送模式可以更加灵活地处理新目标的加入。
- **目标无须暴露端口**：由于数据是从代理端推送到服务器，目标系统不需要暴露 HTTP 端口或其他服务端点，安全性相对较高。
- **支持主动报警**：Zabbix 代理可在监控指标达到预警条件时主动向服务器报告，减少了服务器端的负担。

**缺点**：

- **代理部署**：需要在每个被监控的目标上部署 Zabbix 代理，这可能增加配置和维护的复杂度，尤其在大规模环境中。
- **代理故障风险**：如果 Zabbix 代理停止工作或遇到网络问题，数据就无法推送到 Zabbix 服务器，可能导致监控数据丢失。
- **复杂的网络配置**：需要处理网络访问控制、防火墙等问题，因为被监控的目标需要推送数据到 Zabbix 服务器或代理。
- **不适合大量目标的实时监控**：如果目标非常多，推送模式可能会造成代理和服务器之间的网络压力。

* * *

### 3\. **对比总结** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-%e5%af%b9%e6%af%94%e6%80%bb%e7%bb%93)

| 特性 | **Prometheus 拉取模式** | **Zabbix 推送模式** |
| --- | --- | --- |
| **数据采集方式** | Prometheus 定期拉取数据 | 目标主动推送数据 |
| **配置复杂度** | 无需在目标系统上配置代理 | 需要在每个目标安装代理 |
| **扩展性** | 容易扩展（使用服务发现） | 扩展时需要管理更多的代理 |
| **延迟** | 可能有较高延迟（取决于抓取间隔） | 数据推送较为实时 |
| **安全性** | 需要暴露端口来拉取数据 | 推送数据不需要暴露端口 |
| **可靠性** | 如果目标不可达，会丢失数据 | 如果代理或网络中断，数据丢失 |
| **网络负载** | 定期拉取，可能产生大量请求 | 推送数据，如果目标不稳定则可能造成压力 |
| **适应动态环境（如容器化）** | 适应性较强，通过服务发现动态添加目标 | 需要动态更新代理配置 |
| **报警机制** | 报警通过 Prometheus 的查询和规则引擎触发 | 代理可以主动推送报警信息 |

* * *

### 4\. **适用场景** [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-%e9%80%82%e7%94%a8%e5%9c%ba%e6%99%af-3)

- **Prometheus 拉取模式**：适用于大规模、容器化环境，能够高效地处理动态变化的目标，并且通过灵活的配置和服务发现机制来支持自动化监控。
- **Zabbix 推送模式**：适用于需要实时数据且目标可以主动推送数据的环境，尤其适合传统的 IT 基础设施，并且目标系统的代理能够稳定运行。

### 总结 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%80%bb%e7%bb%93-10)

- **Prometheus 的拉取模式** 更适合大规模、容器化、云原生环境，支持自动化、灵活性高，但对网络依赖较大。
- **Zabbix 的推送模式** 更适合传统环境，支持实时监控，但需要在每个目标上部署代理，且有一定的维护成本。

两者各有优缺点，根据具体的应用场景和基础设施需求来选择适合的方案。

## Prometheus operator怎么添加targets和告警规则 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#prometheus-operator%e6%80%8e%e4%b9%88%e6%b7%bb%e5%8a%a0targets%e5%92%8c%e5%91%8a%e8%ad%a6%e8%a7%84%e5%88%99)

**Prometheus Operator** 是一个用来在 Kubernetes 上管理 Prometheus 部署的工具，它简化了 Prometheus 的创建、配置、管理和扩展。通过 **Prometheus Operator**，你可以通过 Kubernetes 资源对象来管理 Prometheus 集群的配置、告警规则和服务发现（targets）。

以下是如何通过 **Prometheus Operator** 来添加 **targets** 和 **告警规则** 的步骤。

### 1\. 添加 Targets (服务发现) [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-%e6%b7%bb%e5%8a%a0-targets-%e6%9c%8d%e5%8a%a1%e5%8f%91%e7%8e%b0)

在 **Prometheus Operator** 中，目标 (targets) 是通过 **ServiceMonitor** 或 **PodMonitor** 资源来配置的。ServiceMonitor 用于监控服务，PodMonitor 用于监控单个 Pod。

#### 步骤： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%ad%a5%e9%aa%a4-5)

1. 创建一个 **ServiceMonitor** 资源对象来配置 Prometheus 监控的服务。

   **示例：创建 ServiceMonitor 配置**



   ```yaml
   apiVersion: monitoring.coreos.com/v1
   kind: ServiceMonitor
   metadata:
     name: example-servicemonitor
     namespace: monitoring
   spec:
     selector:
       matchLabels:
         app: example-app
     endpoints:
  - port: web
    interval: 30s
```

   - `metadata.name`: 定义 ServiceMonitor 的名称。
   - `spec.selector`: 用于选择匹配的 Kubernetes 服务。
   - `spec.endpoints`: 定义 Prometheus 通过该端点进行监控的方式，通常是一个端口和拉取数据的间隔。
2. 在 Kubernetes 中应用 ServiceMonitor 配置：



```bash
kubectl apply -f servicemonitor.yaml
```

3. 确保 Prometheus Operator 配置了适当的 **Prometheus** 实例和 **ServiceMonitorSelector**。在 Prometheus 配置中，您可以通过 **Prometheus CRD**（Custom Resource Definitions）来指定哪些 ServiceMonitor 对象需要被 Prometheus 监控。

**示例：Prometheus CRD 配置**



```yaml
apiVersion: monitoring.coreos.com/v1
kind: Prometheus
metadata:
     name: prometheus
     namespace: monitoring
spec:
     replicas: 1
     serviceMonitorSelector:
       matchLabels:
         monitoring: enabled
```



这个配置使得 Prometheus 只选择带有 `monitoring: enabled` 标签的 ServiceMonitor 对象进行监控。


### 2\. 添加告警规则 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-%e6%b7%bb%e5%8a%a0%e5%91%8a%e8%ad%a6%e8%a7%84%e5%88%99)

Prometheus 的告警规则通常通过 **PrometheusRule** 资源来定义。告警规则定义了触发条件以及如何通知。

#### 步骤： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%ad%a5%e9%aa%a4-6)

1. 创建一个 **PrometheusRule** 资源对象来定义告警规则。

**示例：创建 PrometheusRule 配置**



```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
     name: example-alert-rules
     namespace: monitoring
spec:
     groups:
  - name: example-alerts
    rules:
    - alert: HighMemoryUsage
      expr: container_memory_usage_bytes{container="example-container"} > 500000000
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Memory usage is over 500MB for 5 minutes"
        description: "Container {{ $labels.container }} is using more than 500MB of memory."
```

   - `alert`: 告警的名称。
   - `expr`: 用 Prometheus 查询语言（PromQL）定义的告警表达式。此表达式监测容器内存使用情况，超过 500MB 触发告警。
   - `for`: 设置告警触发的持续时间。例如，5 分钟内内存使用超过 500MB 才触发告警。
   - `labels`: 添加一些自定义标签，可以在通知时使用。
   - `annotations`: 提供告警的详细信息，通常包括摘要和描述。
2. 在 Kubernetes 中应用 PrometheusRule 配置：



   ```bash
   kubectl apply -f prometheusrule.yaml
   ```

3. 确保 Prometheus 配置了告警规则。

   **示例：Prometheus CRD 配置**



   ```yaml
   apiVersion: monitoring.coreos.com/v1
   kind: Prometheus
   metadata:
     name: prometheus
     namespace: monitoring
   spec:
     replicas: 1
     ruleSelector:
       matchLabels:
         alert: enabled
   ```



   这个配置使得 Prometheus 只选择带有 `alert: enabled` 标签的 PrometheusRule 对象作为告警规则。


### 3\. 告警通知 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-%e5%91%8a%e8%ad%a6%e9%80%9a%e7%9f%a5)

一旦告警规则配置完成，Prometheus 会根据定义的规则评估和触发告警。如果你还需要告警通知（如发送邮件、Slack、PagerDuty 等），可以配置 **Alertmanager** 来处理告警通知。

**示例：Alertmanager 配置（通过 Kubernetes ConfigMap）**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
name: alertmanager-config
namespace: monitoring
data:
alertmanager.yml: |
    global:
      resolve_timeout: 5m
    route:
      receiver: 'slack-notifications'
    receivers:
      - name: 'slack-notifications'
        slack_configs:
          - channel: '#alerts'
            send_resolved: true
```

**应用 Alertmanager 配置**：

```bash
kubectl apply -f alertmanager-config.yaml
```

### 总结 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%80%bb%e7%bb%93-11)

- **添加 Targets**：通过 **ServiceMonitor** 或 **PodMonitor** 来指定 Prometheus 需要监控的目标（服务或 Pod）。
- **添加告警规则**：通过 **PrometheusRule** 来定义 Prometheus 的告警规则。
- **告警通知**：告警会触发后，通过 **Alertmanager** 发送通知，您可以配置通知渠道，如 Slack、邮件等。

在实际使用中，可以通过 **Prometheus Operator** 来自动化和简化这些配置的管理。

## k8s集群外exporter怎么使用Prometheus监控 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#k8s%e9%9b%86%e7%be%a4%e5%a4%96exporter%e6%80%8e%e4%b9%88%e4%bd%bf%e7%94%a8prometheus%e7%9b%91%e6%8e%a7)

在 Kubernetes 集群外部使用 **Prometheus** 监控 **Exporter**，你可以通过设置 **Prometheus** 来直接从外部系统或服务拉取指标数据。这里的关键是配置 **Prometheus** 以使其能够连接到外部的 **Exporter** 并获取数据。

以下是实现的步骤：

### 1\. 确保外部 Exporter 可访问 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#1-%e7%a1%ae%e4%bf%9d%e5%a4%96%e9%83%a8-exporter-%e5%8f%af%e8%ae%bf%e9%97%ae)

首先，确保外部 **Exporter** 可以被 Prometheus 访问。 **Exporter** 可以是运行在物理机、虚拟机或云实例上的应用程序，或者是其他容器化环境中的服务。

- 外部 **Exporter** 需要暴露一个 HTTP 服务接口，Prometheus 可以通过 HTTP 拉取指标。通常， **Exporter** 会暴露在一个特定端口（例如 9100、9182、8080 等）。

### 2\. 配置 Prometheus 来抓取外部 Exporter 数据 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#2-%e9%85%8d%e7%bd%ae-prometheus-%e6%9d%a5%e6%8a%93%e5%8f%96%e5%a4%96%e9%83%a8-exporter-%e6%95%b0%e6%8d%ae)

在 Prometheus 配置中，你需要添加外部 **Exporter** 的目标，通常通过 **`scrape_config`** 配置 Prometheus 从外部目标（IP 地址或域名）拉取数据。

#### 步骤： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%ad%a5%e9%aa%a4-7)

1. **编辑 Prometheus 配置文件**：在 Prometheus 配置文件（通常是 `prometheus.yml`）中，添加一个新的 `scrape_config` 配置，以便 Prometheus 可以从外部的 **Exporter** 中抓取数据。

   **示例：Prometheus 配置**



   ```yaml
   global:
     scrape_interval: 15s

   scrape_configs:
  - job_name: 'external-exporter'
    static_configs:
      - targets: ['<external_exporter_ip>:<port>', '<external_exporter_ip2>:<port>']
```

   - `job_name`：定义一个任务的名称，您可以根据需要修改这个名称。
   - `targets`：是 **Exporter** 的地址列表，通常包括 IP 和端口，确保 Prometheus 可以访问这些地址。你可以列出多个目标，Prometheus 将会并行拉取数据。
2. **重启 Prometheus**：配置修改完成后，重启 Prometheus 服务以使新的配置生效。



```bash
systemctl restart prometheus
```



或者，如果你使用的是 Docker：



```bash
docker restart <prometheus_container_name>
```


### 3\. 验证 Prometheus 是否可以正确抓取外部 Exporter 数据 [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#3-%e9%aa%8c%e8%af%81-prometheus-%e6%98%af%e5%90%a6%e5%8f%af%e4%bb%a5%e6%ad%a3%e7%a1%ae%e6%8a%93%e5%8f%96%e5%a4%96%e9%83%a8-exporter-%e6%95%b0%e6%8d%ae)

完成配置后，你可以通过 Prometheus 的 **Web UI** 来查看是否已经成功抓取到外部 **Exporter** 的数据。

1. 进入 Prometheus Web UI（通常是 `http://<prometheus_ip>:9090`）。

2. 进入

Targets

页面 (


```
http://<prometheus_ip>:9090/targets
```


)，检查


```
scrape_configs
```


中添加的目标是否出现，状态是否为


```
up
```


。

   - 如果目标处于 `up` 状态，表示 Prometheus 成功抓取了指标。
   - 如果状态为 `down`，检查是否能从 Prometheus 机器访问外部 **Exporter** 地址，可能是网络问题、端口没有开放或者防火墙限制等原因。

### 4\. 配置告警（可选） [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#4-%e9%85%8d%e7%bd%ae%e5%91%8a%e8%ad%a6%e5%8f%af%e9%80%89)

你可以根据抓取到的外部指标配置 Prometheus 的告警规则。告警规则可以通过 **PrometheusRule** 或直接在 Prometheus 配置文件中进行设置。

例如，假设你正在抓取一个 Node Exporter 的外部指标并想设置一个告警来监控 CPU 使用率：

#### 配置告警规则： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e9%85%8d%e7%bd%ae%e5%91%8a%e8%ad%a6%e8%a7%84%e5%88%99)

```yaml
groups:
  - name: node-alerts
    rules:
      - alert: HighCPUUsage
        expr: node_cpu_seconds_total{mode="user"} > 80
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "CPU usage is over 80% for 5 minutes."
```

### 5\. 配置告警通知（可选） [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#5-%e9%85%8d%e7%bd%ae%e5%91%8a%e8%ad%a6%e9%80%9a%e7%9f%a5%e5%8f%af%e9%80%89)

如果需要配置告警通知（例如，发送到 Slack、Email 或其他通知渠道），你需要在 Prometheus 配置文件中设置 **Alertmanager** 配置。

**示例：Alertmanager 配置**（发送 Slack 通知）

```yaml
global:
  resolve_timeout: 5m

route:
  receiver: 'slack-notifications'

receivers:
  - name: 'slack-notifications'
    slack_configs:
      - channel: '#alerts'
        send_resolved: true
```

然后，Prometheus 会将告警发送到 **Alertmanager**，由 **Alertmanager** 负责处理和转发通知。

### 总结： [\#](http://devopsz.top/docs/2025-2-28-prometheus%E9%A2%98%E7%9B%AE/\#%e6%80%bb%e7%bb%93-12)

1. 确保外部 **Exporter** 可通过 HTTP 暴露指标，Prometheus 可以访问。
2. 在 Prometheus 配置文件中使用 `scrape_configs` 添加外部 **Exporter** 的地址。
3. 通过 Prometheus Web UI 检查目标是否被正确抓取。
4. 配置告警规则和通知渠道（可选）。

通过这些步骤，Prometheus 就可以监控外部 **Exporter**，并且你可以根据需要设置告警和通知。