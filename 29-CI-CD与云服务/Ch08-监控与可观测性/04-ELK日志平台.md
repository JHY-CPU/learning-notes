# ELK日志平台

## 一、概念说明

ELK是Elasticsearch、Logstash、Kibana三个开源工具的组合，用于日志的收集、存储、搜索和可视化。

| 组件 | 功能 |
|------|------|
| Elasticsearch | 分布式搜索和分析引擎 |
| Logstash | 日志收集和处理管道 |
| Kibana | 数据可视化界面 |
| Beats | 轻量级数据采集器 |
| Filebeat | 日志文件采集 |
| Metricbeat | 系统指标采集 |

## 二、具体用法

### Docker Compose部署

```yaml
# docker-compose.yml
version: '3.8'
services:
  elasticsearch:
    image: elasticsearch:8.12.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    ports:
      - 9200:9200
    volumes:
      - es-data:/usr/share/elasticsearch/data

  logstash:
    image: logstash:8.12.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - 5044:5044
    depends_on:
      - elasticsearch

  kibana:
    image: kibana:8.12.0
    ports:
      - 5601:5601
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  es-data:
```

### Logstash配置

```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  # 解析JSON日志
  if [fields][log_type] == "app" {
    json {
      source => "message"
      target => "parsed"
    }

    # 提取时间戳
    date {
      match => ["parsed.timestamp", "ISO8601"]
      target => "@timestamp"
    }

    # 根据日志级别添加标签
    if [parsed.level] == "ERROR" {
      mutate { add_tag => ["error"] }
    }
  }

  # 解析Nginx访问日志
  if [fields][log_type] == "nginx" {
    grok {
      match => { "message" => "%{COMBINEDAPACHELOG}" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["http://elasticsearch:9200"]
    index => "app-logs-%{+YYYY.MM.dd}"
  }

  if "error" in [tags] {
    elasticsearch {
      hosts => ["http://elasticsearch:9200"]
      index => "error-logs-%{+YYYY.MM.dd}"
    }
  }
}
```

### Filebeat配置

```yaml
# filebeat.yml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/app/*.log
    fields:
      log_type: app
      environment: production
    multiline.pattern: '^\d{4}-\d{2}-\d{2}'
    multiline.negate: true
    multiline.match: after

  - type: log
    enabled: true
    paths:
      - /var/log/nginx/access.log
    fields:
      log_type: nginx

output.logstash:
  hosts: ["logstash:5044"]
```

### Python应用日志

```python
import logging
import json
from pythonjsonlogger import jsonlogger

# 配置JSON日志
logger = logging.getLogger('my-app')
logger.setLevel(logging.INFO)

handler = logging.FileHandler('/var/log/app/app.log')
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(levelname)s %(name)s %(message)s',
    rename_fields={'asctime': 'timestamp', 'levelname': 'level'}
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# 结构化日志
logger.info('用户登录成功', extra={
    'user_id': '12345',
    'ip_address': '192.168.1.100',
    'action': 'login'
})

logger.error('数据库连接失败', extra={
    'error': str(e),
    'database': 'mydb',
    'retry_count': 3
})
```

### Kibana查询语法

```bash
# 全文搜索
error AND database

# 字段搜索
level:ERROR
level:ERROR AND service:web-app

# 通配符
message:timeout*

# 范围查询
response_time:>500

# 正则表达例
message:/user.*failed/
```

## 三、注意事项与常见陷阱

1. **Elasticsearch资源**：分配足够的内存，禁用swap
2. **索引管理**：设置索引生命周期管理（ILM）自动清理
3. **分片规划**：合理设置分片数量，过多影响性能
4. **Logstash性能**：合理配置pipeline workers和batch size
5. **数据量控制**：日志分级采样，避免存储爆炸
6. **安全配置**：生产环境启用X-Pack安全功能
7. **备份策略**：定期快照Elasticsearch数据
