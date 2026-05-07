# OSS对象存储

## 一、概念说明

OSS（Object Storage Service）是阿里云的对象存储服务，提供海量、安全、低成本的云存储。适用于图片视频、静态网站、备份归档等场景。

| 存储类型 | 访问频率 | 最低存储时长 | 适用场景 |
|----------|----------|------------|----------|
| 标准 | 高频 | 无 | 热数据 |
| 低频 | 低频 | 30天 | 不常访问 |
| 归档 | 极低 | 60天 | 冷数据 |
| 冷归档 | 极低 | 180天 | 长期归档 |

## 二、具体用法

### 基本操作

```bash
# 安装ossutil
curl -o /usr/local/bin/ossutil https://gosspublic.alicdn.com/ossutil/v2/2.0.3-beta.09161422/ossutil-v2.0.3-beta.09161422-linux-amd64.zip
chmod 755 /usr/local/bin/ossutil

# 配置
ossutil config --access-key-id YOUR_AK_ID --access-key-secret YOUR_AK_SECRET --endpoint oss-cn-hangzhou.aliyuncs.com

# 创建存储桶
ossutil mb oss://my-bucket --storage-class Standard

# 上传文件
ossutil cp myfile.txt oss://my-bucket/
ossutil cp ./mydir oss://my-bucket/mydir/ -r

# 下载文件
ossutil cp oss://my-bucket/myfile.txt ./

# 列出文件
ossutil ls oss://my-bucket/
ossutil ls oss://my-bucket/ -r --summarize
```

### 静态网站托管

```bash
# 配置静态网站
ossutil website oss://my-bucket --index-page index.html --error-page 404.html

# 设置存储桶ACL为公共读
ossutil set-acl oss://my-bucket -b public-read

# 设置CNAME自定义域名
ossutil set-meta oss://my-bucket --headers "host:static.example.com"
```

### 生命周期规则

```bash
# 设置生命周期规则
ossutil lifecycle --method put oss://my-bucket lifecycle.json
```

```json
{
    "Rule": [{
        "ID": "archive-rule",
        "Prefix": "logs/",
        "Status": "Enabled",
        "Transition": [{
            "Days": 30,
            "StorageClass": "IA"
        }, {
            "Days": 90,
            "StorageClass": "Archive"
        }],
        "Expiration": {
            "Days": 365
        }
    }]
}
```

### 图片处理

```bash
# 上传并处理图片
ossutil cp photo.jpg oss://my-bucket/photo.jpg?x-oss-process=image/resize,w_500/format,webp

# 图片样式URL
# https://my-bucket.oss-cn-hangzhou.aliyuncs.com/photo.jpg?x-oss-process=style/thumbnail
```

### Python SDK

```python
import oss2

# 创建存储桶对象
auth = oss2.Auth('YOUR_ACCESS_KEY_ID', 'YOUR_ACCESS_KEY_SECRET')
bucket = oss2.Bucket(auth, 'https://oss-cn-hangzhou.aliyuncs.com', 'my-bucket')

# 上传文件
bucket.put_object('hello.txt', 'Hello, OSS!')

# 流式上传
with open('large_file.bin', 'rb') as f:
    bucket.put_object('large_file.bin', f)

# 断点续传
oss2.resumable_upload(bucket, 'big_video.mp4', 'big_video.mp4')

# 分片上传
from oss2 import determine_part_size
total_size = os.path.getsize('huge_file.zip')
part_size = determine_part_size(total_size)
```

## 三、注意事项与常见陷阱

1. **Bucket名称**：全局唯一，仅小写字母、数字和连字符
2. **权限设置**：默认私有，公共读需显式设置
3. **跨区域复制**：重要数据启用跨区域复制
4. **防盗链设置**：配置Referer白名单防止盗链
5. **传输加速**：远距离上传使用传输加速Endpoint
6. **分片上传**：大于100MB的文件使用分片上传
7. **数据备份**：开启版本控制防止误删除
