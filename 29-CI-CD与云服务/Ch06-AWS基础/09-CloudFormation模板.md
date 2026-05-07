# CloudFormation模板

## 一、概念说明

CloudFormation是AWS的基础设施即代码（IaC）服务，使用模板文件定义和部署AWS资源，实现基础设施的版本控制和自动化管理。

| 概念 | 说明 |
|------|------|
| Template | JSON/YAML资源定义文件 |
| Stack | 模板部署的资源集合 |
| Change Set | 变更预览 |
| StackSet | 跨账号/区域部署 |
| Nested Stack | 嵌套堆栈复用 |

## 二、具体用法

### 基础模板结构

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Web服务器基础设施模板

Parameters:
  Environment:
    Type: String
    Default: dev
    AllowedValues:
      - dev
      - staging
      - prod

  InstanceType:
    Type: String
    Default: t2.micro

Resources:
  # VPC
  MyVPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-VPC"

  # 子网
  PublicSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref MyVPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-PublicSubnet"

  # EC2实例
  WebServer:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: !Ref InstanceType
      ImageId: ami-0c55b159cbfafe1f0
      SubnetId: !Ref PublicSubnet
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-WebServer"

Outputs:
  ServerURL:
    Description: Web服务器地址
    Value: !Sub "http://${WebServer.PublicDnsName}"
    Export:
      Name: !Sub "${Environment}-WebServerURL"
```

### 操作堆栈

```bash
# 创建堆栈
aws cloudformation create-stack \
    --stack-name my-stack \
    --template-body file://template.yaml \
    --parameters ParameterKey=Environment,ParameterValue=dev \
    --capabilities CAPABILITY_IAM

# 更新堆栈
aws cloudformation update-stack \
    --stack-name my-stack \
    --template-body file://template.yaml \
    --parameters ParameterKey=Environment,ParameterValue=staging

# 查看堆栈事件
aws cloudformation describe-stack-events \
    --stack-name my-stack \
    --query 'StackEvents[*].[Timestamp,ResourceType,ResourceStatus]'

# 删除堆栈
aws cloudformation delete-stack --stack-name my-stack
```

### 变更集

```bash
# 创建变更集
aws cloudformation create-change-set \
    --stack-name my-stack \
    --change-set-name my-changes \
    --template-body file://updated-template.yaml

# 查看变更
aws cloudformation describe-change-set \
    --change-set-name my-changes \
    --stack-name my-stack

# 执行变更
aws cloudformation execute-change-set \
    --change-set-name my-changes \
    --stack-name my-stack
```

### 内置函数

```yaml
# 常用内置函数
Resources:
  MyResource:
    Properties:
      # 引用参数
      Name: !Ref Environment
      # 字符串拼接
      Description: !Sub "Server in ${Environment}"
      # 获取属性
      VpcId: !GetAtt MyVPC.VpcId
      # 条件赋值
      Size: !If [IsProd, "large", "small"]
      # 选择列表元素
      AZ: !Select [0, !GetAZs '']
```

## 三、注意事项与常见陷阱

1. **模板验证**：部署前使用validate-template检查语法
2. **回滚机制**：创建失败默认回滚，可选择禁用以调试
3. **删除保护**：生产堆栈启用termination protection
4. **漂移检测**：定期检测实际资源与模板的差异
5. **嵌套堆栈**：复杂架构使用嵌套堆栈模块化
6. **输出导出**：使用Export/Import跨堆栈引用资源
7. **成本估算**：使用CloudFormation估算堆栈成本
