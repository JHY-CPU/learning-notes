# Ansible与配置管理 - 云服务与DevOps笔记


Ansible是Red Hat开发的开源自动化工具，用于配置管理、应用部署、任务编排和IT自动化。它采用无代理架构，通过SSH连接管理远程主机，使用YAML格式编写配置。


### 1.1 核心特点


- **无代理（Agentless）**
   ：无需在目标机器安装客户端，仅依赖SSH和Python
- **幂等性（Idempotency）**
   ：重复执行不会改变已达到期望状态的系统
- **声明式 + 命令式**
   ：支持声明式描述目标状态，也支持执行具体命令
- **简单易学**
   ：YAML语法清晰，学习曲线低
- **丰富的模块库**
   ：内置数千个模块，覆盖系统管理各方面


### 1.2 架构组件


```
┌────────── 控制节点（Control Node）──────────┐
  │                                              │
  │  [Ansible命令/Playbook]                      │
  │        │                                     │
  │  [Inventory主机清单]  [Modules模块库]         │
  │        │                                     │
  │  [Playbook剧本]  [Role角色]  [Plugin插件]    │
  │        │                                     │
  └────────┼─────────────────────────────────────┘
           │ SSH（无需安装Agent）
    ┌──────┼──────┐
    ↓      ↓      ↓
  目标机  目标机  目标机
  (Linux) (Linux) (Windows/WinRM)
```


### 1.3 安装


```
# 安装 Ansible
pip install ansible

# 或使用包管理器
# Ubuntu/Debian
sudo apt install ansible

# CentOS/RHEL
sudo yum install ansible

# 验证
ansible --version
```


Inventory定义了Ansible管理的主机和主机组。


### 2.1 INI格式


```
# /etc/ansible/hosts 或自定义 inventory 文件
# 单个主机
web1 ansible_host=192.168.1.10 ansible_user=root
web2 ansible_host=192.168.1.11 ansible_user=root

# 主机组
[webservers]
web1
web2

[dbservers]
db1 ansible_host=192.168.1.20
db2 ansible_host=192.168.1.21

# 组的组
[datacenter:children]
webservers
dbservers

# 组变量
[webservers:vars]
http_port=80
max_clients=200

# 全局变量
[all:vars]
ansible_python_interpreter=/usr/bin/python3
```


### 2.2 YAML格式


```
# inventory.yml
all:
  children:
    webservers:
      hosts:
        web1:
          ansible_host: 192.168.1.10
        web2:
          ansible_host: 192.168.1.11
      vars:
        http_port: 80
    dbservers:
      hosts:
        db1:
          ansible_host: 192.168.1.20
  vars:
    ansible_python_interpreter: /usr/bin/python3
```


### 2.3 动态Inventory


从云平台API动态获取主机列表，避免手动维护。


```
# 使用AWS动态Inventory
ansible -i aws_ec2.yml all -m ping

# aws_ec2.yml 配置示例
plugin: amazon.aws.aws_ec2
regions:
  - us-east-1
filters:
  tag:Environment: production
  instance-state-name: running
keyed_groups:
  - key: tags.Role
    prefix: role
  - key: placement.availability_zone
    prefix: az
```


Ad-hoc命令用于执行简单的单次任务，无需编写Playbook。


### 3.1 基本语法


```
# 语法：ansible <主机/组> -m <模块> -a <参数>
# 测试连接
ansible webservers -m ping

# 执行shell命令
ansible webservers -m shell -a "uptime"

# 查看系统信息
ansible webservers -m setup -a "filter=ansible_os_family"

# 复制文件
ansible webservers -m copy -a "src=/local/file dest=/remote/file mode=0644"

# 安装软件包
ansible webservers -m yum -a "name=httpd state=present"

# 管理服务
ansible webservers -m service -a "name=httpd state=started enabled=yes"

# 并行执行（10个并发）
ansible all -m ping -f 10

# 使用sudo
ansible webservers -m yum -a "name=nginx" --become --become-user=root
```


### 3.2 常用模块速查


| 模块 | 用途 | 示例 |
| --- | --- | --- |
| ping | 测试连接 | `-m ping` |
| shell | 执行Shell命令 | `-m shell -a "ls"` |
| command | 执行命令（非Shell） | `-m command -a "ls"` |
| copy | 复制文件 | `-m copy -a "src=... dest=..."` |
| file | 管理文件/目录 | `-m file -a "path=... state=directory"` |
| yum/apt | 包管理 | `-m yum -a "name=nginx state=latest"` |
| service | 管理服务 | `-m service -a "name=nginx state=started"` |
| template | 模板渲染 | `-m template -a "src=t.j2 dest=..."` |
| user | 用户管理 | `-m user -a "name=test state=present"` |
| git | Git操作 | `-m git -a "repo=... dest=..."` |


Playbook是Ansible的核心，用YAML格式编写的自动化任务定义文件，支持多任务编排、条件判断、循环等。


### 4.1 基本结构


```
# site.yml - 一个完整的Playbook
---
- name: 配置Web服务器
  hosts: webservers
  become: yes
  vars:
    http_port: 80
    max_clients: 200

  tasks:
    - name: 安装Nginx
      yum:
        name: nginx
        state: present

    - name: 复制Nginx配置
      template:
        src: templates/nginx.conf.j2
        dest: /etc/nginx/nginx.conf
      notify: restart nginx

    - name: 确保Nginx运行
      service:
        name: nginx
        state: started
        enabled: yes

    - name: 开放防火墙端口
      firewalld:
        port: "{{ http_port }}/tcp"
        permanent: yes
        state: enabled
        immediate: yes

  handlers:
    - name: restart nginx
      service:
        name: nginx
        state: restarted

    - name: reload nginx
      service:
        name: nginx
        state: reloaded
```


### 4.2 条件与循环


```
# 条件执行 (when)
tasks:
  - name: 仅在CentOS上安装
    yum:
      name: httpd
      state: present
    when: ansible_os_family == "RedHat"

  - name: 仅在Ubuntu上安装
    apt:
      name: apache2
      state: present
    when: ansible_os_family == "Debian"
# 循环 (loop)
tasks:
  - name: 创建多个用户
    user:
      name: "{{ item.name }}"
      groups: "{{ item.groups }}"
      state: present
    loop:
      - { name: "alice", groups: "sudo" }
      - { name: "bob", groups: "developers" }
      - { name: "carol", groups: "developers" }

# 条件+循环
tasks:
  - name: 安装多个包
    package:
      name: "{{ item }}"
      state: present
    loop:
      - git
      - vim
      - curl
      - wget
    when: ansible_os_family == "RedHat"
# register 获取命令输出
tasks:
  - name: 检查服务状态
    command: systemctl is-active nginx
    register: nginx_status
    ignore_errors: yes

  - name: 如果服务未运行则启动
    service:
      name: nginx
      state: started
    when: nginx_status.stdout != "active"
```


### 4.3 执行Playbook


```
# 运行Playbook
ansible-playbook site.yml

# 指定Inventory
ansible-playbook -i inventory.yml site.yml

# 仅运行特定Tag的任务
ansible-playbook site.yml --tags "install,config"

# 跳过某些Tag
ansible-playbook site.yml --skip-tags "deploy"

# 检查模式（dry-run）
ansible-playbook site.yml --check --diff

# 限制运行的主机
ansible-playbook site.yml --limit webservers

# 传递额外变量
ansible-playbook site.yml -e "http_port=8080"
```


### 5.1 Role目录结构


Role将Playbook中的任务、变量、模板、处理器等组织成标准化的目录结构，方便复用和共享。


```
# Role标准目录结构
roles/
└── nginx/
    ├── tasks/
    │   └── main.yml       # 任务定义
    ├── handlers/
    │   └── main.yml       # 处理器
    ├── templates/
    │   └── nginx.conf.j2  # Jinja2模板
    ├── files/
    │   └── index.html     # 静态文件
    ├── vars/
    │   └── main.yml       # 变量（优先级高）
    ├── defaults/
    │   └── main.yml       # 默认变量（优先级低）
    ├── meta/
    │   └── main.yml       # Role元数据和依赖
    └── README.md
```


### 5.2 Role定义示例


```
# roles/nginx/tasks/main.yml
---
- name: 安装Nginx
  yum:
    name: nginx
    state: present

- name: 复制配置
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
  notify: restart nginx

- name: 启动Nginx
  service:
    name: nginx
    state: started
    enabled: yes

# roles/nginx/handlers/main.yml
---
- name: restart nginx
  service:
    name: nginx
    state: restarted

# roles/nginx/defaults/main.yml
---
http_port: 80
worker_processes: auto
max_clients: 1024

# roles/nginx/templates/nginx.conf.j2
worker_processes {{ worker_processes }};
events {
    worker_connections {{ max_clients }};
}
http {
    server {
        listen {{ http_port }};
    }
}
```


### 5.3 在Playbook中使用Role


```
# 方式1：简单引用
---
- hosts: webservers
  roles:
    - nginx
    - mysql

# 方式2：带参数传递
---
- hosts: webservers
  roles:
    - role: nginx
      vars:
        http_port: 8080
        worker_processes: 4
    - role: mysql
      vars:
        mysql_port: 3306

# 方式3：使用include_role（动态）
---
- hosts: webservers
  tasks:
    - name: 动态加载Role
      include_role:
        name: nginx
      vars:
        http_port: 80
```


| 维度 | Ansible | Chef | Puppet | SaltStack |
| --- | --- | --- | --- | --- |
| 架构 | 无代理（SSH） | C/S（Agent） | C/S（Agent） | C/S（Agent） |
| 配置语言 | YAML | Ruby DSL | Puppet DSL | YAML + Jinja2 |
| 学习曲线 | 低 | 中高 | 中 | 中 |
| 执行模式 | Push | Pull | Pull | Push + Pull |
| 幂等性 | 内置 | 需要编写 | 内置 | 内置 |
| 社区/模块 | 非常丰富 | 丰富 | 丰富 | 较丰富 |
| 适用规模 | 中小-中大 | 大型 | 大型 | 大型/高性能 |
| GUI | AWX/Tower | Chef Server | Puppet Enterprise | Salt GUI |


### 6.1 工具选型建议


- **Ansible**
   ：最简单易上手，适合中小型团队快速实现自动化，无Agent架构对已有系统侵入最小
- **Puppet**
   ：适合大规模基础设施的持续合规管理，声明式模型强大
- **Chef**
   ：适合Ruby技术栈团队，灵活性最高但学习成本也高
- **SaltStack**
   ：适合需要高性能远程执行和大规模并行管理的场景


**当前趋势：**Ansible因其简单性和无代理特性，成为DevOps领域使用最广泛的配置管理工具。云原生时代，Ansible常与Terraform配合使用：Terraform负责基础设施编排，Ansible负责配置管理和应用部署。


### 6.2 IaC工具 vs 配置管理工具


| 维度 | Terraform（IaC） | Ansible（配置管理） |
| --- | --- | --- |
| 核心能力 | 创建/管理云基础设施 | 配置/管理操作系统和应用 |
| 状态管理 | 有状态文件 | 无状态（每次重新检查） |
| 适用范围 | 云资源（VM/网络/存储/数据库） | 系统配置/软件安装/应用部署 |
| 最佳搭配 | 基础设施层 | 配置和应用层 |


> **Example:** **典型组合用法：**
>
> 1. Terraform 创建VPC、子网、ECS实例、RDS数据库
> 2. Ansible 接管新创建的ECS，安装配置Nginx、部署应用代码
> 3. CI/CD流水线编排两者的执行顺序


<!-- Converted from: 02_Ansible与配置管理.html -->
