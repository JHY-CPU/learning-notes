# 基础设施即代码IaC 初探- Ansible 与Terraform

URL: https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/

\[简体中文\]/ [English](https://blog.lyc8503.net/en/post/iac-explore-ansible-and-terraform/)

[Menu](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#) [Menu](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#) [Top](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#)

- [Previous post](https://blog.lyc8503.net/post/hypervisor-explore/)
- [Next post](https://blog.lyc8503.net/post/ipv6-differences/)
- [Back to top](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#)
- [Share post](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#)

Previous postNext postBack to topShare post

1. [1.背景](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#%E8%83%8C%E6%99%AF)
2. [2.Ansible - 无需 Agent 远程设置 Linux 服务器](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#Ansible-%E6%97%A0%E9%9C%80-Agent-%E8%BF%9C%E7%A8%8B%E8%AE%BE%E7%BD%AE-Linux-%E6%9C%8D%E5%8A%A1%E5%99%A8)
3. [3.Terraform - 使用代码定义云上基础架构](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#Terraform-%E4%BD%BF%E7%94%A8%E4%BB%A3%E7%A0%81%E5%AE%9A%E4%B9%89%E4%BA%91%E4%B8%8A%E5%9F%BA%E7%A1%80%E6%9E%B6%E6%9E%84)
4. [4.小结](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#%E5%B0%8F%E7%BB%93)

很久以前就听说过了 `基础设施即代码 (Infrastructure-as-Code, 缩写 IaC)` 的做法, 最近正好有不错的机会实践了一下, 写此文以记录.

## [背景](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/\#%E8%83%8C%E6%99%AF "背景") 背景

我现有的基础设施主要是我的 [HomeLab](https://blog.lyc8503.net/categories/%E4%B8%93%E9%A2%98%E5%90%91%E7%A0%94%E7%A9%B6/AllInOne%E5%AE%B6%E7%94%A8%E6%9C%8D%E5%8A%A1%E5%99%A8/) 和阿里云的一些 [Serverless 服务](https://blog.lyc8503.net/post/cloud-native/), 两者之前都有相关的博客提到过, 现在都处于稳定运行状态, 相对不需要大的变动.

不过我还有些必要的辅助服务需要在 **海外 VPS** 上运行, 主要出于 **网络连接性** 和成本的考量, VPS 提供商可能会时不时更换. 每次给服务器搬家相对就比较麻烦了.

简单粗暴一点的做法是编写一个 Docker Compose 文件或者 bash 脚本, 每次拿到新 VPS 安装一下 docker 就算搞定了. 不过 Ansible 似乎是一种更加优雅的解决方案, 故来尝试一下.

## [Ansible - 无需 Agent 远程设置 Linux 服务器](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/\#Ansible-%E6%97%A0%E9%9C%80-Agent-%E8%BF%9C%E7%A8%8B%E8%AE%BE%E7%BD%AE-Linux-%E6%9C%8D%E5%8A%A1%E5%99%A8 "Ansible - 无需 Agent 远程设置 Linux 服务器") Ansible - 无需 Agent 远程设置 Linux 服务器

在一台新的 Linux VPS 上配环境和安装需要的软件不算什么难事, 但如果要多次配置/迁移或者一次配置多台机器就是件麻烦事了.

[Ansible 的官方文档](https://docs.ansible.com/ansible/latest/getting_started/index.html) 可以说是比较清晰易懂的, 阅读一遍就能基本掌握 Ansible 的常见用法.

如果使用了 Ansible 配置流程大概是这样:

1. 在本地机器安装 Ansible
2. 在本地机器配置 **inventory**(VPS 连接信息), 编写 **playbook**(VPS 的目标状态)
3. 本地执行 playbook, Ansible 直接通过 SSH 协议连接并操控 VPS, 使得 VPS 达到 playbook 中定义的状态.

大概感受一下 playbook 的样子:

|     |     |
| --- | --- |
| ```<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>``` | ```<br>---<br>- name: Install nginx package<br>  package:<br>    name:<br>      - nginx<br>    state: present<br>- name: Update nginx config<br>  ansible.builtin.template:<br>    src: templates/nginx.conf.j2<br>    dest: /etc/nginx/sites-available/default<br>  notify:<br>    - Restart nginx<br>- name: Start nginx<br>  ansible.builtin.systemd_service:<br>    name: nginx<br>    state: started<br>    daemon_reload: true<br>``` |

上面是一个配置 Nginx 服务的 task 定义: **第一步** 确保 Nginx 包已安装, 第二步更新 Nginx 站点配置文件(如果配置更新还需要重启 Nginx), **第三步** 确保 Nginx 已经启动, 就是这么简单~

**相比 Docker Compose**, Ansible 更加灵活和方便, 可以直接操作主机的各项配置, 并且也可以使用 Ansible 安装并启动 Docker, 两者并不冲突.

**相比直接使用 Bash 脚本**, Ansible 使用更加优雅的 yaml 并且是”幂等”的, 多次执行不会造成冲突, 修改起来远比 Bash 脚本方便.

同时 Ansbile 提供了名为 galaxy 的一个 registry, 可以 **直接使用他人写好的模块 (Role)**, 比如在服务器上安装 Docker 和 Docker Compose 只需要这么写:

|     |     |
| --- | --- |
| ```<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>``` | ```<br>---<br>- include_role:<br>    name: geerlingguy.docker<br>- include_role:<br>    name: geerlingguy.pip<br>  vars:<br>    pip_install_packages:<br>      - name: docker<br>      - name: docker-compose<br>``` |

很多常见的功能都能以模块的形式导入, 不需要再手动编写和维护, 并具有一定的跨发行版运行的能力.

如果我服务器发生迁移, 现在我只需要修改 inventory 文件, 重新执行当前的 playbook, VPS 就会部署好全部的服务, 大幅提高了效率并降低了犯错的可能性, 也给我更强的信心和灵活度去调整我的基础设施.

目前我 VPS 的 playbook 已经 [开源](https://github.com/lyc8503/infra). IaC 也可以直接利用现有的代码版本控制工具 (如 git) 来进行管理, 方便多人协同和历史追溯.

## [Terraform - 使用代码定义云上基础架构](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/\#Terraform-%E4%BD%BF%E7%94%A8%E4%BB%A3%E7%A0%81%E5%AE%9A%E4%B9%89%E4%BA%91%E4%B8%8A%E5%9F%BA%E7%A1%80%E6%9E%B6%E6%9E%84 "Terraform - 使用代码定义云上基础架构") Terraform - 使用代码定义云上基础架构

Ansible 主要解决了 Linux VPS 部署的问题, 但云上的很多基础架构也不是以 Linux 虚拟机的形式提供的.

比如最近我在 GitHub 上开源的 [UptimeFlare](https://github.com/lyc8503/UptimeFlare) 就使用了 **Cloudflare Workers / KV / Pages 这几个服务**, 写部署文档的时候发现… 部署真的好麻烦, 得登录 Cloudflare 的 Dashboard 点一大堆按钮, ~~完全不想写这种又臭又长的文档~~. 本来想写个 bash 脚本进行一键部署, 发现 wrangler (Cloudflare 的部署工具) 也缺一堆参数和文档, 还是得配合手动操作, 还是非常麻烦.

于是懒惰又一次成为了生产力的来源, 正好刚研究完 Ansible, 就发现了 Terraform 这个工具.

Terraform 同样是通过代码定义云上基础设施的状态, Terraform 会自己保证云上的状态与用户的定义一致.

Cloudflare 官方也提供了 [Terraform 的文档](https://developers.cloudflare.com/terraform/), 每个云厂商有自己的 provider 和相关文档, Terraform 本身的语言也十分简单, 按照文档依葫芦画瓢就能完成配置.

同样是 [上一段代码](https://github.com/lyc8503/UptimeFlare/blob/main/deploy.tf) 感受一下:

|     |     |
| --- | --- |
| ```<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36<br>37<br>38<br>39<br>40<br>41<br>42<br>43<br>44<br>45<br>46<br>47<br>48<br>49<br>50<br>51<br>52<br>53<br>54<br>55<br>56<br>57<br>58<br>59<br>``` | ```<br>terraform {<br>  required_providers {<br>    cloudflare = {<br>      source  = "cloudflare/cloudflare"<br>      version = "~> 4"<br>    }<br>  }<br>}<br>provider "cloudflare" {<br>  # read token from $CLOUDFLARE_API_TOKEN<br>}<br>variable "CLOUDFLARE_ACCOUNT_ID" {<br>  # read account id from $TF_VAR_CLOUDFLARE_ACCOUNT_ID<br>  type = string<br>}<br>resource "cloudflare_workers_kv_namespace" "uptimeflare_kv" {<br>  account_id = var.CLOUDFLARE_ACCOUNT_ID<br>  title      = "uptimeflare_kv"<br>}<br>resource "cloudflare_worker_script" "uptimeflare" {<br>  account_id         = var.CLOUDFLARE_ACCOUNT_ID<br>  name               = "uptimeflare_worker"<br>  content            = file("worker/dist/index.js")<br>  module             = true<br>  compatibility_date = "2023-11-08"<br>  kv_namespace_binding {<br>    name         = "UPTIMEFLARE_STATE"<br>    namespace_id = cloudflare_workers_kv_namespace.uptimeflare_kv.id<br>  }<br>}<br>resource "cloudflare_worker_cron_trigger" "uptimeflare_worker_cron" {<br>  account_id  = var.CLOUDFLARE_ACCOUNT_ID<br>  script_name = cloudflare_worker_script.uptimeflare.name<br>  schedules = [<br>    "*/2 * * * *", # every 2 minutes<br>  ]<br>}<br>resource "cloudflare_pages_project" "uptimeflare" {<br>  account_id        = var.CLOUDFLARE_ACCOUNT_ID<br>  name              = "uptimeflare"<br>  production_branch = "main"<br>  deployment_configs {<br>    production {<br>      kv_namespaces = {<br>        UPTIMEFLARE_STATE = cloudflare_workers_kv_namespace.uptimeflare_kv.id<br>      }<br>      compatibility_date  = "2023-11-08"<br>      compatibility_flags = ["nodejs_compat"]<br>    }<br>  }<br>}<br>``` |

基本上就是不同的块定义不同的资源, 传入文档中指定的参数就行了.

随后直接在本地执行 `terraform init` 和 `terraform plan` 就能规划出需要的更改.

执行 `terraform apply` 就能将列出的更改应用到云, 随后基础设施有修改, 只要修改代码再次 `terraform apply` 就能应用新的更改.

在我这个使用场景, 我就不用写很长很繁琐的文档教用户怎么部署了, 直接让用户提供 API Token 即可自动部署~

## [小结](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/\#%E5%B0%8F%E7%BB%93 "小结") 小结

IaC 的一些初体验就是这样了, ~~果然懒才是第一生产力~~, Ansible 和 Terraform 入门门槛很低, 配置简单, 个人开发者使用也不会像某些所谓的”企业级”方案一样繁琐 ~~(aka. 脱裤子放屁)~~, 优势是能提供更高的一致性和可拓展性, 同时可以省去很多繁杂的手动操作步骤提高效率.

本文采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) 许可协议发布.

作者: lyc8503, 文章链接: [https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/)

如果本文给你带来了帮助或让你觉得有趣, 可以考虑 [赞助我](https://blog.lyc8503.net/about/#Sponsor) ¬\_¬

你认为这篇文章怎么样？

- ![](https://blog.lyc8503.net/images/reactions/1f44d.png)

0

- ![](https://blog.lyc8503.net/images/reactions/1f44e.png)

0

- ![](https://blog.lyc8503.net/images/reactions/1f604.png)

0

- ![](https://blog.lyc8503.net/images/reactions/1f389.png)

0

- ![](https://blog.lyc8503.net/images/reactions/1f615.png)

0

- ![](https://blog.lyc8503.net/images/reactions/2764.png)

0

- ![](https://blog.lyc8503.net/images/reactions/1f680.png)

0

- ![](https://blog.lyc8503.net/images/reactions/1f440.png)

0


昵称

邮箱

网址(可选)

* * *

#### 预览:

[Markdown Guide](https://guides.github.com/features/mastering-markdown/ "Markdown Guide")

0   字

提交

1 评论

- 按正序
- 按倒序
- 按热度

![](https://seccdn.libravatar.org/avatar/d41d8cd98f00b204e9800998ecf8427e)

[luckyops](https://github.com/luckyops) 2024-10-05

0

好耶，学到了Terraform，之前一直用ansible，Terraform 还没搞过，嘿嘿。

[订阅本文评论](https://waline.lyc8503.net/api/comment/rss?path=post%2Fiac-explore-ansible-and-terraform%2F) [订阅本站评论](https://waline.lyc8503.net/api/comment/rss)

- [Home](https://blog.lyc8503.net/index.html)
- [About](https://blog.lyc8503.net/about/)
- [Writing](https://blog.lyc8503.net/archives/)
- [Category](https://blog.lyc8503.net/categories/)
- [Friends](https://blog.lyc8503.net/friends/)
- [Search](https://blog.lyc8503.net/search/)

1. [1.背景](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#%E8%83%8C%E6%99%AF)
2. [2.Ansible - 无需 Agent 远程设置 Linux 服务器](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#Ansible-%E6%97%A0%E9%9C%80-Agent-%E8%BF%9C%E7%A8%8B%E8%AE%BE%E7%BD%AE-Linux-%E6%9C%8D%E5%8A%A1%E5%99%A8)
3. [3.Terraform - 使用代码定义云上基础架构](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#Terraform-%E4%BD%BF%E7%94%A8%E4%BB%A3%E7%A0%81%E5%AE%9A%E4%B9%89%E4%BA%91%E4%B8%8A%E5%9F%BA%E7%A1%80%E6%9E%B6%E6%9E%84)
4. [4.小结](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#%E5%B0%8F%E7%BB%93)

[Menu](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#) [TOC](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#) [Top](https://blog.lyc8503.net/post/iac-explore-ansible-and-terraform/#)