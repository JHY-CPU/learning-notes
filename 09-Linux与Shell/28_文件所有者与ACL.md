# 文件所有者与ACL


## 📁 文件所有者与 ACL


chown/chgrp 修改所有者、ACL 访问控制列表、文件特殊属性、umask。


## chown — 修改文件所有者


```
// ========== chown 基础 ==========
// chown = change owner
// 只有 root 可以修改文件所有者

chown alice file.txt              # 修改所有者为 alice
chown alice:developers file.txt   # 修改所有者和组
chown :developers file.txt        # 只修改组
chown alice: file.txt             # 修改为 alice + alice 的主组

// ========== 递归修改 ==========
chown -R alice:developers /home/alice  # 递归修改目录
chown -R --preserve-root alice /       # --preserve-root 防止 / 被递归

// ========== 符号链接 ==========
// chown 默认修改符号链接的目标文件
// 使用 -h 修改链接本身 (部分系统需要)
chown -h alice symlink

// ========== 常用模式 ==========
// 部署后设置 Web 目录权限:
sudo chown -R www-data:www-data /var/www
sudo chown -R alice:alice /home/alice
sudo chown -R $USER:$USER ~/.config

// 安全设置: 文件归应用用户所有,其他用户不可读
sudo chown -R appuser:appuser /opt/myapp
sudo chmod -R 750 /opt/myapp

// ========== chgrp — 修改组 ==========
// chown 可以同时改组,但 chgrp 更明确
chgrp developers file.txt          # 修改组为 developers
chgrp -R developers /shared/project  # 递归改组
chgrp --reference=ref.txt target.txt  # 参考其他文件的组
```


## ACL — 访问控制列表


```
// ========== ACL 概念 ==========
// ACL 提供比传统 rwx 更细粒度的权限控制
// 可以为特定用户或组设置权限
// 需要文件系统支持 (ext4/xfs/btrfs 都支持)

// 查看 ACL 支持:
mount | grep acl
tune2fs -l /dev/sda1 | grep "Default mount options"

// ========== getfacl — 查看 ACL ==========
getfacl file.txt
// # file: file.txt
// # owner: alice
// # group: developers
// user::rw-
// user:bob:r--
// group::r--
// group:devops:rw-
// mask::rw-
// other::r--

// ========== setfacl — 设置 ACL ==========
// 给特定用户设置权限:
setfacl -m u:bob:rw file.txt      # bob 获得读写
setfacl -m u:bob:- file.txt       # 移除 bob 的所有权限

// 给特定组设置权限:
setfacl -m g:devops:rx /opt/app   # devops 组获得 rx
setfacl -m g:devops:- /opt/app    # 移除

// 设置默认 ACL (新建文件继承):
setfacl -m d:u:www-data:rx /var/www/project
// 之后在 /var/www/project 下创建的文件自动继承

// 递归设置 ACL:
setfacl -R -m u:bob:rx /opt/app

// ========== 删除 ACL ==========
setfacl -x u:bob file.txt          # 删除 bob 的 ACL 条目
setfacl -x g:devops file.txt       # 删除 devops 组的 ACL
setfacl -b file.txt                # 删除所有 ACL (回到基本权限)

// ========== ACL 效果 ==========
// ls -l 输出中 ACL 文件会有 +
ls -l file.txt
// -rw-rw-r--+ 1 alice developers ... file.txt
//            ↑ 加号表示设置了 ACL

// ========== ACL 实战场景 ==========
// Web 服务器需要读取项目文件:
setfacl -R -m u:www-data:rx /opt/myapp
setfacl -R -d -m u:www-data:rx /opt/myapp  # 新文件继承

// 多个运维人员共享日志:
setfacl -R -m g:ops:rx /var/log/app
setfacl -R -d -m g:ops:rx /var/log/app
```


> **Note:** 💡 ACL 在需要精细权限控制的场景非常有用:Web 服务器需要读取项目文件、多个团队共享目录。但不要过度使用 ACL——简单的 chown/chmod/chgrp 能解决问题时,优先用传统权限,避免权限管理复杂化。


## 文件属性 (chattr / lsattr)


```
// ========== chattr — 修改文件属性 ==========
// 在 ext4/xfs 文件系统上设置额外属性
// 比权限更底层,甚至 root 也受影响

// 常用属性:
// i (immutable):  不可修改 (连 root 也不能)
// a (append only):只能追加内容
// d (no dump):    dump 备份时跳过
// e (extent):     使用 extents (默认开启)

// ========== 常用 chattr 命令 ==========
sudo chattr +i /etc/passwd        # 锁定,防止修改
sudo chattr +i /etc/shadow        # 锁定密码文件
sudo chattr -i /etc/passwd        # 解锁

sudo chattr +a /var/log/app.log   # 只可追加 (防止日志篡改)
sudo chattr +i ~/.ssh/authorized_keys  # 锁定 SSH 密钥

// 递归设置目录:
sudo chattr -R +i /etc/nginx      # 锁定 Nginx 配置目录

// ========== lsattr — 查看属性 ==========
lsattr file.txt
lsattr -R /etc/nginx/             # 递归查看

// 输出:
// ----i-------- file.txt     # i 属性设置
// ----a-------- log/app.log # a 属性设置

// ========== 注意事项 ==========
// chattr +i 的威力:
// 1. root 也不能修改/删除
// 2. 需要 chattr -i 才能改
// 3. 可用于安全防护 (配置文件锁定)
// 4. 系统更新时记得先 -i

// ========== umask — 默认权限掩码 ==========
// umask 决定新建文件的默认权限
// 文件: 666 - umask
// 目录: 777 - umask

// 当前 umask:
umask                   # 查看 (通常 022)
umask -S                # 符号显示 u=rwx,g=rx,o=rx

// 设置 umask:
umask 022               # 文件 644, 目录 755 (常用)
umask 027               # 文件 640, 目录 750 (安全)
umask 077               # 文件 600, 目录 700 (私密)

// .bashrc 中持久设置:
// echo "umask 027" >> ~/.bashrc
```


## 实战场景


```
// ========== 场景: 多用户 Web 项目 ==========
# 创建项目目录
sudo mkdir -p /var/www/myapp
sudo chown -R alice:www-data /var/www/myapp
sudo chmod -R 750 /var/www/myapp

# Web 服务器需要读取
sudo setfacl -R -m u:www-data:rx /var/www/myapp
sudo setfacl -R -d -m u:www-data:rx /var/www/myapp

// ========== 场景: 共享日志目录 ==========
# 运维组可以读取日志
sudo mkdir -p /var/log/app
sudo chown -R appuser:appuser /var/log/app
sudo chmod 750 /var/log/app
sudo setfacl -R -m g:ops:rx /var/log/app
sudo setfacl -R -d -m g:ops:rx /var/log/app

// ========== 场景: 安全锁定 ==========
# 锁定关键配置文件
sudo chattr +i /etc/ssh/sshd_config
sudo chattr +i /etc/sudoers
sudo chattr +i /etc/nginx/nginx.conf

// ========== 场景: 共享协作目录 ==========
# SGID: 新建文件继承组
sudo mkdir -p /shared/project
sudo chown -R alice:project /shared/project
sudo chmod -R 2775 /shared/project  # SGID + rwxrwxr-x
```


## 练习


<!-- Converted from: 28_文件所有者与ACL.html -->
