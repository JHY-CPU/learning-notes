# path路径模块


## path 路径模块


join / resolve / basename / dirname / extname / parse / format / sep / delimiter。


## path 模块 API


```
// ========== path 路径操作 ==========
const path = require('path');

// ========== 路径拼接 ==========
path.join('/base', 'dir', 'file.txt');
// '/base/dir/file.txt' (自动处理分隔符)

path.resolve('dist', 'js', 'app.js');
// '/current/working/dir/dist/js/app.js' (绝对路径)

// ========== 路径解析 ==========
path.basename('/a/b/file.txt');      // 'file.txt'
path.basename('/a/b/file.txt', '.txt'); // 'file'
path.dirname('/a/b/file.txt');       // '/a/b'
path.extname('/a/b/file.txt');       // '.txt'

// ========== 路径对象 ==========
const parsed = path.parse('/home/user/app.js');
// { root: '/', dir: '/home/user', base: 'app.js',
//   ext: '.js', name: 'app' }

const formatted = path.format({
    dir: '/home/user',
    base: 'app.js'
});
// '/home/user/app.js'

// ========== 路径信息 ==========
path.isAbsolute('/a/b');  // true
path.isAbsolute('a/b');   // false
path.relative('/a/b', '/a/c'); // '../c'

// ========== 分隔符 ==========
path.sep;       // '/' (POSIX) 或 '\' (Windows)
path.delimiter; // ':' (POSIX) 或 ';' (Windows)
```


## 演示：路径操作

点击按钮查看


<!-- Converted from: 8_path路径模块.html -->
