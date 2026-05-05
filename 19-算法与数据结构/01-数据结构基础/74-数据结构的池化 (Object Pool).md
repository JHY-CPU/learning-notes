## Object Pool


```javascript
对象池模式复用对象减少内存分配和GC压力，适合频繁创建/销毁的场景。```


```
class ObjectPool {
  constructor(factory, reset, initialSize=10) {
    this.factory = factory;
    this.reset = reset;
    this.pool = [];
    for (let i = 0; i < initialSize; i++) this.pool.push(factory());
  }
  acquire() {
    if (this.pool.length > 0) return this.pool.pop();
    return this.factory();
  }
  release(obj) {
    this.reset(obj);
    this.pool.push(obj);
  }
  size() { return this.pool.length; }
}
// 用法示例
const vecPool = new ObjectPool(
  () => ({x:0, y:0}),
  (v) => { v.x=0; v.y=0; }
);```


  点击按钮查看结果
