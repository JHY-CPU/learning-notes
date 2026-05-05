## Reservoir Sampling


```javascript
从数据流中随机等概率抽取 k 个元素，无需事先知道总数。```


```
function reservoirSample(stream, k) {
  const reservoir = [];
  for (let i = 0; i < k && i < stream.length; i++) reservoir.push(stream[i]);
  for (let i = k; i < stream.length; i++) {
    const j = Math.floor(Math.random() * (i + 1));
    if (j < k) reservoir[j] = stream[i];
  }
  return reservoir;
}
const stream = [1,2,3,4,5,6,7,8,9,10];
console.log(reservoirSample(stream, 3));
console.log(reservoirSample(stream, 3));
console.log(reservoirSample(stream, 3)); // 每次结果不同```


  点击按钮查看结果
