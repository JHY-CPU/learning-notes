## 最短路径算法对比








  | 算法 | 时间复杂度 | 空间 | 负权 | 负环检测 | 类型 |
| --- | --- | --- | --- | --- | --- |
| BFS | O(V+E) | O(V) | N/A(无权) | 否 | 单源 |
| Dijkstra | O((V+E)logV) | O(V) | 否 | 否 | 单源 |
| Bellman-Ford | O(VE) | O(V) | 是 | 是 | 单源 |
| SPFA | O(kE)~O(VE) | O(V) | 是 | 是 | 单源 |
| Floyd | O(V^3) | O(V^2) | 是 | 是 | 全源 |

  ## 选择指南


    - 无权图 -> BFS

    - 非负权 -> Dijkstra

    - 可能有负权 -> Bellman-Ford/SPFA

    - 需要所有点对 -> Floyd



  ## 交互演示
