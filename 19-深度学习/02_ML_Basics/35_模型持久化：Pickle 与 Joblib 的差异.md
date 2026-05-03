# 35_模型持久化：Pickle 与 Joblib 的差异

## 核心概念

- **模型持久化**：将训练好的机器学习模型保存到磁盘，以便后续加载使用，避免每次使用时重新训练。
- **Pickle**：Python 的标准序列化模块，可将任意 Python 对象转换为字节流，保存到文件后再反序列化恢复。
- **Joblib**：sklearn 生态中推荐的序列化工具，专门针对大规模 NumPy 数组和 sklearn 模型进行了优化。
- **核心差异**：Joblib 对大型 NumPy 数组采用**按内存视图序列化**而非逐对象遍历，速度更快、文件更小；Pickle 更通用但处理大数组时效率低。
- **压缩支持**：Joblib 内置支持多种压缩格式（zlib, gzip, lz4, lzma），可以显著减小模型文件大小（对 Tree-based 模型尤其明显）。
- **安全性**：Pickle 和 Joblib 加载时都会执行任意代码，因此不能从不可信来源加载序列化文件。安全的替代方案有 ONNX、PMML 等跨平台格式。

## 技术对比

**序列化机制对比**：
- Pickle：递归遍历 Python 对象的属性树，对每个对象调用 `__reduce__` 方法序列化。NumPy 数组被序列化为二进制数据块，但需要经过 Python 对象层的包装。
- Joblib：对 numpy 数组使用专门的序列化器，直接保存数组的底层内存缓冲区（buffer protocol），对大数组的序列化速度比 pickle 快 10-100 倍。

**文件大小**：
对于包含大型 numpy 数组的模型（如线性模型的系数矩阵、SVM 的支持向量），joblib 通常比 pickle 小。对于决策树/随机森林等由小型对象组成的模型，两者大小接近。

**兼容性**：
- Pickle：Python 内置，所有 Python 版本都支持。但不同 Python 版本间的 pickle 格式可能不兼容（尤其是 Python 2 vs 3）。
- Joblib：需要单独安装（`pip install joblib`），但更加稳定，在不同 numpy/sklearn 版本间的兼容性更好。

**适用场景**：
- Joblib 更适合：sklearn 模型（尤其是含大量数组的模型）、大规模 numpy 数组
- Pickle 更适合：通用 Python 对象、复杂自定义类的序列化、不含大数组的模型

## 代码示例

```python
import numpy as np
import pickle
import joblib
import time
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.svm import SVR

# 生成大规模数据和模型
X, y = make_regression(n_samples=10000, n_features=100, noise=0.1)
rf_model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf_model.fit(X, y)

svm_model = SVR()
svm_model.fit(X[:500], y[:500])

# Pickle 序列化
start = time.time()
with open('model_rf.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
pickle_time = time.time() - start
pickle_size = os.path.getsize('model_rf.pkl')

# Joblib 序列化
start = time.time()
joblib.dump(rf_model, 'model_rf.joblib')
joblib_time = time.time() - start
joblib_size = os.path.getsize('model_rf.joblib')

# Joblib 压缩序列化
start = time.time()
joblib.dump(rf_model, 'model_rf_compressed.joblib', compress=True)
joblib_comp_time = time.time() - start
joblib_comp_size = os.path.getsize('model_rf_compressed.joblib')

print(f"Pickle:       时间={pickle_time:.3f}s, 大小={pickle_size/1024:.1f}KB")
print(f"Joblib:       时间={joblib_time:.3f}s, 大小={joblib_size/1024:.1f}KB")
print(f"Joblib压缩:   时间={joblib_comp_time:.3f}s, 大小={joblib_comp_size/1024:.1f}KB")

# 验证加载后模型等效
loaded1 = joblib.load('model_rf.joblib')
loaded2 = pickle.load(open('model_rf.pkl', 'rb'))
X_test = np.random.randn(10, 100)
print(f"预测一致: {np.allclose(loaded1.predict(X_test), loaded2.predict(X_test))}")

# 清理文件
for f in ['model_rf.pkl', 'model_rf.joblib', 'model_rf_compressed.joblib']:
    if os.path.exists(f):
        os.remove(f)
```

## 深度学习关联

- **PyTorch 模型保存**：深度学习框架有自己的序列化格式。PyTorch 推荐使用 `torch.save(model.state_dict(), 'model.pth')` 仅保存模型参数（而非整个对象），文件小、兼容性好。这与 joblib 只保存数组的思想一致——大参数用专用格式，而非通用对象序列化。
- **HDF5 / SavedModel**：TensorFlow 使用 SavedModel 或 HDF5 格式保存模型，这两种格式都是跨语言、跨平台的标准格式。HDF5 与 joblib 类似，对大数组（权重张量）做了专门的存储优化。
- **ONNX 跨平台格式**：ONNX (Open Neural Network Exchange) 是深度学习模型的通用中间表示格式，支持跨框架（PyTorch → ONNX → TensorFlow）和跨平台部署。相比 pickle/joblib，ONNX 更安全（不执行 Python 代码）、更适合生产部署环境。
