# Исследование PyTorch Dynamic Quantization

## Обзор

**Dynamic Quantization** (динамическая квантизация) - это метод оптимизации, при котором:
- **Веса** квантизируются заранее (статически) в int8/float16
- **Активации** квантизируются "на лету" (динамически) во время inference

Это самый простой метод квантизации, не требующий калибровки или fine-tuning.

---

## 1. Как Dynamic Quantization ускоряет inference

### Механизм ускорения

1. **Уменьшение размера весов**: FP32 (4 байта) -> INT8 (1 байт) = 4x сжатие
2. **Меньше memory bandwidth**: Быстрее загружаются веса из памяти
3. **INT8 матричные операции**: Используются оптимизированные SIMD инструкции (AVX2, AVX512-VNNI)

### Почему это работает для Transformer моделей

Transformer-based модели (BERT, GPT, Qwen) состоят преимущественно из Linear слоев.
При small batch size время выполнения доминируется **загрузкой весов из памяти**, а не вычислениями.
Динамическая квантизация максимально эффективна именно в этом сценарии.

```
┌─────────────────────────────────────────────────────────┐
│                   FP32 Inference                        │
│  Load weights (4 bytes) -> Compute FP32 -> Output      │
│  Memory: 4 bytes/weight, Compute: FP32 matmul          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              Dynamic Quantization INT8                  │
│  Load weights (1 byte) -> Quantize activations ->      │
│  INT8 matmul -> Dequantize -> Output                   │
│  Memory: 1 byte/weight, Compute: INT8 matmul           │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Поддерживаемые слои

### torch.quantization.quantize_dynamic поддерживает:

| Слой | Поддержка | Примечание |
|------|-----------|------------|
| `nn.Linear` | ✅ Да | Основной use case |
| `nn.LSTM` | ✅ Да | Полная поддержка |
| `nn.GRU` | ✅ Да | Полная поддержка |
| `nn.RNN` | ✅ Да | Полная поддержка |
| `nn.LSTMCell` | ✅ Да | Поддерживается |
| `nn.GRUCell` | ✅ Да | Поддерживается |
| `nn.RNNCell` | ✅ Да | Поддерживается |
| `nn.Conv1d/2d/3d` | ❌ Нет | Требуется Static Quantization |
| `nn.Embedding` | ❌ Нет | Не квантизируется |
| `nn.LayerNorm` | ❌ Нет | Не квантизируется |
| `nn.MultiheadAttention` | ⚠️ Частично | Только Linear внутри |

### Для Transformer моделей:
- **Квантизируются**: все Linear слои (Q, K, V projections, FFN layers)
- **Не квантизируются**: Embeddings, LayerNorm, Softmax

---

## 3. GPU vs CPU поддержка

### КРИТИЧЕСКИ ВАЖНО: Dynamic Quantization работает ТОЛЬКО на CPU!

```python
# ЭТО ВЫЗОВЕТ ОШИБКУ:
quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
quantized_model.cuda()  # RuntimeError!
quantized_model(input.cuda())  # RuntimeError!
```

**Ошибка**: `RuntimeError: Could not run 'quantized::linear_dynamic' with arguments from the 'CUDA' backend`

### Поддерживаемые CPU backends:

| Backend | Платформа | Оптимизации |
|---------|-----------|-------------|
| **fbgemm** | x86 (Intel/AMD) | AVX2, AVX512, VNNI |
| **qnnpack** | ARM | NEON |

### Альтернативы для GPU:

1. **TorchAO** (рекомендуется PyTorch)
   - Int4/Int8 weight-only quantization на GPU
   - `torch.compile()` совместимость
   - Speedup: 1.5-1.73x на H100

2. **Quanto (HuggingFace)**
   - int8-int8, fp16-int4, bf16-int8 на CUDA

3. **TensorRT**
   - INT8 inference на NVIDIA GPU

---

## 4. Примеры кода для Transformer моделей

### Пример 1: Базовая Dynamic Quantization для BERT

```python
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer

# Загрузка модели
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model.eval()

# Применение динамической квантизации
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # Квантизировать только Linear слои
    dtype=torch.qint8  # INT8 квантизация
)

# Inference (ТОЛЬКО на CPU!)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello, world!", return_tensors="pt")

with torch.no_grad():
    outputs = quantized_model(**inputs)

print(f"Original size: {sum(p.numel() * 4 for p in model.parameters()) / 1e6:.1f} MB")
print(f"Quantized size: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f} MB")
```

### Пример 2: Dynamic Quantization для LSTM

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Создание и квантизация
model = LSTMModel(10000, 256, 512, 2)
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.LSTM, nn.Linear},  # Квантизируем LSTM и Linear
    dtype=torch.qint8
)

# Проверка квантизации
print(quantized_model.lstm)  # DynamicQuantizedLSTM
print(quantized_model.fc)    # DynamicQuantizedLinear
```

### Пример 3: Новый API - torch.ao.quantization

```python
import torch
import torch.ao.quantization as ao_quant

# Новый рекомендуемый API (torch.ao.quantization)
quantized_model = ao_quant.quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Linear},  # или qconfig_spec для fine-grained control
    dtype=torch.qint8,
    inplace=False
)
```

### Пример 4: TorchAO (Рекомендуется для новых проектов)

```python
# pip install torchao
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
import torch

# TorchAO - новый стандарт квантизации в PyTorch
model = ...  # Ваша модель
model.eval()

# Применение int8 динамической квантизации
quantize_(model, Int8DynamicActivationInt8WeightConfig())

# Для лучшей производительности - компиляция
model = torch.compile(model, mode="max-autotune")

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

### Пример 5: Benchmarking код

```python
import torch
import torch.nn as nn
import time

def benchmark_model(model, input_tensor, num_runs=100, warmup=10):
    """Benchmark inference time"""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # Benchmark
    torch.cuda.synchronize() if input_tensor.is_cuda else None
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)

    torch.cuda.synchronize() if input_tensor.is_cuda else None
    end = time.perf_counter()

    return (end - start) / num_runs * 1000  # ms per inference

# Использование
original_time = benchmark_model(model.cpu(), input_cpu)
quantized_time = benchmark_model(quantized_model, input_cpu)

print(f"Original: {original_time:.2f} ms")
print(f"Quantized: {quantized_time:.2f} ms")
print(f"Speedup: {original_time / quantized_time:.2f}x")
```

---

## 5. Реальные цифры ускорения

### Официальные бенчмарки PyTorch (BERT на MRPC)

| Метрика | FP32 | INT8 (Dynamic) | Улучшение |
|---------|------|----------------|-----------|
| **Model Size** | 438 MB | 181 MB | **2.4x меньше** |
| **Non-embedding Size** | 350 MB | 90 MB | **3.9x меньше** |
| **F1 Score** | 0.9019 | 0.902 | -0.1% (negligible) |
| **Time (1 thread)** | 160 sec | 90 sec | **1.78x быстрее** |
| **Time (4 threads)** | 85 sec | 46 sec | **1.85x быстрее** |

*Тестировалось на MacBook Pro, датасет MRPC (408 примеров)*

### Зависимость от железа

| Платформа | Speedup | Примечание |
|-----------|---------|------------|
| **Intel AVX512-VNNI** (c5.12xlarge) | **2.5x** | Лучший результат |
| **Intel AVX512** (без VNNI) | 1.5-2x | Хороший |
| **Intel AVX2** | 1.2-1.5x | Умеренный |
| **Older Intel** (Xeon E5-2620 v4) | ~1x | Минимальный или нет |
| **ARM with NEON** | 1.3-1.8x | Зависит от модели |

### Общие ожидания

```
┌────────────────────────────────────────────────────────────────┐
│  Ожидаемые результаты Dynamic Quantization                    │
├────────────────────────────────────────────────────────────────┤
│  Model Size:        2-4x уменьшение                           │
│  Inference Speed:   1.5-3x ускорение (зависит от железа)      │
│  Accuracy Loss:     < 1% (обычно 0.1-0.6%)                    │
│  Memory Bandwidth:  2-4x уменьшение                           │
└────────────────────────────────────────────────────────────────┘
```

### Реальные кейсы из индустрии

1. **Roblox** - 10x throughput improvement для BERT на CPU
2. **HuggingFace** - 30-50% speedup для Transformer моделей

### Когда НЕ ожидать ускорения:

- GPU inference (не поддерживается)
- Старые CPU без AVX2/AVX512
- Large batch size (compute-bound, не memory-bound)
- Модели с большим количеством Conv слоев

---

## 6. Важные предупреждения

### Deprecation Notice (PyTorch 2.10+)

```python
# УСТАРЕВШИЙ API (будет удален в PyTorch 2.10):
import torch.quantization
quantized = torch.quantization.quantize_dynamic(...)

# РЕКОМЕНДУЕМЫЙ API:
import torch.ao.quantization
quantized = torch.ao.quantization.quantize_dynamic(...)

# НОВЫЙ СТАНДАРТ (TorchAO):
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
quantize_(model, Int8DynamicActivationInt8WeightConfig())
```

### Checklist перед использованием

- [ ] Модель будет работать на CPU
- [ ] CPU поддерживает AVX2/AVX512 (проверить: `lscpu | grep avx`)
- [ ] Модель содержит Linear/LSTM/GRU слои
- [ ] Small batch size (1-32)
- [ ] Допустима потеря точности ~0.5%

---

## Источники

1. [PyTorch Quantization Documentation](https://docs.pytorch.org/docs/stable/quantization.html)
2. [quantize_dynamic API Reference](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html)
3. [Dynamic Quantization on BERT Tutorial](https://docs.pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)
4. [TorchAO GitHub](https://github.com/pytorch/ao)
5. [Practical Quantization in PyTorch Blog](https://pytorch.org/blog/quantization-in-practice/)
6. [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
7. [TorchAO Documentation](https://docs.pytorch.org/ao/stable/quantization_overview.html)
