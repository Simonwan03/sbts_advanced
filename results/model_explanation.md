# Benchmark Model Explanation

本文档总结 `benchmark_main_pipeline.ipynb` 默认 benchmark 中使用的模型、架构和主要参数。

默认情况下，notebook 使用：

```python
MODELS_TO_RUN = list(list_models().keys())
```

因此实际模型列表来自 `models/factory.py` 的 `MODEL_REGISTRY`。当前默认会测试 10 个模型：

1. `jd_sbts`
2. `jd_sbts_f`
3. `jd_sbts_neural`
4. `jd_sbts_f_neural`
5. `lightsb`
6. `numba_sb`
7. `timegan`
8. `diffusion_ts`
9. `rnn`
10. `transformer_ar`

## Parameter Scope

这里的“参数”分成两类：

- **Trainable neural parameters**：神经网络中可训练的权重和偏置，例如 LSTM、GRU、Transformer、MLP denoiser、score network。
- **Calibrated / non-neural state**：非神经校准对象，例如 local volatility surface、jump intensity/mean/std、Brownian bridge 的初末状态样本、数值 solver 配置。

这两类不能简单混为一谈。比如 `numba_sb` 的神经参数量是 0，但它仍然存储训练数据的初末状态；SBTS 系列的神经参数量主要来自 drift/jump 网络，但模型复杂度还包括 jump detection、volatility calibration 和 SDE solver。

下面的参数量估算基于默认 `merton` benchmark：

```text
BENCHMARK_DATASET = "merton"
representation = "path"
seq_len = 101
n_features = 1
```

如果换成 Google 数据集，`n_features=6`，部分模型参数量会变化；如果换成不同窗口长度，`lightsb` 的 score network 参数量也会变化，因为它会 flatten 整条窗口。

notebook 中还会覆盖部分训练轮数：

```text
lstm_epochs = 25
timegan_epochs = 20
diffusion_epochs = 25
lightsb_epochs = 25
rnn_epochs = 25
transformer_ar_epochs = 25
```

## Summary Table

| Model | Type | Main architecture | Approx. trainable neural params |
|---|---|---|---:|
| `jd_sbts` | Jump-Diffusion SBTS | Static jumps + local volatility + LSTM drift + SDE solver | 215,809 |
| `jd_sbts_f` | SBTS with feedback | `jd_sbts` + stress-factor volatility feedback | 215,809 |
| `jd_sbts_neural` | SBTS with neural jumps | `jd_sbts` + LSTM jump-intensity network | 268,610 |
| `jd_sbts_f_neural` | SBTS with feedback and neural jumps | Neural jumps + feedback stress factor | 268,610 |
| `lightsb` | Light Schrodinger Bridge | Flattened window + time-conditioned MLP score network | 138,533 |
| `numba_sb` | Brownian bridge baseline | Closed-form / Numba Markovian SB | 0 |
| `timegan` | GAN baseline | GRU embedder/recovery/generator/supervisor/discriminator | 219,010 |
| `diffusion_ts` | DDPM-style baseline | Time-conditioned MLP denoiser | 78,913 |
| `rnn` | Autoregressive baseline | 2-layer LSTM forecaster | 50,497 |
| `transformer_ar` | Autoregressive baseline | Causal Transformer forecaster | 67,265 |

## 1. `jd_sbts`

`jd_sbts` 是基础 Jump-Diffusion Schrodinger Bridge Time Series 模型，实现在 `models/sbts_variants.py` 的 `JDSBTS`。

它不是一个单一神经网络，而是一个组合式半参数生成模型：

```text
raw data
  -> static jump detection
  -> filter and interpolate jumps
  -> local volatility calibration on purified data
  -> LSTM drift estimation on purified data
  -> Euler-Maruyama jump-diffusion generation
```

主要组件：

- `StaticJumpDetector`：用 rolling z-score threshold 检测跳跃。
- `LocalVolatilityCalibrator`：用 Nadaraya-Watson / KDE 思路估计 local volatility surface。
- `LSTMDriftEstimator`：在 purified data 上学习 drift。
- `JumpDiffusionEulerSolver`：用 Euler-Maruyama 生成跳扩散路径。

默认关键参数：

```text
jump_threshold_std = 4.0
jump_rolling_window = 20
use_neural_jumps = False

vol_bandwidth = 0.5
vol_n_t_grid = 50
vol_n_x_grid = 100

drift_estimator = "lstm"
lstm_hidden = 128
lstm_epochs = 25   # notebook override; factory default is 50
lstm_lr = 0.005
lstm_dropout = 0.3
lstm_use_huber = True

use_feedback = False
solver_backend = "numba"
```

LSTM drift network 架构：

```text
Input dimension = n_features
LSTM:
  hidden_dim = 128
  num_layers = 2
  dropout = 0.3
  bidirectional = False
FC:
  Linear(128 -> 128)
  ReLU
  Dropout(0.3)
  Linear(128 -> n_features)
```

在 `merton` 一维输入下：

```text
LSTM drift params ~= 215,809
local volatility surface size ~= 50 * 100 * 1 = 5,000 grid values
jump calibrated parameters ~= intensity, mean, std per feature
```

## 2. `jd_sbts_f`

`jd_sbts_f` 是 `JDSBTSF`，即 `jd_sbts` 的 feedback 版本。它在基础 SBTS 的生成阶段加入 jump-volatility interaction，用一个 transient stress factor 表示跳跃后波动率聚集。

生成动态可以概括为：

```text
dX_t = mu(t, X_t) dt + sigma_LV(t, X_t) * sqrt(1 + S_t) dW_t + dJ_t
dS_t = -kappa * S_t dt + gamma * |dJ_t|
```

相比 `jd_sbts`，它额外使用：

```text
use_feedback = True
feedback_kappa = 5.0
feedback_gamma = 0.5
```

架构差异：

- jump detector、local volatility、drift LSTM 与 `jd_sbts` 相同。
- 生成时每次发生 jump 后更新 stress factor。
- effective volatility 被放大为 `sigma_LV * sqrt(1 + S_t)`。
- stress factor 会以 `kappa` 控制的速度衰减。

参数量：

```text
LSTM drift params ~= 215,809
feedback trainable neural params = 0
total trainable neural params ~= 215,809
```

feedback 是显式机制，不是神经网络，所以不会明显增加 trainable neural parameters。

## 3. `jd_sbts_neural`

`jd_sbts_neural` 是 `JDSBTSNeural`，在基础 SBTS 上启用 neural jump detection。

核心变化：

```text
static jump detector
  -> produce initial jump labels
  -> train LSTM intensity network
  -> sample time-varying jump intensity during generation
```

Neural jump detector 架构：

```text
Input features:
  returns
  abs(returns)
Input dimension = 2 * n_features

LSTM:
  hidden_dim = 64
  num_layers = 2
  dropout = 0.1

FC:
  Linear(64 -> 32)
  ReLU
  Dropout(0.1)
  Linear(32 -> 1)
  Softplus
```

训练目标：

- 先用 static detector 产生 jump labels。
- 用 Focal Loss 处理 jump 稀疏和类别不平衡。
- 网络输出 jump intensity，再转换成 jump probability。

默认关键参数：

```text
use_neural_jumps = True

neural_jump_hidden_dim = 64
neural_jump_epochs = 30
neural_jump_lr = 0.001
neural_jump_seq_len = 10

focal_alpha = 0.25
focal_gamma = 2.0
```

参数量：

```text
LSTM drift params ~= 215,809
LSTM jump intensity params ~= 52,801
total trainable neural params ~= 268,610
```

非神经组件仍然包括 local volatility surface 和 static jump detector 产生的基础 jump statistics。

## 4. `jd_sbts_f_neural`

`jd_sbts_f_neural` 是 `JDSBTSFNeural`，是 SBTS 系列中最完整的版本：

```text
jd_sbts
  + neural jump detector
  + feedback stress factor
```

包含组件：

- Static detector for initial jump labels。
- LSTM jump-intensity network。
- Local volatility surface。
- LSTM drift estimator。
- Euler-Maruyama jump-diffusion solver。
- Stress-factor volatility feedback。

关键参数是 `jd_sbts_neural` 和 `jd_sbts_f` 的并集：

```text
use_neural_jumps = True
use_feedback = True

feedback_kappa = 5.0
feedback_gamma = 0.5

neural_jump_hidden_dim = 64
neural_jump_epochs = 30
neural_jump_lr = 0.001
neural_jump_seq_len = 10
```

参数量：

```text
LSTM drift params ~= 215,809
LSTM jump intensity params ~= 52,801
feedback trainable neural params = 0
total trainable neural params ~= 268,610
```

## 5. `lightsb`

`lightsb` 是 `LightSB`，实现在 `models/lightsb.py`。它是 Light Schrodinger Bridge 风格模型，使用 variance annealing、mini-batch OT 和 score matching。

训练逻辑：

```text
time-series window
  -> normalize
  -> flatten to high-dimensional vector
  -> couple Gaussian source and data target via Sinkhorn OT
  -> sample interpolated bridge points
  -> train score network with score matching
```

对默认 `merton` 数据：

```text
seq_len = 101
n_features = 1
flat_dim = 101
```

Score network 架构：

```text
Time embedding:
  Linear(1 -> 64)
  SiLU
  Linear(64 -> 64)

Main MLP:
  input = flat_dim + 64
  Linear(input -> 256)
  SiLU
  Linear(256 -> 256)
  SiLU
  Linear(256 -> flat_dim)
```

默认关键参数：

```text
lightsb_sigma_min = 0.01
lightsb_sigma_max = 1.0
lightsb_hidden_dim = 256
lightsb_n_layers = 3
lightsb_epochs = 25   # notebook override; factory default is 100
lightsb_lr = 0.001
lightsb_batch_size = 256
lightsb_ot_epsilon = 0.1
lightsb_n_steps = 50
```

参数量：

```text
score network params ~= 138,533
```

复杂度 caveat：

- 参数量只描述 score network。
- 实际训练成本还受到 mini-batch OT / Sinkhorn 的影响。
- 因为输入是 flatten 后的整条窗口，窗口越长，score network 输入输出维度越大。

## 6. `numba_sb`

`numba_sb` 是 `NumbaSB`，实现在 `models/lightsb.py`。它是一个 Numba 加速的 Markovian Schrodinger Bridge / Brownian bridge baseline。

它没有神经网络训练。fit 阶段主要存储数据统计：

```text
x0_samples = training windows 的初始状态
x1_samples = training windows 的终止状态
sigma = std(returns) / sqrt(dt)
```

生成时使用 Brownian bridge closed form：

```text
mean(t) = x0 + (x1 - x0) * t / T
var(t) = sigma^2 * t * (T - t) / T
```

默认参数：

```text
numba_sb_sigma = 0.1
```

注意：`numba_sb_sigma` 是初始值，fit 后会用训练数据估计出来的 `sigma` 覆盖。

参数量：

```text
trainable neural params = 0
```

非神经状态：

```text
x0_samples: shape ~= (n_train, n_features)
x1_samples: shape ~= (n_train, n_features)
sigma: scalar
```

## 7. `timegan`

`timegan` 是 `TimeGAN` baseline，实现在 `models/timegan_baseline.py`。它使用 GRU 组成的 TimeGAN 风格结构。

子网络：

```text
EmbeddingNetwork:
  GRU(input_dim -> hidden_dim)
  Linear(hidden_dim -> hidden_dim)
  Sigmoid

RecoveryNetwork:
  GRU(hidden_dim -> hidden_dim)
  Linear(hidden_dim -> output_dim)

GeneratorNetwork:
  GRU(z_dim -> hidden_dim)
  Linear(hidden_dim -> hidden_dim)
  Sigmoid

SupervisorNetwork:
  GRU(hidden_dim -> hidden_dim)
  Linear(hidden_dim -> hidden_dim)
  Sigmoid

DiscriminatorNetwork:
  GRU(hidden_dim -> hidden_dim)
  Linear(hidden_dim -> 1)
```

默认关键参数：

```text
timegan_hidden_dim = 64
timegan_z_dim = 32
timegan_n_layers = 2
timegan_epochs = 20   # notebook override; factory default is 50
timegan_lr = 0.001
timegan_batch_size = 128
```

训练阶段：

```text
Phase 1: autoencoder training
  embedder + recovery reconstruct real data

Phase 2: supervised training
  supervisor predicts next latent step

Phase 3: joint adversarial training
  generator + supervisor fool discriminator
  discriminator distinguishes real/fake latent sequences
```

参数量分解，按 `n_features=1`：

```text
embedder ~= 41,984
recovery ~= 49,985
generator ~= 47,936
supervisor ~= 29,120
discriminator ~= 49,985
total ~= 219,010
```

## 8. `diffusion_ts`

`diffusion_ts` 是 `DiffusionTS`，实现在 `models/diffusion_ts_baseline.py`。这是一个简化 DDPM-style time-series baseline。

它的 denoiser 名字是 `DiffusionUNet`，但实际结构是 MLP-style per-time-step denoiser，不是卷积 U-Net。

Forward diffusion：

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
```

训练目标：

```text
predict_noise = denoiser(x_t, t)
loss = MSE(predict_noise, true_noise)
```

Denoiser 架构：

```text
Time embedding:
  Linear(1 -> 64)
  SiLU
  Linear(64 -> 64)

Per-time-step MLP:
  concat(x_t, time_embedding)
  Linear(input_dim + 64 -> hidden_dim)
  SiLU
  Linear(hidden_dim -> 2 * hidden_dim)
  SiLU
  Linear(2 * hidden_dim -> hidden_dim)
  SiLU
  skip concat with first hidden layer
  Linear(2 * hidden_dim -> input_dim)
```

默认关键参数：

```text
diffusion_hidden_dim = 128
diffusion_n_steps = 100
diffusion_epochs = 25   # notebook override; factory default is 100
diffusion_lr = 0.001
diffusion_batch_size = 64
diffusion_beta_start = 0.0001
diffusion_beta_end = 0.02
```

参数量，按 `input_dim=1`：

```text
denoiser params ~= 78,913
```

生成复杂度 caveat：

- 生成需要从 `diffusion_n_steps - 1` 反向迭代到 0。
- 因此生成时间与 `diffusion_n_steps` 直接相关。

## 9. `rnn`

`rnn` 是 `RNNBaseline`，实现在 `models/rnn_baseline.py`。它是标准自回归 RNN baseline。

训练目标：

```text
input:  x_0, ..., x_{T-2}
target: x_1, ..., x_{T-1}
loss:   MSE next-step prediction
```

生成方式：

```text
start from real prefix
roll out one step at a time autoregressively
```

默认架构是 LSTM：

```text
RNNForecastNet:
  LSTM(input_dim -> hidden_dim)
  num_layers = 2
  dropout = 0.1
  bidirectional = False
  Linear(hidden_dim -> input_dim)
```

默认关键参数：

```text
rnn_hidden_dim = 64
rnn_num_layers = 2
rnn_dropout = 0.1
rnn_bidirectional = False
rnn_type = "lstm"
rnn_epochs = 25   # notebook override; factory default is 50
rnn_lr = 0.001
rnn_batch_size = 128
rnn_context_len = None
```

`rnn_context_len = None` 时会自动取：

```text
context_len = max(5, seq_len // 2)
```

对默认 `merton`：

```text
seq_len = 101
context_len = 50
```

参数量，按 `input_dim=1`：

```text
2-layer LSTM + readout params ~= 50,497
```

## 10. `transformer_ar`

`transformer_ar` 是 `TransformerARBaseline`，实现在 `models/transformer_ar_baseline.py`。它是 causal autoregressive Transformer baseline。

虽然代码使用 `TransformerEncoderLayer`，但通过 causal mask 实现 decoder-only / autoregressive 行为。

训练目标：

```text
input:  x_0, ..., x_{T-2}
target: x_1, ..., x_{T-1}
loss:   MSE next-step prediction
```

架构：

```text
input_proj:
  Linear(input_dim -> d_model)

positional encoding:
  sinusoidal positional encoding

Transformer:
  n_layers = 2
  n_heads = 4
  d_model = 64
  d_ff = 128
  dropout = 0.1
  activation = GELU
  causal attention mask

output:
  LayerNorm(d_model)
  Linear(d_model -> input_dim)
```

默认关键参数：

```text
transformer_ar_d_model = 64
transformer_ar_n_heads = 4
transformer_ar_n_layers = 2
transformer_ar_d_ff = 128
transformer_ar_dropout = 0.1
transformer_ar_epochs = 25   # notebook override; factory default is 50
transformer_ar_lr = 0.001
transformer_ar_batch_size = 128
transformer_ar_max_seq_len = 64
transformer_ar_context_len = None
```

参数量，按 `input_dim=1`：

```text
causal Transformer params ~= 67,265
```

Implementation caveat:

当前 notebook 使用：

```python
config.setdefault("transformer_ar_max_seq_len", data.shape[1])
```

但 `get_default_config("transformer_ar")` 已经设置了 `transformer_ar_max_seq_len = 64`，所以 `setdefault` 不会覆盖它。默认 `merton` path 的训练输入长度约为 100，这可能导致 Transformer 在训练时遇到 max sequence length 不足的问题。

更稳妥的写法是显式更新：

```python
config["transformer_ar_max_seq_len"] = max(
    config.get("transformer_ar_max_seq_len", 0),
    data.shape[1],
)
```

## Interpretation

如果只比较 trainable neural parameters：

```text
largest group:
  jd_sbts_neural / jd_sbts_f_neural ~= 268,610
  timegan ~= 219,010
  jd_sbts / jd_sbts_f ~= 215,809

middle group:
  lightsb ~= 138,533
  diffusion_ts ~= 78,913
  transformer_ar ~= 67,265
  rnn ~= 50,497

non-neural baseline:
  numba_sb = 0
```

但这个排序不能直接等价于“模型复杂度”或“计算成本”。原因是：

- SBTS 系列还有 jump detection、local volatility surface、SDE solver。
- LightSB 的训练成本很大程度来自 mini-batch OT / Sinkhorn，而不只是 score network 参数量。
- Diffusion 的生成成本与 reverse diffusion steps 相关。
- NumbaSB 虽然没有神经参数，但会存储经验初末状态样本。

更合理的比较方式是同时报告：

1. trainable neural parameters；
2. calibrated / stored state size；
3. training time；
4. generation time；
5. benchmark metrics，例如 Wasserstein distance、ACF MSE、predictive score。

