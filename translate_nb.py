import json
import os

path = "/Users/lizhanhong/X/PSC/codes/SBTS/sbts_advanced/turing_test.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Define translations
translations = {
    # Markdown texts
    "这个 Notebook 实现了一个严格的定量评估指标 —— **Discriminative Score (判别分)**。\n": 
    "This Notebook implements a rigorous quantitative evaluation metric —— **Discriminative Score**.\n",
    
    "### 核心原理\n": "### Core Principle\n",
    
    "我们训练一个简单的神经网络分类器 (Discriminator)，它的任务是区分一段给定的时间序列是**真实的 (Real)** 还是 **生成的 (Synthetic)**。\n":
    "We train a simple neural network classifier (Discriminator) whose task is to distinguish whether a given time series is **Real** or **Synthetic**.\n",
    
    "*   **输入**：形状为 `(Sequence_Length, N_Assets)` 的时间序列片段。\n":
    "*   **Input**: Time series segment with shape `(Sequence_Length, N_Assets)`.\n",
    
    "*   **输出**：概率值 `p`，表示数据为真实的概率。\n":
    "*   **Output**: Probability value `p`, representing the probability that the data is real.\n",
    
    "### 评估标准\n": "### Evaluation Criteria\n",
    
    "*   **Accuracy $\\approx$ 50%**: 完美。分类器无法区分真假，只能瞎猜。说明生成数据极其逼真。\n":
    "*   **Accuracy $\\approx$ 50%**: Perfect. The classifier cannot distinguish between real and fake, and can only guess blindly. It indicates that the generated data is extremely realistic.\n",
    
    "*   **Accuracy $\\gg$ 50%**: 失败。分类器找到了生成数据的破绽（Artifacts）。\n":
    "*   **Accuracy $\\gg$ 50%**: Fail. The classifier found flaws (Artifacts) in the generated data.\n",
    
    "## 1. 数据准备与模型复现 (Data & Generation)\n":
    "## 1. Data Preparation & Model Reproduction (Data & Generation)\n",
    
    "为了进行测试，我们需要先生成一批高质量的合成数据。这里我们复用 `SBTS-LSTM` (Jump-Diffusion) 流程。":
    "To conduct the test, we first need to generate a batch of high-quality synthetic data. Here we reuse the `SBTS-LSTM` (Jump-Diffusion) workflow.",
    
    "## 2. 判别器训练 (Discriminator Training)\n":
    "## 2. Discriminator Training\n",
    
    "我们调用 `metrics.py` 中的 `DiscriminativeScore` 类。为了展示训练过程，我们稍微修改调用方式或直接在这里实例化，以便画出 Loss 曲线。":
    "We call the `DiscriminativeScore` class in `metrics.py`. To show the training process, we slightly modify the calling method or instantiate it directly here to plot the Loss curve.",
    
    "## 3. 结果分析与可视化 (Analysis)\n":
    "## 3. Result Analysis & Visualization (Analysis)\n",
    
    "### 3.1 训练曲线\n":
    "### 3.1 Training Curve\n",
    
    "观察 Loss 是否下降。如果 Accuracy 一直徘徊在 0.5 附近，说明 Discriminator 无法学习到有效的区分特征 —— 这正是我们想要的！":
    "Observe if the Loss decreases. If Accuracy hovers around 0.5, it means the Discriminator cannot learn effective distinguishing features —— this is exactly what we want!",
    
    "### 3.2 最终测试集评分 (Test Score)\n":
    "### 3.2 Final Test Score\n",
    
    "这是最终的结论性指标。":
    "This is the final conclusive metric.",
}

comment_translations = {
    "# 导入项目模块\n": "# Import project modules\n",
    "TICKER = ['SPY'] # 简化为单资产测试，多资产同理\n": "TICKER = ['SPY'] # Simplified for single asset test, similar for multi-asset\n",
    "N_SAMPLES = 2000 # 生成足够多的样本用于训练分类器\n": "N_SAMPLES = 2000 # Generate enough samples to train the classifier\n",
    "# 使用我们之前定义的 DiscriminativeScore 类\n": "# Use the previously defined DiscriminativeScore class\n",
    "# 注意：为了画图，我们这里稍微'打开'一下 train_and_evaluate 的黑盒\n": "# Note: To plot, we slightly 'open' the black box of train_and_evaluate here\n",
    "# 如果你只想看最终分数，直接调用 discriminator.train_and_evaluate 即可\n": "# If you only want the final score, you can call discriminator.train_and_evaluate directly\n",
    "# 这里我们手动跑一遍以便收集 loss history\n": "# Here we run it manually to collect loss history\n",
}

for cell in notebook['cells']:
    if cell['cell_type'] == 'markdown':
        new_source = []
        for line in cell['source']:
            if line in translations:
                new_source.append(translations[line])
            else:
                new_source.append(line)
        cell['source'] = new_source
    elif cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            replaced = False
            # Check for exact matches first
            if line in comment_translations:
                new_source.append(comment_translations[line])
                replaced = True
            
            if not replaced:
                new_source.append(line)
        cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
    # Add a newline at the end if the original had it, though json dump format is standard.
