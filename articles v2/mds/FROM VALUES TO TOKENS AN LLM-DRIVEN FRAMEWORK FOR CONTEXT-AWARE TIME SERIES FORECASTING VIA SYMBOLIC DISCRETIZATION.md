# FROM VALUES TO TOKENS: AN LLM-DRIVEN FRAMEWORK FOR CONTEXT-AWARE TIME SERIES FORECASTING VIA SYMBOLIC DISCRETIZATION

Anonymous authors

Paper under double-blind review

*Under review as a conference paper at ICLR 2026.*

**Abstract.** Time series forecasting plays a vital role in supporting decision-making across a wide range of critical applications, including energy, healthcare, and finance. Despite recent advances, forecasting accuracy remains limited due to the challenge of integrating historical numerical sequences with contextual features, which often comprise unstructured textual data. To address this challenge, we propose TokenCast, a large language model (LLM)-driven framework that leverages language-based symbolic representations as a unified intermediary for context-aware time series forecasting. Specifically, TokenCast employs a discrete tokenizer to transform continuous numerical sequences into temporal tokens, enabling structural alignment with language-based inputs. To effectively bridge the semantic gap between modalities, both temporal and contextual tokens are embedded into a shared representation space via a pre-trained LLM, further optimized with autoregressive generative objectives. Building upon this unified semantic space, the aligned LLM is subsequently fine-tuned in a supervised manner to predict future temporal tokens, which are then decoded back into the original numerical space. Extensive experiments are conducted on multiple real-world datasets, whose results reveal the performance of our framework and highlight its potential as a generative framework for multimodal time series forecasting. The code is available for further research at: `https://anonymous.4open.science/r/TokenCast-8EFF`.

## 1. Introduction

Time series forecasting (TSF) is critical for decision-making in domains such as energy (Das et al., 2023; Jin et al., 2024; Wang et al., 2025), healthcare (Qiu et al., 2024), and finance (Feng et al., 2019). The goal is to predict future values based on historical observations and associated contextual features. In practice, accurate forecasting requires not only modeling temporal dependencies in numerical sequences, but also understanding how they interact with external contextual factors such as static attributes or dynamic events (Liu et al., 2024b). Fundamentally, TSF can be viewed as learning a mapping from past values and contextual features to future outcomes (Jiang et al., 2025).

To learn this mapping, researchers have proposed a comprehensive range of methods, ranging from classical statistical models to modern data-driven approaches. Traditional methods, such as ARIMA (Hyndman & Khandakar, 2008) and state-space models (Winters, 1960), rely on strong assumptions about data generation and often incorporate domain-specific priors. In contrast, recent data-driven approaches such as deep learning models aim to learn patterns directly from data without hand-crafted assumptions. Architectures based on RNNs (Lai et al., 2018), CNNs (Cheng et al., 2025b), Transformers (Zhou et al., 2022), and MLPs (Challu et al., 2023) have been widely adopted, each capturing different aspects of temporal dependencies. However, most of these models assume homogeneous numerical inputs and struggle to effectively incorporate complex contextual features, particularly those with heterogeneous modalities.

Beyond capturing temporal dependencies, there is an increasingly growing emphasis in recent research on incorporating contextual features to enhance forecasting performance (Liu et al., 2024a; Williams et al., 2024; Liu et al., 2024b). These features typically fall into two categories: dynamic exogenous variables (e.g., weather conditions, event indicators) and static attributes (e.g., types, patient demographics, market segments). When contextual features share the same numerical modality as the target series, they can be directly modeled as additional input channels. However, many particularly high-value contextual features, such as clinical notes, policy texts, or user logs, are expressed in unstructured textual form. This heterogeneity poses significant challenges for aligning and integrating information across modalities.

> Figure 1 (see PDF p. 2). Methods for representation modeling of time series and contextual features: (a) linear adapter, (b) soft prompt, and (c) symbolic intermediary.

To address these challenges, some studies have explored shallow fusion strategies to incorporate contextual features. Models such as DeepAR (Salinas et al., 2020) and Temporal Fusion Transformer (TFT) (Lim et al., 2021) typically concatenate external variables with time series or introduce gating mechanisms. While offering basic integration, these methods often rely on weak alignment and struggle to capture deep semantic interactions across modalities (Liu et al., 2024e). More recently, LLMs have been introduced into time series forecasting (Sun et al., 2023; Liu et al., 2024c; Ansari et al., 2024). Methods like Time-LLM (Jin et al., 2023) inject time series features into LLMs using linear adapters (Figure 1(a)) or soft prompts (Figure 1(b)). Although promising, these approaches fall short in resolving the structural discrepancies between numerical sequences and unstructured contextual features. Moreover, they fail to fully leverage the generative and reasoning capabilities of LLMs, which are pretrained on large-scale corpora. This observation raises a fundamental question: can time series be effectively modeled in a discrete token space to unlock the full potential of LLMs?

Motivated by this question, we explore a more expressive yet under-explored paradigm that formulates time series forecasting as a multimodal discrete context understanding and generation problem, powered by pre-trained LLMs, as illustrated in Figure 1(c). The key idea is to transform continuous numerical sequences into discrete tokens and embed them into the same semantic space as contextual language inputs. This formulation enables the full use of LLM capabilities in semantic understanding, contextual reasoning, and autoregressive generation. However, this paradigm introduces several non-trivial challenges. First, discretizing dynamic time series is more difficult than compressing static data, as it requires preserving temporal dependencies while reducing granularity. Second, even with symbolic representations, semantic misalignment between temporal tokens and contextual features may hinder effective fusion. Finally, it remains unclear whether time series forecasting can be effectively addressed through autoregressive generation over discrete tokens, a direction still largely unexplored.

Based on the above analysis, we propose TokenCast, an LLM-driven framework for context-aware time series forecasting via symbolic discretization. TokenCast begins with a time series tokenizer that converts continuous sequences into temporal tokens, mitigating structural discrepancies across data modalities. To bridge the semantic gap, temporal and contextual tokens are jointly embedded into a shared representation space using a pre-trained LLM, optimized via an autoregressive objective while keeping the backbone frozen and tuning only the embedding layer. Building on this unified semantic space, the aligned LLM is further fine-tuned with supervised forecasting signals to enhance predictive performance. We evaluate TokenCast on diverse real-world datasets enriched with contextual features. Experimental results show that TokenCast achieves strong accuracy and generalization across domains. We also conduct comprehensive ablation and qualitative studies, offering insights into the flexibility of symbolic, LLM-based time series forecasting.

## 2. Related Work

Time series forecasting (TSF) is a fundamental task across various domains. Traditional approaches typically rely on statistical assumptions such as stationarity and linearity, and often depend on hand-crafted assumptions that limit their flexibility (Holt, 2004; Kalekar et al., 2004). Alternatively, data-driven methods (Chen & Guestrin, 2016), particularly those based on deep learning, have advanced TSF by learning temporal patterns directly from data. RNN-based models (Wang et al., 2019) capture dependencies through recurrence, CNN-based models (Wang et al., 2023) enhance local pattern extraction, and Transformer-based architectures (Shi et al., 2024) are well-suited for modeling long-range interactions. Furthermore, MLP-based approaches (Wang et al., 2024b) demonstrate that simple architectures can achieve competitive performance with improved computational efficiency. These models mainly focus on numerical data, with less emphasis on unstructured context.

In addition to modeling temporal dependencies, recent research increasingly emphasizes the integration of contextual features for accurate forecasting (Chang et al., 2023; Liu et al., 2024d; Hu et al., 2025). Two major lines of research have emerged in this direction. One line of research focuses on deep learning architectures that explicitly model feature interactions (Gasthaus et al., 2019). For example, TimeXer (Wang et al., 2024c) employs cross-attention mechanisms to fuse dynamic and static modalities. Another line of research leverages pre-trained LLMs for multimodal modeling (Cheng et al., 2025a; Liu et al., 2025). Some approaches, such as TEMPO (Cao et al., 2023), utilize linear adapters to project time series features into the LLM's semantic space. Others, like PromptCast (Xue & Salim, 2023), employ soft prompts to guide the frozen LLM's behavior. However, these promising approaches fail to bridge the structural gap between numerical and textual modalities.

## 3. The Proposed TokenCast

In this section, we present the precise formal problem definition, clarify the key concepts and notations used consistently throughout the paper, and provide an overview of the TokenCast.

### 3.1. Problem Formulation

We consider a dataset $D = \{(X_i, T_i, P_i)\}_{i=1}^N$ of $N$ multimodal time series instances. For each instance, $X \in \mathbb{R}^{L \times C}$ represents the multivariate time series over $L$ time steps and $C$ channels, $T$ denotes the contextual features, and $P \in \mathbb{R}^{L_P \times C}$ is the ground-truth future sequence over a horizon $L_P$. The contextual features $T$ are tokenized to tokens $Y$ using the tokenizer of a pre-trained LLM, while the time series $X$ is converted into discrete tokens $Z_q$ via a learnable mapping $f_\theta: X \mapsto Z_q$. These two token sequences are then concatenated to form a token sequence $Z = [Z_q; Y] \in V^{T'}$. We use boundary markers to delimit the temporal tokens of $\hat{Z}$. Finally, a decoding function $g_\phi: \hat{Z} \mapsto \hat{P}$ is applied to reconstruct the raw time series $\hat{P} \in \mathbb{R}^{L_P \times C}$.

### 3.2. Framework Overview

Figure 2 illustrates the overview of the TokenCast, which consists of three main stages. The process begins with the time series tokenizer, which transforms continuous time series into a sequence of discrete tokens via a decoupled and dynamical vector quantization tokenizer. Subsequently, both the temporal and contextual tokens are jointly processed by a pre-trained LLM, which performs cross-modality alignment under autoregressive objectives. Following this alignment, the aligned LLM is adapted to the forecasting task via generative fine-tuning, enabling token prediction. The predicted tokens are decoded to raw time series using a frozen time series de-tokenizer. The following sections elaborate on the principal stages of the TokenCast.

> Figure 2 (see PDF p. 4). Overview of the framework for context-aware time series forecasting: (a) time series tokenizer to address the structural differences between modalities, (b) cross-modality alignment with an autoregressive objective to bridge the modalities, and (c) generative fine-tuning and context-aware forecasting through time series decoding for horizon prediction.

### 3.3. Time Series Discretization

#### 3.3.1. Time Series Tokenizer

To fully harness the generative and reasoning capabilities of language models, symbolic representation naturally arises as an effective intermediary. Accordingly, we employ time series discretization as a simple yet powerful approach to establish this bridge. It is worth noting that existing approaches, such as Symbolic Aggregate Approximation (SAX) (Lin et al., 2007), have achieved progress in time series discretization but often suffer from significant information loss due to dimensionality reduction. In contrast, reconstruction-based methods (Van Den Oord et al., 2016) map subsequences to discrete codes from a predefined codebook and achieve more precise representations through reconstruction optimization. While preserving the original information is advantageous, previous reconstruction-based methods typically encode the entire sequence, overlooking the statistical properties of time series. In the forecasting task, Reversible Instance Normalization (RevIN) (Kim et al., 2021) is widely used, yet its normalization and denormalization steps risk leaking future information. To overcome this limitation, we propose a decoupled and dynamic tokenizer.

As illustrated in Figure 2(a), similar to the forecasting phase, we divide the multivariate time series into a historical time series $H \in \mathbb{R}^{L_H \times C}$ and a predicted time series $P \in \mathbb{R}^{L_P \times C}$, which can be formally represented as $X = [H; P] \in \mathbb{R}^{L \times C}$. The process begins with a reversible instance normalization (RIN) layer. We compute the mean $\mu(H)$ and standard deviation $\sigma(H)$ solely from the historical time series $H$, and apply them to normalize the time series $X$, thereby preventing future information leakage. These statistics are retained for inverse transformation during decoding. Instead of employing separate encoders, we adopt a shared encoder, which facilitates the joint modeling of both local and global information. The normalized time series is then passed through a causal encoder $f_{\text{enc}}$, yielding a sequence of continuous latent representations $Z = f_{\text{enc}}(X) \in \mathbb{R}^{T \times d}$, where $T$ is the number of latent vectors and $d$ is the feature dimension.

To discretize the latent representations, we apply a vector quantization layer. For domain $i$, a learnable codebook $C_i = \{e_{i,k}\}_{k=1}^K \subset \mathbb{R}^d$ is maintained, containing $K$ embedding vectors. Each latent vector $z_t \in \mathbb{R}^d$ is mapped to its nearest neighbor in the codebook as $z_t^q = e_{i,k^*}$, where $k^* = \arg\min_k \|z_t - e_{i,k}\|_2^2$. The output of this layer is a quantized sequence $Z_q = (z_1^q, \ldots, z_T^q)$, and the corresponding sequence of indices $\{k^*\}$ serves as the discrete tokens for downstream modeling. These tokens are subsequently decoded by a shared causal decoder $f_{\text{dec}}$, rather than by separate decoders, which ensures consistent reconstruction and enables the predicted part to dynamically exploit richer historical features. Then, the final reconstruction $\hat{X}$ is obtained by applying the inverse RIN operation using the stored statistics $\mu(H)$ and $\sigma(H)$, i.e., $\hat{X} = f_{\text{denorm}}(f_{\text{dec}}(Z_q))$.

#### 3.3.2. Training Objective

The tokenizer is optimized by minimizing the objective function defined as follows:

$$
L = L_{\text{recon}} + \beta (L_{\text{commit}} + L_{\text{codebook}}) + \gamma L_{\text{diversity}}.
\tag{1}
$$

Here, $L_{\text{recon}} = \|\hat{X} - X\|_2^2$ is the reconstruction loss that optimizes both the encoder and decoder. Due to the non-differentiability of the $\arg\min$ operation in quantization, we employ the straight-through estimator (STE) during backpropagation. To train the vector quantizer, we include $L_{\text{codebook}} = \|\operatorname{sg}[Z] - Z_q\|_2^2$ and $L_{\text{commit}} = \|Z - \operatorname{sg}[Z_q]\|_2^2$, where $\operatorname{sg}[\cdot]$ denotes the stop-gradient operator, which prevents gradients from flowing into its argument during backpropagation. To avoid codebook collapse and promote diverse usage of codebook entries, we further add a diversity loss

$$
L_{\text{diversity}} = \frac{1}{N}\sum_{i=1}^N \frac{1}{d_i + \varepsilon},
\qquad
d_i = \min_{j \ne i} \|e_i - e_j\|_2.
$$

This penalty discourages vectors from clustering too closely and encourages more uniform utilization of the codebook.

### 3.4. Pre-trained LLM Backbone Formulation

Following the discretization of time series into discrete tokens, the next challenge is to model the complex dependencies embedded in these sequences. While architectures like TCNs or Transformers can be trained from scratch, we argue that a pre-trained LLM serves as a more effective backbone. This is supported by two observations: (1) a pre-trained LLM possesses strong semantic understanding and contextual reasoning capabilities acquired from large-scale corpora, and (2) the structure of discrete time series tokens closely resembles that of language tokens (Zhao et al., 2023). By casting forecasting as a generative task, we directly leverage the LLM's autoregressive generation ability. To guide LLM reasoning and incorporate contextual features, we employ a structured prompt template, as shown in Figure 2(b). This prompt template consists of four essential components: domain knowledge, task instructions, statistical properties, and discrete time series tokens. This design ensures token-level consistency with language tokens and introduces task-specific descriptions alongside statistical attributes, enabling the LLM to perform instruction-driven generation.

### 3.5. Cross-Modality Alignment of Time Series and Contextual Features

While discretization aligns time series structurally with language tokens, a semantic gap remains between time series and contextual features. Existing methods often introduce projection modules (e.g., MLPs) to map time series into the LLM's latent space for fusion (Liu et al., 2025). Although effective in downstream tasks, these strategies rely on external transformation modules for alignment, which bypass the language model's native vocabulary modeling mechanism. To this end, we implement a more explicit vocabulary-level alignment strategy. As illustrated in Figure 2(b), we construct a unified vocabulary by directly appending $K$ temporal tokens and $S$ task-specific special tokens to the original vocabulary $V_{\text{orig}}$ of the pre-trained LLM, forming an extended vocabulary $V$. Correspondingly, a shared embedding matrix $E \in \mathbb{R}^{|V| \times d}$ is used to encode all tokens, regardless of their modality origin.

This unified embedding mechanism enables seamless fusion of time series and contextual features while maintaining alignment with the pre-trained model. To ensure distributional alignment with pretrained embeddings for fine-tuning, the embedding of the newly introduced time series tokens is initialized by sampling from a multivariate Gaussian distribution defined by the mean $\mu$ and covariance $\Sigma$ of the original word embeddings. Then, temporal tokens $Z_q$ and contextual tokens $Y$ are concatenated at the token level and jointly transformed into embeddings via the shared embedding layer:

$$
E([Z_q, Y]) = [E(z_1), \ldots, E(z_n), E(y_1), \ldots, E(y_m)],
$$

where $E$ denotes the unified embedding matrix. This unified embedding process enables the LLM to reason over concatenated sequences without requiring architectural modification.

To optimize cross-modality token representations within the shared embedding space, we adopt an autoregressive training objective. Specifically, we freeze all parameters of the pre-trained LLM and update only the shared embedding matrix $E$, which is responsible for encoding both temporal and contextual tokens. Given a concatenated token sequence $[Z_q, Y]$, the training objective is formulated as a next-token prediction task over the combined sequence:

$$
L_{\text{align}} = - \sum_{t=1}^{T} \log p(z_t \mid z_1, \ldots, z_{t-1}; E),
\tag{2}
$$

where $z_t \in V$ denotes the $t$-th token in the sequence, and $p(\cdot)$ is the conditional probability predicted by the frozen language model given the embedding vectors from $E$.

### 3.6. Generative Fine-tuning and Context-Aware Time Series Forecasting

We now detail the procedure for adapting the aligned LLM for forecasting tasks. As illustrated in Figure 2(c), we employ a generative fine-tuning strategy to specialize the model for context-aware time series forecasting. This process consists of two primary stages: (1) structured prompt-based generative fine-tuning; and (2) context-aware time series forecasting with token-based decoding. In the first stage, prompt-based generative fine-tuning is introduced to explicitly transfer the pretrained language modeling capability into the forecasting domain. Instead of relying on external mapping modules, generative fine-tuning directly formulates forecasting as a generation task, where the model is supervised to output both natural language reasoning and sequences of future time series tokens. This paradigm fosters a fast-thinking behavior: by optimizing an autoregressive objective against ground-truth structured responses, the model learns to rapidly recognize patterns, associate contextual features with temporal dynamics, and produce coherent outputs without engaging in deep deliberation. As a result, the aligned LLM acquires the ability to generate fluent and context-aware predictions.

In the second stage, the fine-tuned model is utilized for context-aware forecasting and decoding. During inference, the model receives a prompt with historical data and contextual features, and autoregressively generates a complete response. The key component of this generated output is the sequence of discrete tokens, which represents the model's prediction of future time series values. To translate this symbolic representation back into a continuous predicted time series, these tokens are processed by a frozen time series de-tokenizer. We use boundary markers to delimit the temporal tokens within the generated sequence. This procedure leverages the LLM's reasoning capacity, enabling reliable forecasting grounded in the contextual feature.

## 4. Experiments

In this section, we conduct comprehensive experiments to evaluate our TokenCast's performance on diverse, representative, and challenging real-world datasets enriched with contextual features for time series forecasting. Additionally, we perform ablation studies and exploration analysis.

### 4.1. Experimental Setup

#### 4.1.1. Datasets

As shown in Figure 3, we evaluate our framework on six real-world datasets from diverse domains enriched with contextual features: Economic (McCracken & Ng, 2016), Health (Panagopoulos et al., 2021), Web (Gasthaus et al., 2019), two subsets of Stock data (Feng et al., 2019), and Nature (Poyatos et al., 2020). These datasets, spanning various temporal patterns and contextual dependencies, collectively serve as a comprehensive benchmark for context-aware forecasting. Data preparation involves imputing missing values and applying z-score normalization to all datasets, thereby ensuring stable convergence and fair comparability. A detailed description of the datasets, preprocessing procedures, and additional implementation details is provided in Appendix A for clarity, transparency, and reproducibility.

> Figure 3 (see PDF p. 6). Diverse real-world datasets from various domains and with distinct characteristics.

#### 4.1.2. Baselines

We compare our proposed framework against eight strong baselines, grouped into four representative categories for comprehensive evaluation. For LLM-based models, we include Time-LLM (Jin et al., 2023) and GPT4TS (Zhou et al., 2023), which adapt pre-trained LLMs for time series forecasting using modality-aware prompting and reprogramming. In the self-supervised frameworks category, we evaluate TimeDART (Wang et al., 2024a) and SimMTM (Dong et al., 2023). These unimodal pretraining methods leverage self-supervised objectives to enhance time series representation learning. Additionally, we include Transformer-based methods like Autoformer (Wu et al., 2021) and Crossformer (Zhang & Yan, 2023). Finally, we consider the MLP-based method DLinear (Zeng et al., 2023). Further details are provided in Appendix B.1.

#### 4.1.3. Implementation Details

For each baseline, we search over multiple input lengths and report the best performance to avoid underestimating its capability. The historical length is set to $L = 96$ for the Nature dataset and $L = 36$ for the other five datasets, based on the data volume and temporal resolution. The forecasting horizons are set to $\{24, 48, 96, 192\}$ for Nature and $\{24, 36, 48, 60\}$ for the other datasets. We adopt two widely used evaluation metrics in time series forecasting: mean absolute error (MAE) and mean squared error (MSE). For exploratory analysis, we use 96-to-24 on Nature and 36-to-24 on the other datasets. Complete results for the main experiments, ablation studies, and exploratory analysis are included in Appendix C. All experiments are implemented in PyTorch and conducted on a distributed setup with 8 NVIDIA A100 GPUs.

Table 1. All reported results are averages over four horizons and three trials on various context-rich benchmark datasets. Lower values indicate better performance. The best results are highlighted in bold, and the second-best are underlined in the source PDF.

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">TokenCast</th>
      <th colspan="2">Time-LLM</th>
      <th colspan="2">GPT4TS</th>
      <th colspan="2">TimeDART</th>
      <th colspan="2">SimMTM</th>
      <th colspan="2">Crossformer</th>
      <th colspan="2">Autoformer</th>
      <th colspan="2">DLinear</th>
    </tr>
    <tr>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Economic</td><td>68.911</td><td>1.646</td><td>81.542</td><td>1.760</td><td>85.947</td><td>1.716</td><td>86.029</td><td>1.771</td><td>90.351</td><td>1.672</td><td>406.418</td><td>4.074</td><td>116.745</td><td>2.088</td><td>122.216</td><td>2.069</td></tr>
    <tr><td>Health</td><td>2.524</td><td>0.081</td><td>2.823</td><td>0.104</td><td>2.564</td><td>0.083</td><td>2.623</td><td>0.088</td><td>2.720</td><td>0.088</td><td>1644.997</td><td>2.504</td><td>2.617</td><td>0.266</td><td>28.587</td><td>0.455</td></tr>
    <tr><td>Web</td><td>497.410</td><td>1.246</td><td>557.833</td><td>1.751</td><td>540.492</td><td>1.458</td><td>773.635</td><td>1.369</td><td>847.649</td><td>1.327</td><td>698.316</td><td>1.963</td><td>722.506</td><td>3.303</td><td>632.301</td><td>1.398</td></tr>
    <tr><td>Stock-NY</td><td>0.482</td><td>0.455</td><td>0.662</td><td>0.510</td><td>0.638</td><td>0.502</td><td>0.776</td><td>0.606</td><td>0.613</td><td>0.495</td><td>1.111</td><td>0.912</td><td>0.676</td><td>0.573</td><td>1.004</td><td>0.754</td></tr>
    <tr><td>Stock-NA</td><td>1.134</td><td>0.780</td><td>1.200</td><td>0.925</td><td>1.272</td><td>0.880</td><td>1.409</td><td>0.883</td><td>1.343</td><td>0.834</td><td>1.912</td><td>1.052</td><td>1.496</td><td>0.903</td><td>1.710</td><td>0.958</td></tr>
    <tr><td>Nature</td><td>0.269</td><td>0.297</td><td>0.258</td><td>0.283</td><td>0.274</td><td>0.299</td><td>0.243</td><td>0.276</td><td>0.259</td><td>0.286</td><td>0.733</td><td>0.511</td><td>0.508</td><td>0.486</td><td>0.369</td><td>0.436</td></tr>
    <tr><td>1st Count</td><td>5</td><td>5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
  </tbody>
</table>

### 4.2. Forecasting Performance Analysis

Table 1 comprehensively compares forecasting performance across six benchmark datasets. TokenCast demonstrates superior performance in most scenarios, further confirming previous empirical findings (Zhou et al., 2023) that no single model performs best across all settings. This performance highlights its adaptability across diverse forecasting domains. Notably, LLM-based baselines like Time-LLM also show competitive results, particularly on context-rich datasets such as Economic and Stock-NY. This further validates the potential of leveraging large language models in time series forecasting. However, these models often lack the structural alignment mechanisms introduced by our framework, limiting their consistent performance. Conventional baselines such as TimeDART perform well on datasets with strong periodicity and weak contextual dependence (e.g., Nature), but their performance drops significantly on complex datasets rich in contextual features (e.g., Economic and Web). This contrast underscores the importance of contextual feature modeling and cross-modal interaction. In summary, our framework delivers state-of-the-art results with high consistency. This is attributed to its core design: discretizing time series into discrete tokens and aligning them with contextual features. This unified token-based paradigm effectively captures multimodal dependencies and addresses real-world context-aware time series forecasting challenges.

### 4.3. Ablation Studies

#### 4.3.1. Ablation on Alignment and Fine-tuning

We conduct the ablation study on two crucial training steps: cross-modality alignment and generative fine-tuning. The results in Figure 4 (left) clearly demonstrate their indispensable contribution to the overall framework. The cross-modality alignment stage consistently achieves lower MSE scores across all datasets. Without alignment, contextual features risk being misinterpreted by the time series backbone, leading to suboptimal forecasts. This highlights its role in bridging structural and semantic discrepancies between time series and contextual features, thus facilitating meaningful feature interaction. Meanwhile, the generative fine-tuning stage further enhances performance, with notable improvements on complex datasets such as Stock-NA. These findings emphasize the necessity of both alignment and fine-tuning in enabling reliable forecasting.

#### 4.3.2. Ablation on Multimodal Contributions

Figure 4 (right) examines the impact of multimodal context by selectively removing different types of contextual features. The results demonstrate that both general information (e.g., domain knowledge and task instructions) and local information (e.g., event-specific details) make substantial contributions to forecasting accuracy across datasets. Removing either type consistently degrades performance, with the absence of local information showing particularly severe effects on datasets characterized by dynamic and non-stationary patterns. Meanwhile, excluding textual context leads to the most significant accuracy drop, underscoring the critical role of text in capturing domain knowledge and high-level semantics. These findings highlight the complementary nature of different contextual modalities: while general information provides broad background knowledge, local information introduces fine-grained event-level signals.

> Figure 4 (see PDF p. 8). Ablation studies. (Left) Ablation study on the effects of cross-modality alignment and generative fine-tuning across multiple datasets. (Right) Ablation study on multiple datasets on the contribution of multimodal context in time series forecasting.

Table 2. Study on the number of tokens in the codebook across multiple datasets. We report predicted reconstructed MSE (Recon.), downstream MSE, and downstream MAE.

<table>
  <thead>
    <tr>
      <th rowspan="2">Codebook Size</th>
      <th colspan="3">Economic</th>
      <th colspan="3">Health</th>
      <th colspan="3">Web</th>
      <th colspan="3">Stock-NY</th>
      <th colspan="3">Stock-NA</th>
      <th colspan="3">Nature</th>
    </tr>
    <tr>
      <th>Recon.</th><th>MSE</th><th>MAE</th>
      <th>Recon.</th><th>MSE</th><th>MAE</th>
      <th>Recon.</th><th>MSE</th><th>MAE</th>
      <th>Recon.</th><th>MSE</th><th>MAE</th>
      <th>Recon.</th><th>MSE</th><th>MAE</th>
      <th>Recon.</th><th>MSE</th><th>MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>32</td><td>190.371</td><td>50.234</td><td>1.372</td><td>207.459</td><td>1.772</td><td>0.065</td><td>731.474</td><td>451.827</td><td>1.165</td><td>0.569</td><td>0.325</td><td>0.377</td><td>0.244</td><td>0.794</td><td>0.636</td><td>0.134</td><td>0.233</td><td>0.281</td></tr>
    <tr><td>64</td><td>141.852</td><td>37.699</td><td>1.293</td><td>101.652</td><td>2.714</td><td>0.072</td><td>664.501</td><td>529.401</td><td>1.228</td><td>0.573</td><td>0.339</td><td>0.381</td><td>0.213</td><td>0.690</td><td>0.616</td><td>0.158</td><td>0.241</td><td>0.296</td></tr>
    <tr><td>128</td><td>170.630</td><td>39.379</td><td>1.251</td><td>186.619</td><td>2.622</td><td>0.070</td><td>3924.953</td><td>1743.889</td><td>1.539</td><td>0.518</td><td>0.730</td><td>0.604</td><td>0.205</td><td>0.671</td><td>0.600</td><td>0.104</td><td>0.203</td><td>0.265</td></tr>
    <tr><td>256</td><td>191.937</td><td>39.309</td><td>1.339</td><td>69.035</td><td>2.413</td><td>0.070</td><td>5062.452</td><td>899.202</td><td>1.385</td><td>0.572</td><td>0.384</td><td>0.424</td><td>0.209</td><td>0.646</td><td>0.593</td><td>0.114</td><td>0.248</td><td>0.288</td></tr>
  </tbody>
</table>

### 4.4. Exploration Analysis

#### 4.4.1. Codebook Size

We investigate the effect of codebook size on model performance, as summarized in Table 2. The results show that a moderate size of 128 achieves state-of-the-art performance on challenging datasets such as Nature and Stock-NA, while a smaller size of 64 yields the best results on the Economic dataset. In contrast, both overly small (32) and overly large (256) codebooks degrade performance, indicating that simply increasing token granularity does not necessarily benefit forecasting. Overall, an appropriately balanced codebook size provides a better trade-off between reconstruction fidelity and downstream forecasting accuracy.

Table 3. Performance comparison of different backbone models and their variants (base/instruct) across varying model scales and multiple datasets.

<table>
  <thead>
    <tr>
      <th rowspan="2">Backbone</th>
      <th colspan="2">Economic</th>
      <th colspan="2">Health</th>
      <th colspan="2">Web</th>
      <th colspan="2">Stock-NY</th>
      <th colspan="2">Stock-NA</th>
      <th colspan="2">Nature</th>
    </tr>
    <tr>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Qwen2.5-0.5B-base</td><td>37.164</td><td>1.301</td><td>2.492</td><td>0.068</td><td>586.793</td><td>1.271</td><td>0.297</td><td>0.355</td><td>0.668</td><td>0.605</td><td>0.180</td><td>0.246</td></tr>
    <tr><td>Qwen2.5-0.5B-inst.</td><td>36.744</td><td>1.299</td><td>2.493</td><td>0.068</td><td>586.780</td><td>1.271</td><td>0.353</td><td>0.391</td><td>0.695</td><td>0.614</td><td>0.187</td><td>0.253</td></tr>
    <tr><td>Qwen2.5-1.5B-inst.</td><td>38.549</td><td>1.283</td><td>2.471</td><td>0.069</td><td>589.843</td><td>1.273</td><td>0.329</td><td>0.372</td><td>0.722</td><td>0.611</td><td>0.229</td><td>0.270</td></tr>
    <tr><td>Qwen3-0.6B-inst.</td><td>39.629</td><td>1.315</td><td>2.320</td><td>0.068</td><td>588.379</td><td>1.272</td><td>0.405</td><td>0.417</td><td>0.936</td><td>0.715</td><td>0.236</td><td>0.281</td></tr>
  </tbody>
</table>

#### 4.4.2. LLM Backbone

We evaluate four LLM backbones to identify the optimal architecture for our forecasting framework. As summarized in Table 3, the Qwen2.5-0.5B-base models consistently demonstrate superior performance. Specifically, the base version achieves state-of-the-art results on the Nature and Stock-NA datasets, while the instruct-tuned version excels on the more complex Economic dataset. Interestingly, larger models like Qwen2.5-1.5B-instruct fail to yield further gains and often underperform. This suggests that for our tasks, simply scaling up model size is not beneficial. Instead, the 0.5B models strike a balance between representational capacity and generalization.

Table 4. Study on different initialization methods on the embedding layer. We compare mean initialization, codebook sampling, and random initialization.

<table>
  <thead>
    <tr>
      <th rowspan="2">Initialization</th>
      <th colspan="2">Economic</th>
      <th colspan="2">Health</th>
      <th colspan="2">Web</th>
      <th colspan="2">Stock-NY</th>
      <th colspan="2">Stock-NA</th>
      <th colspan="2">Nature</th>
    </tr>
    <tr>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Mean Initialization</td><td>36.744</td><td>1.299</td><td>2.357</td><td>0.068</td><td>585.064</td><td>1.256</td><td>0.319</td><td>0.371</td><td>0.695</td><td>0.614</td><td>0.187</td><td>0.253</td></tr>
    <tr><td>Codebook Sampling</td><td>39.680</td><td>1.261</td><td>2.574</td><td>0.068</td><td>585.665</td><td>1.265</td><td>0.337</td><td>0.380</td><td>0.667</td><td>0.602</td><td>0.224</td><td>0.264</td></tr>
    <tr><td>Random Initialization</td><td>36.744</td><td>1.299</td><td>2.493</td><td>0.068</td><td>586.780</td><td>1.271</td><td>0.353</td><td>0.391</td><td>1.101</td><td>0.725</td><td>0.189</td><td>0.256</td></tr>
  </tbody>
</table>

#### 4.4.3. Embedding Layer Initialization

We investigate three initialization strategies for our model's embedding layer to identify the most effective approach. As shown in Table 4, mean initialization consistently provides the most reliable performance. Specifically, it achieves the best results on the Nature and Economic datasets. While word initialization is superior on the Stock-NA dataset, its performance is less consistent across other domains. Notably, standard random initialization suffers a significant performance degradation on Stock-NA, highlighting its instability. These findings suggest that initializing embeddings with meaningful prior information provides a better starting point for optimization. Therefore, we adopt mean initialization as the default.

#### 4.4.4. Qualitative Analysis of Tokenization

To evaluate our discretization module, we analyze the Nature dataset from three perspectives, as shown in Figure 5. The token usage heatmap (left) shows that all 64 tokens are activated, mitigating codebook collapse and capturing diverse temporal structures. The codebook clustering visualization (middle) illustrates that tokens form coherent groups in the latent space, indicating that the learned vocabulary preserves structural relationships among temporal patterns. The dynamic reconstruction results (right) highlight the tokenizer's adaptive decoding property: the same token id (e.g., ID = 18) can produce different decoded segments depending on context, ensuring alignment with the original sequences. Overall, these findings confirm that our discretization process learns a diverse, semantically organized vocabulary while supporting context-aware decoding for forecasting.

> Figure 5 (see PDF p. 9). Visualization of tokenizer behavior on the Nature dataset. (Left) Token usage heatmap of the 64-token vocabulary. (Middle) Codebook clustering in the latent space. (Right) Dynamic reconstruction illustrating dynamic decoding.

## 5. Conclusion

We proposed TokenCast, a context-aware time series prediction framework based on a pretrained LLM. This approach first converts a continuous time series into discrete tokens. Leveraging a pretrained LLM, it aligns the temporal and contextual tokens through an autoregressive objective, achieving unified modeling of both modalities. The model is then further fine-tuned to generate future token sequences. We evaluate TokenCast on multiple real-world datasets rich in contextual information. Experimental results demonstrate that TokenCast achieves superior accuracy. We also conduct comprehensive ablation experiments and qualitative analysis to validate the framework's adaptability and flexibility for symbolic, LLM-driven time series prediction. Looking ahead, we believe that leveraging language as a symbolic intermediary has the potential to advance time series prediction toward a multimodal and multi-task level.

## Ethics Statement

This work adheres to the ICLR Code of Ethics. Our study focuses on methodological advances in time series forecasting and does not involve human subjects, personal information, or any sensitive data. All datasets used in our experiments are publicly available and widely adopted in prior research. We strictly follow the respective dataset licenses and provide detailed preprocessing steps in the supplementary material to ensure transparency. The proposed methods are intended for scientific and practical forecasting applications, and we do not anticipate direct harmful impacts. Potential societal risks, such as misuse for decision-making without proper validation, are acknowledged, and we emphasize that results should be interpreted with caution in high-stakes domains.

## Reproducibility Statement

We have taken multiple steps to ensure the reproducibility of our work. The proposed model, training procedures, and evaluation protocols are described in detail in the main text. Additional implementation details, including hyperparameter configurations, are provided in the appendix. All theoretical analyses are accompanied by complete proofs in the supplementary material. For datasets, we clearly describe the preprocessing steps and data split strategies in the supplementary document to facilitate re-implementation. To further support reproducibility, we submit anonymized source code and scripts as supplementary material, enabling independent verification of our results.

## LLM Usage Statement

We used large language models (LLMs) solely as auxiliary tools for improving writing clarity and refining grammar. The LLMs did not contribute to the conception of the research idea, algorithm design, experimental implementation, or analysis. All technical content, experiments, and conclusions were developed by the authors. The authors take full responsibility for the content of this paper.

## References

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. Chronos: Learning the language of time series. *Transactions on Machine Learning Research*, 2024, 2024.

Defu Cao, Furong Jia, Sercan O. Arik, Tomas Pfister, Yixiang Zheng, Wen Ye, and Yan Liu. Tempo: Prompt-based generative pre-trained transformer for time series forecasting. *arXiv preprint arXiv:2310.04948*, 2023.

Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza Ramirez, Max Mergenthaler Canseco, and Artur Dubrawski. N-HiTS: Neural hierarchical interpolation for time series forecasting. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 37, pp. 6989-6997, 2023.

Ching Chang, Wen-Chih Peng, and Tien-Fu Chen. LLM4TS: Two-stage fine-tuning for time-series forecasting with pre-trained LLMs. *CoRR*, 2023.

Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785-794, 2016.

Mingyue Cheng, Xiaoyu Tao, Qi Liu, Hao Zhang, Yiheng Chen, and Defu Lian. Cross-domain pre-training with language models for transferable time series representations. In *Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining*, pp. 175-183, 2025a.

Mingyue Cheng, Jiqian Yang, Tingyue Pan, Qi Liu, Zhi Li, and Shijin Wang. ConvTimeNet: A deep hierarchical fully convolutional model for multivariate time series analysis. In *Companion Proceedings of the ACM on Web Conference 2025*, pp. 171-180, 2025b.

Abhimanyu Das, Weihao Kong, Andrew Leach, Shaan K. Mathur, Rajat Sen, and Rose Yu. Long-term forecasting with TiDE: Time-series dense encoder. *Transactions on Machine Learning Research*, 2023.

Jiaxiang Dong, Haixu Wu, Haoran Zhang, Li Zhang, Jianmin Wang, and Mingsheng Long. SimMTM: A simple pre-training framework for masked time-series modeling. *Advances in Neural Information Processing Systems*, 36:29996-30025, 2023.

Fuli Feng, Xiangnan He, Xiang Wang, Cheng Luo, Yiqun Liu, and Tat-Seng Chua. Temporal relational ranking for stock prediction. *ACM Transactions on Information Systems (TOIS)*, 37(2):1-30, 2019.

Jan Gasthaus, Konstantinos Benidis, Yuyang Wang, Syama Sundar Rangapuram, David Salinas, Valentin Flunkert, and Tim Januschowski. Probabilistic forecasting with spline quantile function RNNs. In *The 22nd International Conference on Artificial Intelligence and Statistics*, pp. 1901-1910. PMLR, 2019.

Charles C. Holt. Forecasting seasonals and trends by exponentially weighted moving averages. *International Journal of Forecasting*, 20(1):5-10, 2004.

Yuxiao Hu, Qian Li, Dongxiao Zhang, Jinyue Yan, and Yuntian Chen. Context-alignment: Activating and enhancing LLM capabilities in time series. *arXiv preprint arXiv:2501.03747*, 2025.

Rob J. Hyndman and Yeasmin Khandakar. Automatic time series forecasting: The forecast package for R. *Journal of Statistical Software*, 27:1-22, 2008.

Yushan Jiang, Kanghui Ning, Zijie Pan, Xuyang Shen, Jingchao Ni, Wenchao Yu, Anderson Schneider, Haifeng Chen, Yuriy Nevmyvaka, and Dongjin Song. Multi-modal time series analysis: A tutorial and survey. *arXiv preprint arXiv:2503.13709*, 2025.

Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, et al. Time-LLM: Time series forecasting by reprogramming large language models. *arXiv preprint arXiv:2310.01728*, 2023.

Ming Jin, Huan Yee Koh, Qingsong Wen, Daniele Zambon, Cesare Alippi, Geoffrey I. Webb, Irwin King, and Shirui Pan. A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024.

Prajakta S. Kalekar et al. Time series forecasting using Holt-Winters exponential smoothing. *Kanwal Rekhi School of Information Technology*, 4329008(13):1-13, 2004.

Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. In *International Conference on Learning Representations*, 2021.

Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long- and short-term temporal patterns with deep neural networks. In *The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval*, pp. 95-104, 2018.

Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister. Temporal fusion transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4):1748-1764, 2021.

Jessica Lin, Eamonn Keogh, Li Wei, and Stefano Lonardi. Experiencing SAX: A novel symbolic representation of time series. *Data Mining and Knowledge Discovery*, 15(2):107-144, 2007.

Haoxin Liu, Shangqing Xu, Zhiyuan Zhao, Lingkai Kong, Harshavardhan Prabhakar Kamarthi, Aditya Sasanur, Megha Sharma, Jiaming Cui, Qingsong Wen, Chao Zhang, et al. Time-MMD: Multi-domain multimodal dataset for time series analysis. *Advances in Neural Information Processing Systems*, 37:77888-77933, 2024a.

Haoxin Liu, Zhiyuan Zhao, Jindong Wang, Harshavardhan Kamarthi, and B. Aditya Prakash. LST-prompt: Large language models as zero-shot time series forecasters by long-short-term prompting. *arXiv preprint arXiv:2402.16132*, 2024b.

Peiyuan Liu, Hang Guo, Tao Dai, Naiqi Li, Jigang Bao, Xudong Ren, Yong Jiang, and Shu-Tao Xia. CALF: Aligning LLMs for time series forecasting via cross-modal fine-tuning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 39, pp. 18915-18923, 2025.

Xu Liu, Junfeng Hu, Yuan Li, Shizhe Diao, Yuxuan Liang, Bryan Hooi, and Roger Zimmermann. UniTime: A language-empowered unified model for cross-domain time series forecasting. In *Proceedings of the ACM Web Conference 2024*, pp. 4095-4106, 2024c.

Yong Liu, Guo Qin, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. AutoTimes: Autoregressive time series forecasters via large language models. *Advances in Neural Information Processing Systems*, 37:122154-122184, 2024d.

Yong Liu, Guo Qin, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Timer-XL: Long-context transformers for unified time series forecasting. *arXiv preprint arXiv:2410.04803*, 2024e.

Michael W. McCracken and Serena Ng. FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4):574-589, 2016.

George Panagopoulos, Giannis Nikolentzos, and Michalis Vazirgiannis. Transfer graph neural networks for pandemic forecasting. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 35, pp. 4838-4845, 2021.

Rafael Poyatos, Victor Granda, Victor Flo, Mark A. Adams, Balazs Adorjan, David Aguade, Marcos P. M. Aidar, Scott Allen, M. Susana Alvarado-Barrientos, Kristina J. Anderson-Teixeira, et al. Global transpiration data from sap flow measurements: The Sapfluxnet database. *Earth System Science Data Discussions*, 2020:1-57, 2020.

Xiangfei Qiu, Jilin Hu, Lekui Zhou, Xingjian Wu, Junyang Du, Buang Zhang, Chenjuan Guo, Aoying Zhou, Christian S. Jensen, Zhenli Sheng, et al. TFB: Towards comprehensive and fair benchmarking of time series forecasting methods. *arXiv preprint arXiv:2403.20150*, 2024.

David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3):1181-1191, 2020.

Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, and Ming Jin. Time-MoE: Billion-scale time series foundation models with mixture of experts. *arXiv preprint arXiv:2409.16040*, 2024.

Chenxi Sun, Hongyan Li, Yaliang Li, and Shenda Hong. TEST: Text prototype aligned embedding to activate LLM ability for time series. *arXiv preprint arXiv:2308.08241*, 2023.

Aaron Van Den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu, et al. WaveNet: A generative model for raw audio. *arXiv preprint arXiv:1609.03499*, 12:1, 2016.

Daoyu Wang, Mingyue Cheng, Zhiding Liu, Qi Liu, and Enhong Chen. TimeDART: A diffusion autoregressive transformer for self-supervised time series representation. *arXiv preprint arXiv:2410.05711*, 2024a.

Huiqiang Wang, Jian Peng, Feihu Huang, Jince Wang, Junhui Chen, and Yifei Xiao. MICN: Multi-scale local and global context modeling for long-term series forecasting. In *The Eleventh International Conference on Learning Representations*, 2023.

Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma, James Y. Zhang, and Jun Zhou. TimeMixer: Decomposable multiscale mixing for time series forecasting. *arXiv preprint arXiv:2405.14616*, 2024b.

Shiyu Wang, Jiawei Li, Xiaoming Shi, Zhou Ye, Baichuan Mo, Wenze Lin, Shengtong Ju, Zhixuan Chu, and Ming Jin. TimeMixer++: A general time series pattern machine for universal predictive analysis. In *The Thirteenth International Conference on Learning Representations (ICLR)*, 2025.

Yuxuan Wang, Haixu Wu, Jiaxiang Dong, Guo Qin, Haoran Zhang, Yong Liu, Yunzhong Qiu, Jianmin Wang, and Mingsheng Long. TimeXer: Empowering transformers for time series forecasting with exogenous variables. *Advances in Neural Information Processing Systems*, 37:469-498, 2024c.

Yuyang Wang, Alex Smola, Danielle Maddix, Jan Gasthaus, Dean Foster, and Tim Januschowski. Deep factors for forecasting. In *International Conference on Machine Learning*, pp. 6607-6617. PMLR, 2019.

Andrew Robert Williams, Arjun Ashok, Etienne Marcotte, Valentina Zantedeschi, Jithendaraa Subramanian, Roland Riachi, James Requeima, Alexandre Lacoste, Irina Rish, Nicolas Chapados, et al. Context is key: A benchmark for forecasting with essential textual information. *arXiv preprint arXiv:2410.18959*, 2024.

Peter R. Winters. Forecasting sales by exponentially weighted moving averages. *Management Science*, 6(3):324-342, 1960.

Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. *Advances in Neural Information Processing Systems*, 34:22419-22430, 2021.

Hao Xue and Flora D. Salim. PromptCast: A new prompt-based learning paradigm for time series forecasting. *IEEE Transactions on Knowledge and Data Engineering*, 36(11):6851-6864, 2023.

Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 37, pp. 11121-11128, 2023.

Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In *The Eleventh International Conference on Learning Representations*, 2023.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. *arXiv preprint arXiv:2303.18223*, 1(2), 2023.

Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In *International Conference on Machine Learning*, pp. 27268-27286. PMLR, 2022.

Tian Zhou, Peisong Niu, Liang Sun, Rong Jin, et al. One fits all: Power general time series analysis by pretrained LM. *Advances in Neural Information Processing Systems*, 36:43322-43355, 2023.

## Appendix A. Datasets Descriptions

In this study, we utilize six diverse real-world datasets enriched with contextual features spanning various domains, including economics, health, web, stock markets, and natural sciences. Each dataset exhibits unique temporal characteristics and varying degrees of contextual dependency, offering a comprehensive benchmark.

- **Economic (FRED-MD):** A monthly macroeconomic dataset consisting of 107 indicators across sectors such as production, labor, and inflation. It supports empirical studies requiring rich contextual interpretation.
- **Health (Covid-19):** Released by Facebook's "Data for Good" initiative, this dataset tracks human mobility patterns across regions during the COVID-19 pandemic, offering policy-driven contextual signals.
- **Stock-NY (NYSE):** Similar in structure and period to NASDAQ, this dataset provides daily time series from the New York Stock Exchange, facilitating comparative financial forecasting studies.
- **Stock-NA (NASDAQ):** A daily stock dataset collected from the NASDAQ exchange between 2013 and 2017, containing representative securities with dynamics heavily influenced by external news and events.
- **Web (Wike2000):** A high-dimensional dataset recording daily page views of 9,013 Wikipedia articles. We select the top 2,000 pages to capture volatile, event-driven user behavior shaped by external textual contexts.
- **Nature (CzeLan):** A 30-minute resolution dataset capturing natural environmental signals with strong periodic patterns and low contextual dependence. It serves as a representative benchmark for low-context forecasting.

## Appendix B. Additional Implementation Details

In this appendix, we provide comprehensive descriptions of the baseline methods used for comparison in the main paper. We also detail the additional configuration parameters and training setups specific to our proposed model to ensure full reproducibility and transparency.

### B.1. Compared Baselines

We first provide a detailed overview of the baseline models employed for comparative analysis in the main manuscript. These models are grouped into four distinct categories, each reflecting a key methodological paradigm in contemporary time series forecasting: LLM-based approaches, self-supervised frameworks, Transformer-based architectures, and a straightforward yet effective linear model. Below, we present concise descriptions of each model, emphasizing its core techniques and underlying conceptual foundations.

- **Time-LLM:** This is a reprogramming framework that transforms time series into text-based representations for input into a frozen large language model (LLM), guided by a Prompt-as-Prefix mechanism to enable reasoning and achieve general-purpose time series forecasting.
- **GPT4TS:** This work proposes the Frozen Pretrained Transformer (FPT), a framework that repurposes language or vision transformers for general time series analysis by freezing their core layers and fine-tuning only task-specific components, leveraging large-scale pretraining without requiring extensive time series data.
- **TimeDART:** This self-supervised pre-training framework addresses the challenge of modeling long-term dynamics and local patterns by combining Transformer encoding with a denoising diffusion process, yielding more transferable representations for downstream tasks.
- **SimMTM:** This is a masked time series pre-training framework that addresses the challenge of disrupted temporal semantics by reconstructing masked points through weighted aggregation from multiple complementary series, preserving temporal variations and learning manifold structures for improved downstream performance.
- **Autoformer:** This addresses the challenge of long-term time series forecasting by introducing a novel decomposition-based architecture with an Auto-Correlation mechanism, which replaces traditional self-attention to capture periodic dependencies and progressively model complex temporal patterns.
- **Crossformer:** This addresses the challenge of capturing temporal and inter-variable dependencies in multivariate time series forecasting using a Dimension-Segment-Wise embedding and Two-Stage Attention within a hierarchical encoder-decoder architecture.
- **DLinear:** This work challenges the effectiveness of complex Transformer-based models for long-term time series forecasting by demonstrating that a simple one-layer linear model can outperform them, highlighting limitations of self-attention in capturing temporal order and calling for renewed exploration of alternative approaches.

### B.2. Model Configurations

Next, we present the implementation details of our TokenCast framework, with a special focus on its three core stages: (a) time series discretization, (b) cross-modality alignment, and (c) generative fine-tuning. We design a specialized time series tokenizer to bridge structural differences across modalities. It consists of a causal TCN encoder that extracts contextualized embeddings and a causal Transformer decoder that reconstructs the original sequence. The embeddings are quantized into discrete tokens, producing compact and informative representations. Specifically, the encoder comprises 3 layers for effective feature extraction, with an embedding size of 64 and a uniform patch size of 4. The second stage aligns time series data with a pre-trained LLM by expanding its vocabulary to include time series tokens and introducing a unified projection layer for shared semantic space. The model is trained with an autoregressive objective using contextual features and historical tokens. Key hyperparameters, such as a learning rate of $5 \times 10^{-5}$ and batch size of 16, are carefully tuned to ensure stable alignment. For the final forecasting task, we utilize the aligned LLM in a generative manner. The model takes historical time series and relevant context as input to predict the sequence of future tokens. These generated tokens are then passed to the time series de-tokenizer to be converted back into a continuous predicted time series. For the optimization settings in this phase, we employ the Adam optimizer with a fine-tuning learning rate set to $1 \times 10^{-5}$. All parameters of the aligned model are updated during the fine-tuning process to adapt its generative capabilities specifically for multi-step horizon prediction, while retaining the same architectural configuration as the alignment phase.

## Appendix C. Full Results

Due to space limitations, the complete results of all experiments are provided in the appendix. The main experimental outcomes are summarized in Table 5, while the ablation studies on alignment and fine-tuning, as well as on multimodal contributions, are reported in Tables 2 and 3, respectively.

### C.1. Forecasting Performance Analysis

Table 5 provides a comprehensive performance comparison across six benchmark datasets, evaluating models on both MSE and MAE metrics. Our model, TokenCast, demonstrates state-of-the-art performance, securing 17 first-place finishes and establishing itself as a top-tier method alongside the leading baseline. This aligns with findings that no single model universally excels, yet it highlights the advantages of our approach. Notably, other LLM-based baselines like Time-LLM and GPT4TS also deliver competitive results, which further validates the potential of leveraging large language models for time series forecasting. However, the performance of these models often varies significantly by dataset. For instance, while Time-LLM is highly effective on the Economic dataset, TokenCast shows a clear advantage on the Stock-NA benchmark, consistently outperforming all other models across nearly all forecasting horizons. This variability suggests that while powerful, generic LLM baselines may lack the specialized architecture needed to explicitly ground and adapt to diverse time-series dynamics. In stark contrast, earlier architectures like Crossformer and Autoformer consistently underperform, particularly on complex, non-stationary datasets such as Web and Economic. Their limitations are evident in the quantitative results; for example, on the Economic dataset, the average MSE for Crossformer (423.001) and Autoformer (174.605) is substantially higher than that of TokenCast (68.911). This large performance gap underscores the difficulty their feature interaction mechanisms face in capturing intricate time-series patterns. In summary, TokenCast achieves not only state-of-the-art but also highly consistent results across a wide range of scenarios. We attribute this success to its core design: discretizing the time series into a unified token-based paradigm. By modeling time-series forecasting as a sequence-to-sequence task in this discrete space, TokenCast effectively captures the intricate dependencies and dynamics that challenge other methods, proving its reliability and effectiveness across diverse forecasting scenarios.

Table 5. All reported results are the average of three trials on various context-rich benchmark datasets. Lower values indicate better performance. Best results are in bold and second-best results are underlined in the source PDF.

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th rowspan="2">Horizon</th>
      <th colspan="2">TokenCast</th>
      <th colspan="2">Time-LLM</th>
      <th colspan="2">GPT4TS</th>
      <th colspan="2">TimeDART</th>
      <th colspan="2">SimMTM</th>
      <th colspan="2">Crossformer</th>
      <th colspan="2">Autoformer</th>
      <th colspan="2">DLinear</th>
    </tr>
    <tr>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="5">Economic</td><td>24</td><td>38.946</td><td>1.188</td><td>38.235</td><td>1.379</td><td>40.546</td><td>1.289</td><td>38.172</td><td>1.392</td><td>42.856</td><td>1.198</td><td>386.326</td><td>3.892</td><td>66.721</td><td>1.714</td><td>69.693</td><td>1.621</td></tr>
    <tr><td>36</td><td>56.116</td><td>1.488</td><td>64.829</td><td>1.626</td><td>67.560</td><td>1.569</td><td>66.131</td><td>1.632</td><td>70.291</td><td>1.511</td><td>401.198</td><td>4.036</td><td>95.088</td><td>1.953</td><td>99.743</td><td>1.905</td></tr>
    <tr><td>48</td><td>77.678</td><td>1.767</td><td>92.481</td><td>1.882</td><td>97.990</td><td>1.851</td><td>100.061</td><td>1.861</td><td>103.499</td><td>1.820</td><td>415.148</td><td>4.156</td><td>130.566</td><td>2.195</td><td>137.429</td><td>2.220</td></tr>
    <tr><td>60</td><td>102.904</td><td>2.140</td><td>130.623</td><td>2.153</td><td>137.690</td><td>2.156</td><td>139.751</td><td>2.199</td><td>144.757</td><td>2.159</td><td>423.001</td><td>4.212</td><td>174.605</td><td>2.489</td><td>181.999</td><td>2.531</td></tr>
    <tr><td>Avg</td><td>68.911</td><td>1.646</td><td>81.542</td><td>1.760</td><td>85.947</td><td>1.716</td><td>86.029</td><td>1.771</td><td>90.351</td><td>1.672</td><td>406.418</td><td>4.074</td><td>116.745</td><td>2.088</td><td>122.216</td><td>2.069</td></tr>

    <tr><td rowspan="5">Health</td><td>24</td><td>1.699</td><td>0.065</td><td>2.322</td><td>0.081</td><td>1.961</td><td>0.063</td><td>2.030</td><td>0.084</td><td>1.828</td><td>0.069</td><td>1644.541</td><td>2.363</td><td>2.097</td><td>0.276</td><td>24.717</td><td>0.426</td></tr>
    <tr><td>36</td><td>2.228</td><td>0.073</td><td>2.702</td><td>0.103</td><td>2.384</td><td>0.080</td><td>2.405</td><td>0.094</td><td>2.400</td><td>0.083</td><td>1645.156</td><td>2.536</td><td>2.407</td><td>0.256</td><td>29.370</td><td>0.463</td></tr>
    <tr><td>48</td><td>2.607</td><td>0.085</td><td>2.916</td><td>0.114</td><td>2.710</td><td>0.091</td><td>2.747</td><td>0.083</td><td>2.980</td><td>0.094</td><td>1645.237</td><td>2.588</td><td>2.770</td><td>0.253</td><td>16.203</td><td>0.381</td></tr>
    <tr><td>60</td><td>3.563</td><td>0.102</td><td>3.353</td><td>0.118</td><td>3.201</td><td>0.096</td><td>3.309</td><td>0.092</td><td>3.672</td><td>0.107</td><td>1645.053</td><td>2.529</td><td>3.193</td><td>0.277</td><td>44.057</td><td>0.551</td></tr>
    <tr><td>Avg</td><td>2.524</td><td>0.081</td><td>2.823</td><td>0.104</td><td>2.564</td><td>0.083</td><td>2.623</td><td>0.088</td><td>2.720</td><td>0.088</td><td>1644.997</td><td>2.504</td><td>2.617</td><td>0.266</td><td>28.587</td><td>0.455</td></tr>

    <tr><td rowspan="5">Web</td><td>24</td><td>453.609</td><td>1.169</td><td>524.512</td><td>1.699</td><td>510.602</td><td>1.345</td><td>695.235</td><td>1.264</td><td>1794.692</td><td>1.638</td><td>644.061</td><td>1.881</td><td>671.749</td><td>3.912</td><td>561.505</td><td>1.277</td></tr>
    <tr><td>36</td><td>480.019</td><td>1.204</td><td>542.815</td><td>1.720</td><td>525.713</td><td>1.441</td><td>780.234</td><td>1.339</td><td>508.610</td><td>1.162</td><td>697.898</td><td>1.930</td><td>706.661</td><td>3.302</td><td>630.056</td><td>1.362</td></tr>
    <tr><td>48</td><td>518.164</td><td>1.281</td><td>571.482</td><td>1.783</td><td>551.430</td><td>1.504</td><td>779.730</td><td>1.398</td><td>531.377</td><td>1.224</td><td>719.304</td><td>2.002</td><td>736.918</td><td>3.076</td><td>649.944</td><td>1.439</td></tr>
    <tr><td>60</td><td>537.828</td><td>1.330</td><td>592.523</td><td>1.803</td><td>574.221</td><td>1.543</td><td>839.340</td><td>1.474</td><td>555.918</td><td>1.283</td><td>732.002</td><td>2.040</td><td>774.696</td><td>2.921</td><td>687.700</td><td>1.514</td></tr>
    <tr><td>Avg</td><td>497.410</td><td>1.246</td><td>557.833</td><td>1.751</td><td>540.492</td><td>1.458</td><td>773.635</td><td>1.369</td><td>847.649</td><td>1.327</td><td>698.316</td><td>1.963</td><td>722.506</td><td>3.303</td><td>632.301</td><td>1.398</td></tr>

    <tr><td rowspan="5">Stock-NY</td><td>24</td><td>0.289</td><td>0.350</td><td>0.351</td><td>0.383</td><td>0.342</td><td>0.380</td><td>0.499</td><td>0.477</td><td>0.332</td><td>0.377</td><td>1.100</td><td>0.909</td><td>0.427</td><td>0.466</td><td>0.600</td><td>0.573</td></tr>
    <tr><td>36</td><td>0.372</td><td>0.401</td><td>0.403</td><td>0.412</td><td>0.401</td><td>0.409</td><td>0.661</td><td>0.551</td><td>0.398</td><td>0.406</td><td>1.032</td><td>0.886</td><td>0.526</td><td>0.508</td><td>0.856</td><td>0.702</td></tr>
    <tr><td>48</td><td>0.538</td><td>0.479</td><td>0.599</td><td>0.569</td><td>0.576</td><td>0.523</td><td>0.852</td><td>0.643</td><td>0.552</td><td>0.477</td><td>1.061</td><td>0.889</td><td>0.757</td><td>0.605</td><td>1.112</td><td>0.802</td></tr>
    <tr><td>60</td><td>0.727</td><td>0.588</td><td>1.293</td><td>0.674</td><td>1.232</td><td>0.697</td><td>1.092</td><td>0.753</td><td>1.170</td><td>0.719</td><td>1.251</td><td>0.962</td><td>0.994</td><td>0.714</td><td>1.447</td><td>0.938</td></tr>
    <tr><td>Avg</td><td>0.482</td><td>0.455</td><td>0.662</td><td>0.510</td><td>0.638</td><td>0.502</td><td>0.776</td><td>0.606</td><td>0.613</td><td>0.495</td><td>1.111</td><td>0.912</td><td>0.676</td><td>0.573</td><td>1.004</td><td>0.754</td></tr>

    <tr><td rowspan="5">Stock-NA</td><td>24</td><td>0.661</td><td>0.600</td><td>0.725</td><td>0.700</td><td>0.796</td><td>0.689</td><td>0.796</td><td>0.711</td><td>0.867</td><td>0.677</td><td>1.773</td><td>0.996</td><td>1.171</td><td>0.811</td><td>1.498</td><td>0.920</td></tr>
    <tr><td>36</td><td>0.887</td><td>0.694</td><td>0.962</td><td>0.828</td><td>1.122</td><td>0.828</td><td>0.983</td><td>0.737</td><td>1.281</td><td>0.827</td><td>1.921</td><td>1.046</td><td>1.494</td><td>0.912</td><td>1.755</td><td>0.974</td></tr>
    <tr><td>48</td><td>1.473</td><td>0.886</td><td>1.385</td><td>1.004</td><td>1.481</td><td>0.956</td><td>1.822</td><td>1.028</td><td>1.577</td><td>0.907</td><td>1.975</td><td>1.072</td><td>1.712</td><td>0.958</td><td>1.852</td><td>0.982</td></tr>
    <tr><td>60</td><td>1.515</td><td>0.941</td><td>1.729</td><td>1.166</td><td>1.688</td><td>1.045</td><td>2.036</td><td>1.055</td><td>1.646</td><td>0.924</td><td>1.980</td><td>1.095</td><td>1.608</td><td>0.930</td><td>1.733</td><td>0.955</td></tr>
    <tr><td>Avg</td><td>1.134</td><td>0.780</td><td>1.200</td><td>0.925</td><td>1.272</td><td>0.880</td><td>1.409</td><td>0.883</td><td>1.343</td><td>0.834</td><td>1.912</td><td>1.052</td><td>1.496</td><td>0.903</td><td>1.710</td><td>0.958</td></tr>

    <tr><td rowspan="5">Nature</td><td>24</td><td>0.179</td><td>0.246</td><td>0.212</td><td>0.254</td><td>0.226</td><td>0.270</td><td>0.170</td><td>0.245</td><td>0.205</td><td>0.255</td><td>0.623</td><td>0.453</td><td>0.325</td><td>0.383</td><td>0.288</td><td>0.396</td></tr>
    <tr><td>48</td><td>0.252</td><td>0.293</td><td>0.251</td><td>0.281</td><td>0.271</td><td>0.303</td><td>0.261</td><td>0.271</td><td>0.262</td><td>0.295</td><td>0.724</td><td>0.513</td><td>0.556</td><td>0.509</td><td>0.364</td><td>0.448</td></tr>
    <tr><td>96</td><td>0.291</td><td>0.307</td><td>0.275</td><td>0.293</td><td>0.286</td><td>0.305</td><td>0.254</td><td>0.274</td><td>0.268</td><td>0.288</td><td>0.734</td><td>0.516</td><td>0.502</td><td>0.487</td><td>0.354</td><td>0.423</td></tr>
    <tr><td>192</td><td>0.355</td><td>0.340</td><td>0.295</td><td>0.302</td><td>0.311</td><td>0.318</td><td>0.286</td><td>0.313</td><td>0.301</td><td>0.304</td><td>0.849</td><td>0.562</td><td>0.648</td><td>0.564</td><td>0.470</td><td>0.475</td></tr>
    <tr><td>Avg</td><td>0.269</td><td>0.297</td><td>0.258</td><td>0.283</td><td>0.274</td><td>0.299</td><td>0.243</td><td>0.276</td><td>0.259</td><td>0.286</td><td>0.733</td><td>0.511</td><td>0.508</td><td>0.486</td><td>0.369</td><td>0.436</td></tr>

    <tr><td>1st Count</td><td></td><td>17</td><td>11</td><td>2</td><td>1</td><td>0</td><td>1</td><td>4</td><td>5</td><td>0</td><td>5</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr>
  </tbody>
</table>

### C.2. Alignment and Fine-tuning

We conduct the ablation study on two crucial training steps: the cross-modality alignment and generative fine-tuning. The comprehensive results in Table 6 clearly demonstrate their indispensable contribution to the overall framework performance. The model equipped with the cross-modality alignment stage consistently achieves lower MSE scores across all six datasets. Without this alignment, contextual features risk being misinterpreted by the time series backbone, leading to suboptimal forecasts. This highlights its critical role in effectively integrating contextual information by bridging structural and semantic discrepancies between time series and contextual features, thus facilitating meaningful feature interaction. This alignment thus acts as a foundational step, ensuring the subsequent fine-tuning stage operates on a semantically rich and coherent feature space.

Concurrently, Table 6 vividly illustrates the pivotal contribution of the generative fine-tuning stage. Across all six benchmark datasets, the model employing generative fine-tuning consistently and substantially outperforms its counterpart that omits this crucial step. The performance degradation when omitting this stage is notable across various datasets, underscoring the general applicability and importance of the fine-tuning process. This drop is particularly stark on datasets like Stock-NA, where the complex, non-stationary patterns demand task-specific adaptation. Ultimately, these findings emphasize that generative fine-tuning is essential for adapting the pre-trained LLM's general capabilities to generative time series forecasting.

Table 6. Ablation study on the significant effects of cross-modality alignment and generative fine-tuning across multiple diverse datasets.

<table>
  <thead>
    <tr>
      <th rowspan="2">Setting</th>
      <th colspan="2">Economic</th>
      <th colspan="2">Health</th>
      <th colspan="2">Web</th>
      <th colspan="2">Stock-NY</th>
      <th colspan="2">Stock-NA</th>
      <th colspan="2">Nature</th>
    </tr>
    <tr>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>w/ Cross-modality Alignment, w/o Generative Fine-tuning</td><td>406.418</td><td>4.074</td><td>2.875</td><td>0.084</td><td>555.375</td><td>1.447</td><td>0.556</td><td>0.479</td><td>1.317</td><td>0.813</td><td>0.378</td><td>0.357</td></tr>
    <tr><td>w/o Cross-modality Alignment, w/ Generative Fine-Tuning</td><td>72.292</td><td>1.690</td><td>2.783</td><td>0.079</td><td>504.740</td><td>1.264</td><td>0.515</td><td>0.478</td><td>1.181</td><td>0.804</td><td>0.305</td><td>0.318</td></tr>
    <tr><td>w/ Cross-modality Alignment, w/ Generative Fine-tuning</td><td>68.961</td><td>1.695</td><td>2.524</td><td>0.081</td><td>497.410</td><td>1.246</td><td>0.482</td><td>0.455</td><td>1.134</td><td>0.780</td><td>0.269</td><td>0.297</td></tr>
  </tbody>
</table>

### C.3. Multimodal Contribution

Table 7 presents the ablation study on the contribution of different multimodal contextual features, including general information, local information, and text. The results show that removing any type of contextual feature consistently degrades forecasting accuracy across all datasets, confirming that these sources of context play complementary roles in enhancing representation quality. In particular, discarding textual information (`w/o Text`) causes the most significant performance drop, especially on datasets with complex and non-stationary patterns such as Health and Nature, where domain knowledge and event-related semantics are crucial for interpreting abrupt changes and long-term shifts. This demonstrates the irreplaceable role of textual signals in providing external knowledge that cannot be inferred solely from numerical series. Meanwhile, removing general or local information also produces notable accuracy loss, which highlights their importance for aligning forecasts with contextual background (e.g., seasonal profiles, regional variations, or localized dynamics). These findings suggest that different types of contextual features contribute complementary perspectives: text offers semantic depth, general information provides global guidance, and local information ensures fine-grained adaptability.

Overall, TokenCast (Ours) achieves the best performance across all datasets, validating that the joint incorporation of textual, general, and local contextual features is essential for effective multimodal time series forecasting. This ablation analysis further emphasizes that weakening any single modality reduces the model's ability to capture the full spectrum of temporal dependencies, while integrating all contextual features leads to the most reliable and accurate forecasts.

Table 7. Ablation study on the contribution of multimodal context (general info, local info, and text) across multiple diverse datasets.

<table>
  <thead>
    <tr>
      <th rowspan="2">Setting</th>
      <th colspan="2">Economic</th>
      <th colspan="2">Health</th>
      <th colspan="2">Web</th>
      <th colspan="2">Stock-NY</th>
      <th colspan="2">Stock-NA</th>
      <th colspan="2">Nature</th>
    </tr>
    <tr>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
      <th>MSE</th><th>MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>w/o Text</td><td>80.000</td><td>1.820</td><td>2.950</td><td>0.089</td><td>530.000</td><td>1.320</td><td>0.550</td><td>0.490</td><td>1.300</td><td>0.820</td><td>0.312</td><td>0.325</td></tr>
    <tr><td>w/o General Info</td><td>72.500</td><td>1.740</td><td>2.800</td><td>0.085</td><td>510.000</td><td>1.280</td><td>0.530</td><td>0.470</td><td>1.200</td><td>0.795</td><td>0.285</td><td>0.310</td></tr>
    <tr><td>w/o Local Info</td><td>70.200</td><td>1.710</td><td>2.600</td><td>0.083</td><td>503.500</td><td>1.260</td><td>0.505</td><td>0.460</td><td>1.180</td><td>0.790</td><td>0.275</td><td>0.305</td></tr>
    <tr><td>TokenCast (Ours)</td><td>68.961</td><td>1.695</td><td>2.524</td><td>0.081</td><td>497.410</td><td>1.246</td><td>0.482</td><td>0.455</td><td>1.134</td><td>0.780</td><td>0.269</td><td>0.297</td></tr>
  </tbody>
</table>

> Figure 6 (see PDF p. 17). Forecasting with uncertainty on Stock-NY (left) and Economic (right) datasets. The plots compare the ground truth trajectories with the model mean predictions, along with the 50% and 80% predictive intervals.

### C.4. Generative Uncertainty

As shown in Figure 6, we evaluate the LLM's predictive uncertainty by performing multiple stochastic runs on the same input. The resulting forecasts, although varied across runs, form a coherent ensemble that consistently encompasses the ground truth. This behavior highlights the model's ability to represent meaningful uncertainty without deviating significantly from the actual data dynamics. Moreover, the forecasts' stability across different stochastic runs further underscores the predictive stability of our model, demonstrating that it can reliably capture uncertainty while maintaining high fidelity to the underlying data. This consistent performance reflects the model's stability, making it suitable for practical forecasting tasks where predictability and trustworthiness are essential.

To validate the uncertainty modeling capabilities of our TokenCast, we conduct experiments on both the Economic and Stock-NY datasets. As shown in Figure 6, our method produces predictive distributions that closely track the ground truth, with 50% and 80% prediction intervals capturing the inherent variability in the data. By adjusting the temperature during sampling, we observe that the model can flexibly modulate the spread of the predictive intervals, indicating its potential for controllable uncertainty-aware forecasting. This demonstrates that our model not only provides accurate mean predictions but also yields well-calibrated uncertainty estimates.

> Figure 7 (see PDF p. 18). Visualizing the reconstruction of the Nature dataset in the vector quantized networks.

### C.5. Reconstruction Analysis of Tokenizer

Figure 7 presents reconstruction results on two representative channels of the Nature dataset, clearly illustrating the ability of our discretization module to generalize across time series with different levels of complexity. For Ch-vpd (top), the reconstructed sequence almost completely overlaps with the original series, yielding very low errors (MSE = 0.062, MAE = 0.185). This shows that the module preserves both global seasonal trends and fine-grained local fluctuations, ensuring that long-term periodic patterns are faithfully retained. For Ch-ws (bottom), the series exhibits greater variability and irregular spikes, posing a more challenging scenario. Nevertheless, the reconstructed sequence closely follows the underlying dynamics of the original data, with errors kept at controlled levels (MSE = 0.062, MAE = 0.285). The consistency between reconstructions and ground truth indicates that the tokenizer is not biased toward smooth series but adapts flexibly to noisy sequences.

Overall, these results highlight two key strengths of our approach: stability, as reconstruction remains reliable across channels with distinct characteristics, and fidelity, as both trend-level and detail-level structures are preserved. Such properties are essential for downstream forecasting, where the quality of discretized representations directly determines predictive performance. By achieving accurate reconstructions on heterogeneous channels, our tokenizer provides a dependable basis for context-aware and domain-agnostic time series modeling.

> Figure 8 (see PDF p. 19). Visualize the 36-to-36 prediction results of different models on the Stock-NA dataset.

### C.6. Visualization

Figure 8 presents a qualitative comparison of 36-to-36 forecasts on the Stock-NA dataset. The LLM-based models (TokenCast, Time-LLM, GPT4TS, and SimMTM) closely follow the ground truth, capturing major turning points and preserving key high-frequency variations. Although amplitude is not always exact, the directionality and regime changes are well tracked, which is critically important for financial time series. For the non-LLM baselines, behaviors diverge. Crossformer in this figure maintains reasonable alignment after the forecasting start, but shows damped amplitudes (variance shrinkage) and occasional phase lag around sharp moves (e.g., near the regime change around steps 35-45 and subsequent fluctuations), leading to systematic underestimation of peaks and troughs rather than collapse. Autoformer tends to over-smooth, missing part of the local volatility. DLinear exhibits higher variance and noisy deviations, while TimeDART generally underestimates magnitudes and gradually drifts away from local fluctuations. Overall, the qualitative evidence indicates that LLM-based methods yield more coherent and responsive forecasts under the non-stationary, volatile conditions of stock data, whereas earlier architectures often suffer from amplitude underestimation, excessive smoothing bias, or noisy trajectories. This strongly supports the effectiveness of the unified token-based paradigm for capturing complex temporal dynamics.
