# One-for-All: A Lightweight Stabilized and Parameter-Efficient Pre-trained LLM for Time Series Forecasting

Prasanjit Dey, Soumyabrata Dev, *Member, IEEE*, and Bianca Schoen-Phelan

**Abstract.** We address the challenge of adapting pre-trained Large Language Models (LLMs) for multivariate time-series analysis, where their deployment is often hindered by prohibitive computational and memory demands. Our solution, One-for-All, introduces Gaussian Rank-Stabilized Low-Rank Adapters (rsLoRA) to enable parameter-efficient fine-tuning of frozen LLMs. While inspired by LoRA, rsLoRA introduces a mathematically grounded rank-stabilization mechanism that enables provable gradient stability at low ranks, a novel contribution absent in prior PEFT methods. Our framework injects trainable rank decomposition matrices (rank 16) into positional embeddings and output layers, while keeping self-attention weights fixed. This design reduces trainable parameters by 6.8× (vs. TimesNet), 21× (vs. GPT4TS), and 11.8× (vs. TIME-LLM), while achieving a 168–1,776× smaller memory footprint (2.2 MiB vs. 340 MiB–4.18 GiB in SOTA models). Rigorous evaluation across six time-series tasks demonstrates that One-for-All achieves state-of-the-art efficiency-accuracy trade-offs: 5.5× higher parameter efficiency (MSE = 5.50) than TimesNet and 21× better than GPT4TS, while matching their forecasting accuracy (MSE = 0.33). The framework's stability is validated through consistent performance across diverse horizons (96–720 steps) and datasets (ETT, Weather, M3, M4), with 98.3% fewer parameters than conventional transformers. These advances enable deployment on edge devices for healthcare, finance, and environmental monitoring without compromising performance. Code is available at: `https://github.com/Prasanjit-Dey/One_for_All`.

**Index Terms.** Time-series, LLMs, parameter efficient, forecasting, low rank adapters.

Manuscript received 22-May-2025; revised Month Day 2025; accepted Month Day 2025.

P. Dey is with ADAPT SFI Research Centre, School of Computer Science, Technological University Dublin, Ireland (`d22124678@mytudublin.ie`).

S. Dev is with ADAPT SFI Research Centre, School of Computer Science, University College Dublin (`soumyabrata.dev@ucd.ie`).

B. Schoen-Phelan is with ADAPT SFI Research Centre, School of Computer Science, Technological University Dublin, Ireland (`bianca.schoenphelan@tudublin.ie`).

## I. INTRODUCTION

Multivariate time-series analysis is crucial for tasks like predicting air pollution [1], weather patterns [2], financial trends [3], detecting anomalies in industrial data [4], and classifying time-series data across different domains [5]. In recent years, numerous deep learning and machine learning methods, especially in NLP and computer vision domains, have been proposed [6], [7]. The transformer architecture has been introduced to multivariate time-series analysis, achieving promising results, particularly in forecasting time series data [8], [9].

Recent advancements in Large Language Models (LLMs) have significantly enhanced their capabilities in the NLP domain [10], [11]. The main advantage is using pre-trained LLMs on billions of tokens for downstream tasks, even with few or no labeled instances. Another advantage provided by LLMs is their unified architecture for multiple downstream applications. Despite this, very few developments have been made to explore pre-trained LLMs for multivariate time-series analysis due to three key challenges. The primary challenge is the lack of sufficient data to train LLMs for multivariate time-series analysis. The maximum dataset size available for training is less than 10 GB [12], which is much less than what is typically available for NLP. The second challenge is that pre-trained LLMs require significant storage. Unified adaptation is typically achieved through fine-tuning, which updates all parameters of the pre-trained models. A major disadvantage of fine-tuning is that the new models contain as many parameters as the original models. The third and most fundamental challenge stems from the mismatch between time-series characteristics and existing parameter-efficient adaptation methods designed for NLP/CV:

1. Non-stationarity: time-series exhibit time-varying statistics (e.g., trends, seasonality) that violate the stationarity assumptions of standard adapters.
2. Irregular sampling: missing values and heterogeneous frequencies complicate tokenization compared to fixed-length text or images.
3. Multi-scale dependencies: local noise and global trends require adaptive rank allocation, whereas NLP-focused PEFT uses fixed ranks.

Many sought to address these problems by adopting only a few parameters or learning embedding and output layers for new tasks. This approach requires storing and loading only a few additional parameters with the pre-trained LLM for time-series analysis, significantly enhancing operational efficiency upon deployment. However, existing solutions often increase inference latency by extending model depth or reducing usable sequence length [13], [14]. Moreover, these algorithms often fail to match the fine-tuning baseline, which reduces the model's efficiency and quality.

Recent efforts to adapt LLMs for time-series [15] partially address data scarcity but fail to solve these inherent mismatches. Various enhancements have been implemented to advance LLM capabilities in time-series forecasting, including refined fine-tuning techniques [16], sequence decomposition strategies [17], and integrating textual prompts [18]. Despite promising results, many methods require a large number of trainable parameters, increasing storage and computational demands. Moreover, a surge in cross-model parameter-efficient LLMs has emerged, facilitating knowledge transfer across domains such as semantic segmentation [19], speaker recognition [20], and action detection [21]. However, this development primarily focuses on NLP and computer vision, leaving untapped potential in exploring cross-model parameter efficiency for temporal modalities.

## Novelty of the Proposed Framework

Our *One-for-All* framework introduces fundamental advances in time-series analysis through parameter-efficient adaptation of pre-trained LLMs. While leveraging established components (e.g., instance normalization, patching), we make the following key innovations:

- **Gaussian Rank-Stabilized LoRA (rsLoRA):** While inspired by LoRA, rsLoRA introduces a mathematically grounded rank-stabilization mechanism that enables provable gradient stability at low ranks, a novel contribution absent in prior PEFT methods. Our rank-16 adapter features Gaussian-distributed scaling $\beta_r = \alpha / \sqrt{r}$ (Theorem 1), specifically designed to address time-series challenges.
- **95% accuracy saturation by Rank 16:** Compared to Rank 256+ required in standard LoRA, this enables efficient modeling of multi-scale dependencies.
- **Theoretical guarantees:** Stable gradients under non-stationarity, the first formal proof for time-series adapters.
- **21× fewer parameters than GPT4TS:** 0.55 M vs. 3.9–24.0 M while matching accuracy.
- **First Unified Time-Series Architecture:** The framework simultaneously provides cross-task unification, horizon-agnostic stability, and provable efficiency.
- **Cross-task unification:** A single 2.2 MiB model handles irregular sampling across forecasting, classification, and anomaly detection.
- **Horizon-agnostic stability:** Less than 1% MSE variance across 96–720 steps.
- **Provable efficiency:** 5.5× better parameter efficiency (`Eff.*MSE = 5.50`) than TimesNet.
- **Breakthrough in Resource Efficiency:** Our frozen backbone + rsLoRA design achieves unprecedented efficiency.
- **98.3% parameter reduction vs. transformers:** 0.55 M vs. 10.53 M in Autoformer.
- **Edge deployment capability:** 1,776× smaller memory than TIME-LLM (2.2 MiB vs. 3.9 GiB).
- **Sub-0.35 MSE with 170× fewer resources than GPT4TS.**

These advances directly address the core challenges of time-series PEFT. rsLoRA's Gaussian scaling ensures stability under non-stationarity, and patch-based tokens effectively handle irregular sampling. Prior cross-modal adapters have not achieved either of these capabilities.

## II. RELATED WORK

### A. Traditional Approach

Time series analysis models are primarily classified into two groups: classical ARIMA [22] models and the latest transformer models. The first-generation models, such as ARIMA, which follow the Markov process [23], are good at forecasting stationary time series. However, recently, most time series tasks are non-stationary in nature, and first-generation models are limited in handling this type of data. Additionally, with the rise of deep learning-based models such as Long Short-Term Memory (LSTM) [24], and Gated Recurrent Unit (GRU) [25], which were developed for sequential applications, recurrent models have become ineffective in handling long-term dependencies and are still unresolved.

### B. In Modality Adaptation

In recent years, a large number of pre-trained NLP and CV models have demonstrated effectiveness, allowing them to be fine-tuned for a variety of tasks without requiring training from scratch [26]. OpenAI developed GPT models [27] that train transformer decoders using larger language datasets and then fine-tune them for specific tasks. GPT-2 [28] is trained using a larger dataset with billions of parameters and can be fine-tuned for various downstream tasks. Inspired by this research, recent investigations have focused on developing pre-training models specifically tailored for time series data. The initial steps among them involve adopting supervised [29] and self-supervised [30] learning strategies for time series pre-training. This enables the model to learn representations of multiples input time series. However, while pre-training and fine-tuning models have achieved success in NLP and CV, their application in time series tasks remains limited on a smaller scale due to dataset constraints.

### C. Cross Modality Adaptation

In recent papers, there has been improvement in transformer-based models [31] for time series prediction through the incorporation of decomposition, patching, frequency analysis, and exponential smoothing techniques. For instance, ETSformer [32] utilizes exponential smoothing attention and frequency attention instead of self-attention in the transformer architecture to enhance prediction accuracy in time series data. FEDformer [33] integrates seasonal-trend decomposition into the transformer architecture, enhancing global profile capture. It also utilizes frequency augmentation to enhance long-term prediction accuracy compared to traditional transformer architectures. Autoformer [34] presented progressive decomposition by utilizing autocorrelation for complex time-series forecasting. Designed with a focus on temporal periodicity, it tackles dependency disorder and representation aggregation, outperforming self-attention in performance.

Although these models excel in domain-specific tasks and enhance prediction accuracy compared to traditional transformers, their performance heavily relies on task-specific datasets, limiting generalization across diverse time series data. Advancing general time series analysis requires more flexible models capable of adapting to a wide range of tasks without extensive fine-tuning. An ideal model should be capable of handling diverse time-series applications and transferring knowledge across domains. Developing such universally applicable models remains a significant challenge. Previous related work has seen the adoption of a few pre-trained LLM-based models for various time series analyses. More efforts are needed to advance unified forecasting systems for time-series data, which is the focus of our research.

We extend related work by exploring cross-domain modalities and leveraging pre-trained NLP and CV models for time series analysis through multi-modal fine-tuning [35] and model reprogramming [36]. For instance, Yang et al. [37] applied the Voice2Series framework, initially designed for speech recognition, to time series classification. Similarly, Zhou et al. [15] introduced the GPT4TS model, keeping self-attention and feed-forward layers fixed, then fine-tuned it for various time series tasks. Despite their success, such models often require numerous trainable parameters. To address this, we propose the One-for-All framework, which utilizes pre-trained LLMs with Gaussian rank stabilization (rsLoRA) for parameter-efficient time series forecasting.

> Figure 1 (see PDF p. 3). One-for-All Framework: A parameter-efficient LLM unifying long-term, few-shot, zero-shot, short-term forecasting, classification, and anomaly detection. By integrating Gaussian rank-stabilized LoRA (rsLoRA) into positional embeddings and output layers while freezing the pre-trained LLM weights, the framework minimizes trainable parameters without compromising stability.

## III. METHODOLOGY

Our One-for-All framework, depicted in Figure 1, leverages rsLoRA [38] (rank 16), a Gaussian-distributed parameter-efficient low-rank stabilized matrix, in the positional embedding and output layers of pre-trained LLMs models like GPT-2 [28]. This optimization enhances time series tasks without the need to fine-tune the backbone LLM model. rsLoRA incorporates the "intrinsic dimension" during weight updates in pre-trained LLMs. For these models, the weight matrix $W_o \in \mathbb{R}^{d \times d'}$, where $d$ and $d'$ are the dimensions, is updated using a parameter-efficient stabilized low-rank decomposition:

$$
W_o + \Delta W = W_o + \beta_r Y X
\tag{1}
$$

Where matrices $Y \in \mathbb{R}^{d \times r}$ and $X \in \mathbb{R}^{r \times d'}$, and $r \ll \min(d, d')$ denotes the rank, and $\beta_r \in \mathbb{R}^+$ is a scaling factor. Throughout the fine-tuning phase, the weight matrix $W_o$ remains fixed, undergoing no gradient updates, while matrices $X$ and $Y$ remain trainable. We initialize $X$ randomly with a Gaussian distribution and set $Y$ to zero. The gradient update $\Delta W = YX$ is zero at the beginning of training. The scaling factor $\beta_r$ acts as a parameter-efficient mechanism for the matrix product $YX$, adapting its value based on the rank $r$. Specifically, $\beta_r$ is defined as $\alpha / \sqrt{r}$, where $\alpha$ represents a hyperparameter. Our ablation studies (Section VI, Table VII) reveal that $\alpha \in [0.8, 1.2]$ achieves optimal performance across all tasks, with $\alpha = 1.0$ providing the best balance between stability and convergence speed. Notably, performance variations were minimal ($\Delta$MSE < 2%) for $r \ge 16$ (Figure 6), demonstrating that rsLoRA's effectiveness is more sensitive to rank selection than to the exact value of $\alpha$. This aligns with Theorem 1's stability guarantees, which hold for any $\alpha \in \Theta(1)$. For reproducibility, we fix $\alpha = 1.0$ as the default setting. Remarkably, this scaling factor $\beta_r$ exhibits stability in gradients, facilitating the increase in rank.

### A. Frozen Pre-trained LLM

Our One-for-All framework preserves the self-attention block from the pre-trained LLM, transferring its knowledge to our downstream time series analysis. We opt to maintain this layer frozen during fine-tuning.

### B. Positional and Output Layer Stabilization

**Definition 1.** An adapter $\beta_r Y X$ (either positional embedding or output layer) is Gaussian rank-stabilized if the following conditions hold:

1. If the inputs to the adapter are i.i.d. Gaussian random variables with the $n$-th moment $\Theta_r(1)$ in each entry, then the $n$-th moment of the outputs of the adapter is also $\Theta_r(1)$ in each entry.
2. If the gradient of the loss with respect to the adapter outputs is $\Theta_r(1)$ in each entry, then the loss gradients with respect to the inputs of the adapter are also $\Theta_r(1)$ in each entry.

**Theorem 1.** Consider a pre-trained language model with an adapter $\beta_r Y X$, where $Y \in \mathbb{R}^{d \times r}$ is initialized to $0_{d \times r}$, and entries of $X \in \mathbb{R}^{r \times d'}$ are i.i.d. Gaussian random variables with zero mean and variance $\sigma_X^2$. Here, $d$ and $d'$ are the dimensions. The scaling factor $\beta_r \in \mathbb{R}$ is such that $\lim_{r \to \infty} \beta_r = 0$. In expectation over initialization, all adapters (positional embedding or output layer) are Gaussian rank-stabilized if and only if $\beta_r \in \Theta_r(1 / \sqrt{r})$.

**Detailed Proof.** Let $f(z) = \beta_r Y X z$ represent the positional embedding and output layer adapter, where $\mathcal{L}(f(z))$ denotes the loss function for both the adapter and the input data $z$. After the $i$-th SGD update on input $z_i$ with learning rate $\eta$, let $Y$ and $X$ be updated to $Y_i$ and $X_i$ respectively. The gradient updates for both the positional and output adapters are then defined as:

$$
\nabla_{Y_i}\mathcal{L} = \beta_r V_i z_i^T X_i^T
\tag{2}
$$

$$
\nabla_{X_i}\mathcal{L} = \beta_r Y_i^T V_i z_i^T
\tag{3}
$$

where $V_i = \nabla_{f(z_i)} \mathcal{L}(f(z_i))$. By mathematical induction for $i \ge 1$, we obtain:

$$
Y_i = \left(-\eta \beta_r \sum_{k=0}^{i-1} V_k z_k^T + O(\beta_r^2)\right) X_0^T
\tag{4}
$$

$$
X_i = X_0 \left(1 + O(\beta_r^2)\right)
\tag{5}
$$

Consequently, following the $i$-th updates for both the positional embedding and output layer adapter, we obtain:

$$
\beta_r Y_i X_i = -\beta_r^2 \eta \sum_{k=0}^{i-1} V_k z_k^T X_0^T X_0 + O_r(\beta_r^3) X_0^T X_0
\tag{6}
$$

By considering the expectation of the initialization $X_0$, with $\mathbb{E}_{X_0}(X_0^T X_0)$, we find that:

$$
\mathbb{E}_{X_0}(\beta_r Y_i X_i) = -\beta_r^2 r \sigma_X^2 \eta \sum_{k=0}^{i-1} V_k z_k^T + O_r(\beta_r^3 r)
\tag{7}
$$

**Gradient Updates for Backward Pass.** On new input data $z_i$, the gradient of the positional embedding and output embedding are updated as follows:

$$
\nabla_{z_i}\mathcal{L}(\beta_r Y_i X_i z_i)
= -\beta_r^2 r \sigma_X^2 \eta \sum_{k=0}^{i-1} z_k V_k^T V_i
+ O_r(\beta_r^3 r)
\in \Theta_r(\beta_r^2 r)
\tag{8}
$$

**Gradient Updates for Forward Pass.** For new input data $z_i$, assume the inputs are i.i.d. Gaussian random variables with moments $\Theta_r(1)$, where $\mathbb{E}_z((z_k^T z_i)^n) \in \Theta_r(1)$. Then, assuming $\beta_r \to 0$, the gradient updates are as follows:

$$
\mathbb{E}_{z, X_0}\!\left((\beta_r Y_i X_i z_i)^n\right)
= (-\beta_r^2 r \sigma_X^2 \eta)^n \sum_{k=0}^{i-1} V_k^n \, \mathbb{E}_z((z_k^T z_i)^n)
+ O_r((\beta_r^3 r)^n)
\in \Theta_r((\beta_r^2 r)^n)
\tag{9}
$$

The rank is stable up to $r \to \infty$ if and only if:

$$
\Theta_r((\beta_r^2 r)^n) = \Theta_r(1)
\tag{10}
$$

$$
\Leftrightarrow \beta_r \in \Theta_r\!\left(\frac{1}{\sqrt{r}}\right)
\tag{11}
$$

### C. Input Embedding

To process the raw time series $X \in \mathbb{R}^{L \times d}$, where $L$ is the sequence length and $d$ is the number of variables, we apply a shared linear projection to map inputs into the model space:

$$
X_{\text{embed}} = X \cdot W_{\text{embed}}, \qquad W_{\text{embed}} \in \mathbb{R}^{d \times D}
\tag{12}
$$

where $D$ is the hidden dimension. Unlike transformer models for text, we do not use positional encodings, as prior work (e.g., TimesNet [39]) has shown minimal benefit for temporal dynamics. This also reduces parameter overhead and simplifies deployment.

### D. Normalization

Before patching, we apply Z-score normalization independently to each time series:

$$
\tilde{X} = \frac{X - \mu}{\sigma}
\tag{13}
$$

where $\mu$ and $\sigma$ are the mean and standard deviation of the series computed over the input window. This standardization is done both during training and inference, and is reversed post-prediction. It ensures consistent feature scaling across time and tasks.

### E. Patching

To improve computational efficiency and capture local trends, we segment the input series into non-overlapping patches of size $P$. Each patch is flattened and projected via a learnable linear layer:

$$
Z_i = \operatorname{Linear}(\operatorname{Flatten}(\tilde{X}_{i:i+P-1}))
\tag{14}
$$

resulting in a token sequence $\{Z_1, \ldots, Z_N\} \in \mathbb{R}^{N \times D}$ where $N = L / P$. This tokenized form is passed to the frozen LLM backbone.

## IV. EXPERIMENT DETAILS

### A. Implementation Details

We adopt the TimesNet configuration [39] for all baseline models to ensure a fair comparison. Our framework is implemented in PyTorch, with GPT-2 [40] as the default backbone LLM, enhanced with rank-16 rsLoRA. All models are trained on NVIDIA A100 80 GB GPUs. To ensure robustness, experiments are repeated three times, and results are reported as the mean performance. Detailed hyperparameters and runtime settings are provided in Section A of the supplementary material.

### B. Datasets for Time-Series Tasks

We evaluate our approach across six time-series tasks:

1. **Long-term and Few-shot Forecasting:** Benchmarked on five standard datasets: ETT (ETTh, ETTm) and Weather [41].
2. **Zero-shot Forecasting:** Evaluated on the M3 and M4 datasets, which span diverse frequencies (yearly, quarterly, monthly).[^tsforecasting]
3. **Short-term Forecasting:** Assessed on the M4 dataset, a large-scale collection of business, financial, and economic time series with heterogeneous frequencies.
4. **Classification:** Tested on multivariate UEA [42] classification datasets.
5. **Anomaly Detection:** Evaluated on five widely used benchmarks: SMD [43], MSL [44], SMAP [44], SWaT [45], and PSM [46].

For dataset statistics (e.g., dimensions, splits, and frequencies), refer to Supplementary Tables 1–2.

[^tsforecasting]: `https://github.com/rakshitha123/TSForecasting`

## V. EXPERIMENTAL RESULTS

We evaluate the Gaussian rank-stabilized parameter-efficient One-for-All framework through comprehensive experiments across multiple time series forecasting tasks. Our results demonstrate that the proposed method achieves state-of-the-art performance while maintaining exceptional efficiency in both parameter count and memory footprint. Using a GPT-2 backbone enhanced with Gaussian rsLoRA [38], the framework shows consistent performance across different prediction horizons while reducing trainable parameters by 6.8–21× compared to contemporary approaches. The rank stabilization properties are rigorously validated across six distinct time-series analysis tasks (Supplementary Tables 4–13), with detailed parameter counts and memory requirements provided in Supplementary Tables 14–17.

**Baseline Models.** Our comparative analysis includes: (1) modern LLM-based temporal models (GPT4TS [15], TIME-LLM [18], TEST [47], TEMPO [17]), (2) transformer variants (FEDformer [33], Non-Stationary Transformer [48]), and (3) established forecasting architectures (TimeNet [39], ETSformer [32], LightTS [49]). We also benchmark against conventional temporal transformers (Autoformer [34], Informer [41], Reformer [50]) to demonstrate comprehensive improvements in both efficiency and accuracy.

### A. Model Efficiency Analysis Across Forecasting Horizons

The One-for-All framework demonstrates exceptional efficiency across both trainable parameters and memory footprint when evaluated against state-of-the-art time series forecasting models, as evidenced in Figure 2. Across all prediction horizons (96 to 720 steps), our method maintains a consistently low parameter count of 0.546–0.556 million, representing a 6.8× reduction compared to TimesNet (0.605–0.666 M), a 21× improvement over GPT4TS (3.916–24.04 M), and an 11.8× advantage relative to TIME-LLM (6.39–6.55 M). Notably, it achieves a 98.3% parameter reduction versus conventional transformers like Autoformer (10.53 M) and Informer (11.33 M), occupying a distinct efficiency regime in the logarithmic-scale comparison. The memory efficiency proves equally compelling, with a fixed 2.2 MiB footprint that is 168× smaller than GPT4TS (340–420 MiB), 1,776× more compact than TIME-LLM (3.68–4.18 GiB), and 28–32× leaner than medium-sized GPT-2 variants (TEST/TEMPO: 701–710 MiB). Unlike other methods that exhibit 5–25% size variations across horizons, One-for-All maintains consistent resource requirements regardless of prediction length. The "Avg" columns further highlight this sustained advantage, with our framework achieving 5.5× higher parameter efficiency (`Eff.*MSE`) than TimesNet while using 30% fewer parameters (0.55 M vs. 0.63 M). This dual efficiency, coupled with competitive accuracy (Avg MSE = 0.33, Table I), enables deployment on resource-constrained edge devices without compromising forecasting capability.

> Figure 2 (see PDF p. 6). Comparison of model efficiency across different forecast horizons. (a) Trainable parameters (in millions, log scale) and (b) model size (in MiB, log scale) for various time-series forecasting approaches. The One-for-All model (highlighted in red) demonstrates consistently low parameter counts and memory footprint across all horizons, while other methods (dashed lines) show varying computational requirements. Notably, large pretrained models (e.g., TIME-LLM and GPT4TS) exhibit significantly higher resource demands, particularly at longer horizons (336 and 720). The Avg column represents the average across all horizons, further emphasizing the efficiency advantages of the proposed approach.

### B. Long-term Forecasting

The experimental results demonstrate that our One-for-All framework establishes new benchmarks in efficiency-accuracy trade-offs for long-term time series forecasting. As quantified in Table I, the proposed method achieves an unmatched parameter efficiency (`Eff.*MSE = 5.50`) that is 5.5× higher than TimesNet (4.41) and 21× superior to GPT4TS (0.26), while maintaining competitive prediction accuracy (Avg MSE = 0.33). This efficiency advantage manifests most dramatically in memory requirements, where our 2.2 MiB footprint represents a 170× reduction compared to GPT4TS (371 MiB) and a remarkable 1,800× improvement over TIME-LLM (3907 MiB). The visualization in Figure 3 shows that One-for-All occupies a unique position in the design space, as no other method simultaneously achieves a sub-0.35 MSE with memory requirements under 3 MiB.

Detailed analysis shows our framework matches the accuracy of GPT4TS (0.33 vs. 0.33 avg MSE) while using just 4.7% of its parameters (0.55 M vs. 11.7 M) and 0.6% of its memory footprint. Against the more accurate but resource-intensive TIME-LLM (0.31 MSE), our solution provides 91.5% parameter reduction and 99.94% memory savings with only a 6.5% relative increase in MSE. The efficiency gains become even more pronounced when compared to conventional transformers, requiring 98% fewer parameters than Autoformer (10.53 M) while delivering 33.3% better accuracy.

The consistent performance across all five benchmark datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather) further validates the robustness of our approach. As shown in Table I, One-for-All maintains stable MSE values within a narrow 0.23–0.43 range, outperforming TimesNet in 4 of 5 datasets despite using 13% fewer parameters. The method particularly excels in the Weather dataset (MSE = 0.23), where it achieves 8% better accuracy than TIME-LLM with 11.7× greater parameter efficiency.

These results collectively demonstrate that our Gaussian rank stabilization and parameter-efficient design successfully decouple model size from forecasting capability, enabling deployment in resource-constrained environments without sacrificing prediction quality. The comprehensive advantages are visually apparent in Figure 3, where One-for-All appears as a clear outlier in the desirable lower-left quadrant of the accuracy-efficiency space.

> Table I (see PDF p. 6). Comparative analysis of efficiency-accuracy trade-offs in long-term time-series forecasting. One-for-All achieves state-of-the-art efficiency with minimal resource overhead: 5.5× higher parameter efficiency (`Eff.*MSE = 5.50` vs. 0.05–4.41 in SOTA) and 2.2 MiB memory (170–1,800× smaller than pretrained models), while matching top accuracy (Avg MSE = 0.33) across all benchmarks.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Parameters (M)</th>
      <th rowspan="2">Memory (MiB)</th>
      <th rowspan="2">Eff.*MSE/MAE</th>
      <th>ETTh1</th>
      <th>ETTh2</th>
      <th>ETTm1</th>
      <th>ETTm2</th>
      <th>Weather</th>
      <th>Avg</th>
    </tr>
    <tr>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>One-for-All</td>
      <td>0.55</td>
      <td>2.2</td>
      <td>5.50/5.05</td>
      <td>0.43/0.43</td>
      <td>0.36/0.39</td>
      <td>0.36/0.38</td>
      <td>0.26/0.33</td>
      <td>0.23/0.27</td>
      <td>0.33/0.36</td>
    </tr>
    <tr>
      <td>GPT4TS</td>
      <td>11.7</td>
      <td>371.0</td>
      <td>0.26/0.24</td>
      <td>0.43/0.43</td>
      <td>0.35/0.40</td>
      <td>0.35/0.38</td>
      <td>0.27/0.33</td>
      <td>0.24/0.27</td>
      <td>0.33/0.36</td>
    </tr>
    <tr>
      <td>TIME-LLM</td>
      <td>6.46</td>
      <td>3907</td>
      <td>0.50/0.44</td>
      <td>0.41/0.42</td>
      <td>0.34/0.38</td>
      <td>0.33/0.37</td>
      <td>0.25/0.32</td>
      <td>0.23/0.26</td>
      <td>0.31/0.35</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>2.00</td>
      <td>701</td>
      <td>1.56/1.39</td>
      <td>0.42/0.43</td>
      <td>0.34/0.38</td>
      <td>0.35/0.38</td>
      <td>0.26/0.32</td>
      <td>0.23/0.27</td>
      <td>0.32/0.36</td>
    </tr>
    <tr>
      <td>TEMPO</td>
      <td>5.00</td>
      <td>710</td>
      <td>0.54/0.51</td>
      <td>0.43/0.43</td>
      <td>0.36/0.40</td>
      <td>0.50/0.46</td>
      <td>0.28/0.33</td>
      <td>0.28/0.32</td>
      <td>0.37/0.39</td>
    </tr>
    <tr>
      <td>TimesNet</td>
      <td>0.63</td>
      <td>2.9</td>
      <td>4.41/4.18</td>
      <td>0.46/0.45</td>
      <td>0.41/0.43</td>
      <td>0.40/0.41</td>
      <td>0.29/0.33</td>
      <td>0.26/0.29</td>
      <td>0.36/0.38</td>
    </tr>
    <tr>
      <td>FEDformer</td>
      <td>16.8</td>
      <td>87.8</td>
      <td>0.15/0.15</td>
      <td>0.44/0.46</td>
      <td>0.44/0.45</td>
      <td>0.45/0.45</td>
      <td>0.30/0.35</td>
      <td>0.31/0.36</td>
      <td>0.39/0.41</td>
    </tr>
    <tr>
      <td>TStationary</td>
      <td>2.02</td>
      <td>13.2</td>
      <td>1.12/1.12</td>
      <td>0.57/0.54</td>
      <td>0.53/0.52</td>
      <td>0.48/0.46</td>
      <td>0.31/0.35</td>
      <td>0.29/0.31</td>
      <td>0.44/0.44</td>
    </tr>
    <tr>
      <td>ETSformer</td>
      <td>5.28</td>
      <td>31.4</td>
      <td>0.48/0.46</td>
      <td>0.54/0.51</td>
      <td>0.44/0.45</td>
      <td>0.43/0.43</td>
      <td>0.29/0.34</td>
      <td>0.27/0.31</td>
      <td>0.39/0.41</td>
    </tr>
    <tr>
      <td>Autoformer</td>
      <td>10.53</td>
      <td>62.6</td>
      <td>0.22/0.22</td>
      <td>0.50/0.49</td>
      <td>0.45/0.46</td>
      <td>0.59/0.52</td>
      <td>0.33/0.37</td>
      <td>0.34/0.38</td>
      <td>0.44/0.44</td>
    </tr>
    <tr>
      <td>Informer</td>
      <td>11.33</td>
      <td>65.8</td>
      <td>0.05/0.10</td>
      <td>1.04/0.79</td>
      <td>4.43/1.73</td>
      <td>0.96/0.73</td>
      <td>1.41/0.81</td>
      <td>0.64/0.55</td>
      <td>1.70/0.92</td>
    </tr>
    <tr>
      <td>Reformer</td>
      <td>5.80</td>
      <td>33.4</td>
      <td>0.08/0.16</td>
      <td>1.06/0.81</td>
      <td>6.74/2.19</td>
      <td>0.80/0.67</td>
      <td>1.48/0.92</td>
      <td>0.80/0.64</td>
      <td>2.18/1.05</td>
    </tr>
  </tbody>
</table>

`Eff.*MSE = (1 / MSE) / Parameters`; `Eff.*MAE = (1 / MAE) / Parameters`. Higher efficiency scores indicate better parameter efficiency for a given metric.

> Figure 3 (see PDF p. 6). Trade-offs Between Model Accuracy, Efficiency, and Scalability for Long-Term Forecasting. The average MSE (y-axis) measures prediction accuracy (lower is better), while the number of trainable parameters (x-axis) reflects model efficiency (leftward is better). The bubble sizes represent memory usage (smaller is better), highlighting scalability constraints.

### C. Few-shot Forecasting

The few-shot evaluation demonstrates our framework's exceptional data efficiency across both 10% and 5% training scenarios. As shown in Tables II and III, One-for-All maintains its Pareto-optimal efficiency-accuracy balance, achieving 4.37 and 3.92 `Eff.*MSE` respectively, 3.9–4.4× higher than TimesNet and 14–180× better than pretrained baselines. While TIME-LLM shows marginally better accuracy (0.372 vs. 0.416 MSE with 10% data), it requires 11.7× more parameters and 1,776× greater memory. Our solution proves particularly effective in extreme low-data conditions, where its 5% performance (0.464 MSE) surpasses TimesNet's 10% results (0.526 MSE) despite equivalent parameter counts.

The visualization in Figure 4 reveals three key insights: (1) One-for-All occupies the exclusive low-resource/high-accuracy quadrant, (2) model accuracy degrades gracefully with reduced data (only +11.5% MSE from 10% to 5% training), and (3) maintains 99.4% smaller memory footprint than comparable performers. Notably, it achieves 98% parameter reduction versus GPT4TS with just +4% relative MSE increase in 10% scenarios, demonstrating superior few-shot adaptation capability through its stabilized parameter-efficient design.

> Table II (see PDF p. 7). Comparative analysis of efficiency-accuracy trade-offs in few-shot time-series forecasting with 10% data.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Parameters (M)</th>
      <th rowspan="2">Memory (MiB)</th>
      <th rowspan="2">Eff.*MSE/MAE</th>
      <th>ETTh1</th>
      <th>ETTh2</th>
      <th>ETTm1</th>
      <th>ETTm2</th>
      <th>Weather</th>
      <th>Avg</th>
    </tr>
    <tr>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>One-for-All</td>
      <td>0.55</td>
      <td>2.2</td>
      <td>4.37/4.48</td>
      <td>0.65/0.55</td>
      <td>0.43/0.44</td>
      <td>0.47/0.44</td>
      <td>0.29/0.33</td>
      <td>0.24/0.27</td>
      <td>0.416/0.406</td>
    </tr>
    <tr>
      <td>GPT4TS</td>
      <td>11.7</td>
      <td>371.0</td>
      <td>0.21/0.21</td>
      <td>0.59/0.53</td>
      <td>0.40/0.42</td>
      <td>0.47/0.44</td>
      <td>0.30/0.34</td>
      <td>0.24/0.27</td>
      <td>0.400/0.400</td>
    </tr>
    <tr>
      <td>TIME-LLM</td>
      <td>6.46</td>
      <td>3907</td>
      <td>0.43/0.40</td>
      <td>0.56/0.52</td>
      <td>0.37/0.40</td>
      <td>0.41/0.43</td>
      <td>0.28/0.33</td>
      <td>0.24/0.28</td>
      <td>0.372/0.392</td>
    </tr>
    <tr>
      <td>TEST</td>
      <td>2.00</td>
      <td>701</td>
      <td>1.24/1.26</td>
      <td>0.59/0.53</td>
      <td>0.40/0.43</td>
      <td>0.46/0.45</td>
      <td>0.32/0.31</td>
      <td>0.25/0.27</td>
      <td>0.404/0.398</td>
    </tr>
    <tr>
      <td>TimesNet</td>
      <td>0.63</td>
      <td>2.9</td>
      <td>3.02/3.46</td>
      <td>0.87/0.63</td>
      <td>0.48/0.47</td>
      <td>0.68/0.54</td>
      <td>0.32/0.35</td>
      <td>0.28/0.30</td>
      <td>0.526/0.458</td>
    </tr>
    <tr>
      <td>FEDformer</td>
      <td>16.8</td>
      <td>87.8</td>
      <td>0.12/0.12</td>
      <td>0.64/0.56</td>
      <td>0.47/0.48</td>
      <td>0.72/0.61</td>
      <td>0.46/0.49</td>
      <td>0.29/0.32</td>
      <td>0.516/0.492</td>
    </tr>
    <tr>
      <td>TStationary</td>
      <td>2.02</td>
      <td>13.2</td>
      <td>0.88/1.05</td>
      <td>0.92/0.64</td>
      <td>0.46/0.45</td>
      <td>0.80/0.58</td>
      <td>0.33/0.37</td>
      <td>0.32/0.32</td>
      <td>0.566/0.472</td>
    </tr>
    <tr>
      <td>ETSformer</td>
      <td>5.28</td>
      <td>31.4</td>
      <td>0.25/0.31</td>
      <td>1.18/0.83</td>
      <td>0.89/0.71</td>
      <td>0.98/0.72</td>
      <td>0.45/0.49</td>
      <td>0.32/0.36</td>
      <td>0.764/0.622</td>
    </tr>
    <tr>
      <td>Autoformer</td>
      <td>10.53</td>
      <td>62.6</td>
      <td>0.13/0.16</td>
      <td>0.70/0.60</td>
      <td>0.49/0.50</td>
      <td>0.80/0.63</td>
      <td>1.34/0.93</td>
      <td>0.30/0.34</td>
      <td>0.726/0.600</td>
    </tr>
    <tr>
      <td>Informer</td>
      <td>11.33</td>
      <td>65.8</td>
      <td>0.04/0.09</td>
      <td>1.20/0.81</td>
      <td>3.87/1.51</td>
      <td>1.19/0.82</td>
      <td>3.37/1.44</td>
      <td>0.60/0.50</td>
      <td>2.046/1.016</td>
    </tr>
    <tr>
      <td>Reformer</td>
      <td>5.80</td>
      <td>33.4</td>
      <td>0.08/0.09</td>
      <td>1.25/0.83</td>
      <td>3.48/1.49</td>
      <td>1.43/0.86</td>
      <td>3.98/1.59</td>
      <td>0.54/0.47</td>
      <td>2.136/1.048</td>
    </tr>
  </tbody>
</table>

> Table III (see PDF p. 7). Comparative analysis of efficiency-accuracy trade-offs in few-shot time-series forecasting with 5% data.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Parameters (M)</th>
      <th rowspan="2">Memory (MiB)</th>
      <th rowspan="2">Eff.*MSE/MAE</th>
      <th>ETTh1</th>
      <th>ETTh2</th>
      <th>ETTm1</th>
      <th>ETTm2</th>
      <th>Weather</th>
      <th>Avg</th>
    </tr>
    <tr>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
      <th>MSE/MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>One-for-All</td>
      <td>0.55</td>
      <td>2.2</td>
      <td>3.92/4.17</td>
      <td>0.71/0.57</td>
      <td>0.52/0.49</td>
      <td>0.52/0.47</td>
      <td>0.30/0.35</td>
      <td>0.27/0.30</td>
      <td>0.464/0.436</td>
    </tr>
    <tr>
      <td>GPT4TS</td>
      <td>11.7</td>
      <td>371.0</td>
      <td>0.20/0.20</td>
      <td>0.68/0.56</td>
      <td>0.40/0.43</td>
      <td>0.47/0.45</td>
      <td>0.31/0.35</td>
      <td>0.27/0.30</td>
      <td>0.426/0.418</td>
    </tr>
    <tr>
      <td>TIME-LLM</td>
      <td>6.46</td>
      <td>3907</td>
      <td>0.39/0.38</td>
      <td>0.63/0.55</td>
      <td>0.39/0.42</td>
      <td>0.43/0.44</td>
      <td>0.28/0.32</td>
      <td>0.26/0.31</td>
      <td>0.398/0.408</td>
    </tr>
    <tr>
      <td>TimesNet</td>
      <td>0.63</td>
      <td>2.9</td>
      <td>2.89/3.39</td>
      <td>0.92/0.64</td>
      <td>0.46/0.45</td>
      <td>0.72/0.56</td>
      <td>0.35/0.37</td>
      <td>0.30/0.32</td>
      <td>0.550/0.468</td>
    </tr>
    <tr>
      <td>FEDformer</td>
      <td>16.8</td>
      <td>87.8</td>
      <td>0.12/0.13</td>
      <td>0.66/0.56</td>
      <td>0.44/0.45</td>
      <td>0.73/0.59</td>
      <td>0.40/0.41</td>
      <td>0.31/0.35</td>
      <td>0.508/0.472</td>
    </tr>
    <tr>
      <td>TStationary</td>
      <td>2.02</td>
      <td>13.2</td>
      <td>0.84/1.03</td>
      <td>0.94/0.64</td>
      <td>0.47/0.46</td>
      <td>0.86/0.60</td>
      <td>0.34/0.37</td>
      <td>0.33/0.33</td>
      <td>0.588/0.480</td>
    </tr>
    <tr>
      <td>ETSformer</td>
      <td>5.28</td>
      <td>31.4</td>
      <td>0.24/0.29</td>
      <td>1.19/0.84</td>
      <td>0.81/0.68</td>
      <td>1.13/0.79</td>
      <td>0.53/0.55</td>
      <td>0.33/0.37</td>
      <td>0.798/0.646</td>
    </tr>
    <tr>
      <td>Autoformer</td>
      <td>10.53</td>
      <td>62.6</td>
      <td>0.18/0.20</td>
      <td>0.72/0.60</td>
      <td>0.47/0.49</td>
      <td>0.80/0.49</td>
      <td>0.39/0.44</td>
      <td>0.31/0.35</td>
      <td>0.538/0.474</td>
    </tr>
    <tr>
      <td>Informer</td>
      <td>11.33</td>
      <td>65.8</td>
      <td>0.04/0.08</td>
      <td>1.22/0.82</td>
      <td>3.92/1.65</td>
      <td>1.16/0.79</td>
      <td>3.66/1.49</td>
      <td>0.59/0.53</td>
      <td>2.110/1.056</td>
    </tr>
    <tr>
      <td>Reformer</td>
      <td>5.80</td>
      <td>33.4</td>
      <td>0.09/0.17</td>
      <td>1.24/0.83</td>
      <td>3.53/1.47</td>
      <td>1.27/0.83</td>
      <td>3.58/1.49</td>
      <td>0.45/0.45</td>
      <td>2.014/1.014</td>
    </tr>
  </tbody>
</table>

> Figure 4 (see PDF p. 6). Trade-offs Between Model Accuracy, Efficiency, and Scalability for few-shot forecasting with 10% data. The average MSE (y-axis) measures prediction accuracy (lower is better), while the number of trainable parameters (x-axis) reflects model efficiency (leftward is better). The bubble sizes represent memory usage (smaller is better), highlighting scalability constraints.

### D. Zero-shot Forecasting

The zero-shot evaluation reveals distinct performance patterns across different transfer directions. For M3→M4 transfer, One-for-All achieves the lowest average sMAPE (13.27) among all methods, outperforming GPT4TS (13.55) and TimesNet (15.01). The framework demonstrates particular strength in yearly forecasting (13.53 sMAPE), where it surpasses GPT4TS by 2.89 points and TimesNet by 5.22 points. Notably, it ties with GPT4TS for the most top-performing configurations (3 each) in this direction. The M4→M3 transfer presents a more challenging scenario, where GPT4TS maintains an advantage with lower average sMAPE (13.05 vs. 15.10). However, One-for-All still shows competitive performance in monthly forecasting (14.30 sMAPE), outperforming GPT4TS (14.63) and achieving the best result in this category. The quarterly frequency reveals the most significant gap, with GPT4TS (10.78) leading our method (11.77) by nearly 1 point.

Across both transfer directions, One-for-All consistently outperforms conventional transformers, showing particular advantages over Autoformer (15.67/18.98 avg sMAPE) and Informer (16.20/19.63). The results demonstrate the framework's robust cross-domain capabilities, especially in the M3→M4 direction where it achieves the overall best average performance while matching the top-performing count of the larger GPT4TS model.

> Table IV (see PDF p. 7). Zero-shot forecasting performance (sMAPE) showing cross-domain transfer learning results between M3 and M4 datasets.

| Transfer | Frequency | One-for-All | GPT4TS | TimeNet | FEDformer | TStationary | ETSformer | LightTS | Autoformer | Informer | Reformer |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M3→M4 | Yearly | 13.53 | 16.42 | 18.75 | 16.00 | 17.05 | 20.56 | 15.63 | 16.18 | 19.70 | 16.03 |
| M3→M4 | Quarterly | 11.00 | 10.13 | 12.26 | 9.48 | 12.56 | 11.65 | 9.40 | 13.92 | 13.00 | 9.76 |
| M3→M4 | Monthly | 15.28 | 14.10 | 14.01 | 15.12 | 16.82 | 16.97 | 24.60 | 16.91 | 15.91 | 14.80 |
| M3→M4 | Avg | 13.27 | 13.55 | 15.01 | 13.53 | 15.48 | 16.39 | 16.54 | 15.67 | 16.20 | 13.53 |
| M4→M3 | Yearly | 19.23 | 13.74 | 15.65 | 13.88 | 14.98 | 27.84 | 13.78 | 14.55 | 18.54 | 15.65 |
| M4→M3 | Quarterly | 11.77 | 10.78 | 11.87 | 11.51 | 11.68 | 36.13 | 11.28 | 17.34 | 16.90 | 11.05 |
| M4→M3 | Monthly | 14.30 | 14.63 | 16.16 | 18.15 | 16.09 | 25.11 | 15.18 | 25.06 | 23.45 | 15.60 |
| M4→M3 | Avg | 15.10 | 13.05 | 14.56 | 14.51 | 14.25 | 29.69 | 13.41 | 18.98 | 19.63 | 14.10 |
| Top-1 Count |  | 3 | 3 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 |

### E. Short-term Forecasting

Our evaluation of short-term forecasting performance on the M4 dataset reveals the consistent effectiveness of the One-for-All framework compared to state-of-the-art baselines. As shown in Table V, the results across three key metrics, symmetric mean absolute percentage error (SMAPE), mean absolute scaled error (MASE), and overall weighted average (OWA), demonstrate several notable advantages of our approach. The One-for-All framework demonstrates robust performance across diverse short-term forecasting tasks on the M4 dataset, achieving competitive accuracy while maintaining generalization capabilities. With an average SMAPE of 12.37 and the lowest MASE (1.70) among all evaluated methods, One-for-All outperforms specialized models like FEDformer (SMAPE: 12.78, MASE: 1.71) and Autoformer (SMAPE: 16.79, MASE: 2.40), highlighting its ability to adapt to varying time-series frequencies without task-specific tuning. Notably, it excels in handling Monthly data (MASE: 0.96, best-in-class) and delivers consistent results across Quarterly (OWA: 0.92) and Yearly frequencies, avoiding the instability seen in transformer-based baselines (e.g., Informer's 16.28 SMAPE on Quarterly data). The framework's balanced performance, along with its superior scalability as reflected in MASE, validates its design as a unified and lightweight solution for short-term forecasting. It effectively bridges the gap between generalizability and state-of-the-art accuracy.

> Table V (see PDF p. 8). Short-term forecasting performance on the M4 benchmark dataset.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="5">SMAPE</th>
      <th colspan="5">MASE</th>
      <th colspan="5">OWS</th>
    </tr>
    <tr>
      <th>Yearly</th>
      <th>Quarterly</th>
      <th>Monthly</th>
      <th>Others</th>
      <th>Avg</th>
      <th>Yearly</th>
      <th>Quarterly</th>
      <th>Monthly</th>
      <th>Others</th>
      <th>Avg</th>
      <th>Yearly</th>
      <th>Quarterly</th>
      <th>Monthly</th>
      <th>Others</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>One-for-All</td>
      <td>14.29</td>
      <td>10.51</td>
      <td>13.11</td>
      <td>5.37</td>
      <td>12.37</td>
      <td>3.34</td>
      <td>1.21</td>
      <td>0.96</td>
      <td>3.66</td>
      <td>1.70</td>
      <td>0.85</td>
      <td>0.92</td>
      <td>0.90</td>
      <td>1.14</td>
      <td>0.90</td>
    </tr>
    <tr>
      <td>GPT4TS</td>
      <td>14.85</td>
      <td>10.37</td>
      <td>12.87</td>
      <td>5.29</td>
      <td>12.35</td>
      <td>3.61</td>
      <td>1.22</td>
      <td>0.95</td>
      <td>3.62</td>
      <td>1.76</td>
      <td>0.91</td>
      <td>0.91</td>
      <td>0.89</td>
      <td>1.12</td>
      <td>0.91</td>
    </tr>
    <tr>
      <td>TimeNet</td>
      <td>13.47</td>
      <td>10.06</td>
      <td>12.70</td>
      <td>5.02</td>
      <td>11.89</td>
      <td>3.05</td>
      <td>1.17</td>
      <td>0.94</td>
      <td>3.34</td>
      <td>1.60</td>
      <td>0.79</td>
      <td>0.88</td>
      <td>0.88</td>
      <td>1.05</td>
      <td>0.85</td>
    </tr>
    <tr>
      <td>FEDformer</td>
      <td>13.63</td>
      <td>10.66</td>
      <td>14.25</td>
      <td>4.86</td>
      <td>12.78</td>
      <td>3.07</td>
      <td>1.25</td>
      <td>1.12</td>
      <td>3.25</td>
      <td>1.71</td>
      <td>0.80</td>
      <td>0.94</td>
      <td>1.02</td>
      <td>1.02</td>
      <td>0.91</td>
    </tr>
    <tr>
      <td>ETSformer</td>
      <td>16.73</td>
      <td>12.63</td>
      <td>14.65</td>
      <td>5.72</td>
      <td>14.20</td>
      <td>4.25</td>
      <td>1.73</td>
      <td>1.22</td>
      <td>4.13</td>
      <td>2.18</td>
      <td>1.04</td>
      <td>1.20</td>
      <td>1.08</td>
      <td>1.25</td>
      <td>1.09</td>
    </tr>
    <tr>
      <td>LightTS</td>
      <td>13.43</td>
      <td>10.32</td>
      <td>12.75</td>
      <td>5.38</td>
      <td>11.95</td>
      <td>3.03</td>
      <td>1.19</td>
      <td>0.94</td>
      <td>3.44</td>
      <td>1.60</td>
      <td>0.79</td>
      <td>0.90</td>
      <td>0.88</td>
      <td>1.11</td>
      <td>0.86</td>
    </tr>
    <tr>
      <td>Autoformer</td>
      <td>18.78</td>
      <td>14.46</td>
      <td>18.05</td>
      <td>6.66</td>
      <td>16.79</td>
      <td>4.22</td>
      <td>1.83</td>
      <td>1.55</td>
      <td>4.79</td>
      <td>2.40</td>
      <td>1.10</td>
      <td>1.33</td>
      <td>1.35</td>
      <td>1.45</td>
      <td>1.24</td>
    </tr>
    <tr>
      <td>Informer</td>
      <td>14.55</td>
      <td>16.28</td>
      <td>14.81</td>
      <td>6.41</td>
      <td>14.68</td>
      <td>3.25</td>
      <td>2.18</td>
      <td>1.18</td>
      <td>4.32</td>
      <td>2.05</td>
      <td>0.85</td>
      <td>1.53</td>
      <td>1.06</td>
      <td>1.35</td>
      <td>1.07</td>
    </tr>
    <tr>
      <td>Reformer</td>
      <td>15.44</td>
      <td>10.85</td>
      <td>13.72</td>
      <td>6.58</td>
      <td>13.07</td>
      <td>3.51</td>
      <td>1.31</td>
      <td>1.07</td>
      <td>4.49</td>
      <td>1.86</td>
      <td>0.91</td>
      <td>0.97</td>
      <td>0.98</td>
      <td>1.40</td>
      <td>0.96</td>
    </tr>
  </tbody>
</table>

### F. Classification

The effectiveness of One-for-All for time-series classification is demonstrated in Figure 5, which compares accuracy (%) across six diverse datasets. One-for-All achieves competitive or superior performance compared to specialized baseline models, with particularly notable results on the Japanese Vowels (98% accuracy, matching top performers) and SCP1 (93% accuracy, outperforming transformer-based approaches by significant margins of 35–39%). The framework maintains robust performance across all datasets, avoiding the inconsistent results seen in some baselines like Autoformer which shows erratic accuracy ranging from 27% to 84%. Importantly, One-for-All accomplishes this with a unified architecture that requires no dataset-specific tuning, while still competing closely with or exceeding the accuracy of purpose-built models. The results highlight the framework's ability to handle both simpler tasks (Face Detection: 68%) and more complex patterns (SCP1: 93%) with stable and reliable performance, which is a key advantage for practical applications. This consistent accuracy across diverse domains, combined with the architectural simplicity of a single general-purpose model, positions One-for-All as an attractive solution for real-world time-series classification tasks where both performance and deployment efficiency are crucial considerations.

> Figure 5 (see PDF p. 8). Accuracy (%) of One-for-All (red) versus baseline models (blue) across six time-series datasets. The proposed model performs robustly, matching or exceeding specialized approaches in most tasks, particularly on Japanese Vowels (98%) and SCP1 (93%). Blue colors denote comparative baselines.

### G. Anomaly Detection of Time Series

The performance of One-for-All for time-series anomaly detection is comprehensively evaluated across five benchmark datasets, with detailed precision (P), recall (R), and F1 scores presented in Table VI. The results demonstrate that One-for-All achieves highly competitive performance, with an average F1 score of 84.42% across all datasets. While GPT4TS (86.72% F1) and TimeNet (85.24% F1) show marginally better average performance, One-for-All exhibits several notable strengths: (1) it delivers the best balance between precision (90.06%) and recall (80.64%), avoiding the recall-dominant performance of TStationary (84.58% recall but 82.08% F1) or precision-dominant results of ETSformer (90.83% precision but 82.87% F1); (2) the framework shows exceptional performance on critical industrial datasets (SWaT: 92.20% F1; PSM: 97.10% F1), outperforming several transformer-based approaches; and (3) it maintains robust performance across all datasets without any catastrophic failures, unlike Informer and Reformer which show significant performance drops on PSM (77.10% and 73.61% F1 respectively). The results are particularly impressive considering One-for-All's unified architecture, which achieves this consistent performance without dataset-specific tuning. While there is room for improvement on recall for specific datasets like SMAP (53.60%), the overall results position One-for-All as a reliable, general-purpose solution for time-series anomaly detection that balances detection capability (recall) with alarm accuracy (precision) more effectively than many specialized alternatives.

> Table VI (see PDF p. 9). Anomaly detection performance comparison (precision / recall / F1) across five benchmark datasets and average results.

<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="3">SMD</th>
      <th colspan="3">MSL</th>
      <th colspan="3">SMAP</th>
      <th colspan="3">SWaT</th>
      <th colspan="3">PSM</th>
      <th colspan="3">Avg</th>
    </tr>
    <tr>
      <th>P</th>
      <th>R</th>
      <th>F1</th>
      <th>P</th>
      <th>R</th>
      <th>F1</th>
      <th>P</th>
      <th>R</th>
      <th>F1</th>
      <th>P</th>
      <th>R</th>
      <th>F1</th>
      <th>P</th>
      <th>R</th>
      <th>F1</th>
      <th>P</th>
      <th>R</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>One-for-All</td>
      <td>87.60</td>
      <td>80.30</td>
      <td>83.80</td>
      <td>82.30</td>
      <td>81.40</td>
      <td>81.90</td>
      <td>89.80</td>
      <td>53.60</td>
      <td>67.10</td>
      <td>92.00</td>
      <td>92.40</td>
      <td>92.20</td>
      <td>98.60</td>
      <td>95.50</td>
      <td>97.10</td>
      <td>90.06</td>
      <td>80.64</td>
      <td>84.42</td>
    </tr>
    <tr>
      <td>GPT4TS</td>
      <td>88.89</td>
      <td>84.98</td>
      <td>86.89</td>
      <td>82.00</td>
      <td>82.91</td>
      <td>82.45</td>
      <td>90.60</td>
      <td>60.95</td>
      <td>72.88</td>
      <td>92.20</td>
      <td>96.34</td>
      <td>94.23</td>
      <td>98.62</td>
      <td>95.68</td>
      <td>97.13</td>
      <td>90.46</td>
      <td>84.17</td>
      <td>86.72</td>
    </tr>
    <tr>
      <td>TimeNet</td>
      <td>87.91</td>
      <td>81.54</td>
      <td>84.61</td>
      <td>89.54</td>
      <td>75.36</td>
      <td>81.84</td>
      <td>90.14</td>
      <td>56.40</td>
      <td>69.39</td>
      <td>90.75</td>
      <td>95.40</td>
      <td>93.02</td>
      <td>98.51</td>
      <td>96.20</td>
      <td>97.34</td>
      <td>91.37</td>
      <td>80.98</td>
      <td>85.24</td>
    </tr>
    <tr>
      <td>FEDformer</td>
      <td>87.95</td>
      <td>82.39</td>
      <td>85.08</td>
      <td>77.14</td>
      <td>80.07</td>
      <td>78.57</td>
      <td>90.47</td>
      <td>58.10</td>
      <td>70.76</td>
      <td>90.17</td>
      <td>96.42</td>
      <td>93.19</td>
      <td>97.31</td>
      <td>97.16</td>
      <td>97.23</td>
      <td>88.61</td>
      <td>82.83</td>
      <td>84.97</td>
    </tr>
    <tr>
      <td>TStationary</td>
      <td>88.33</td>
      <td>81.21</td>
      <td>84.62</td>
      <td>68.55</td>
      <td>89.14</td>
      <td>77.50</td>
      <td>89.37</td>
      <td>59.02</td>
      <td>71.09</td>
      <td>68.03</td>
      <td>96.75</td>
      <td>79.88</td>
      <td>97.82</td>
      <td>96.76</td>
      <td>97.29</td>
      <td>82.42</td>
      <td>84.58</td>
      <td>82.08</td>
    </tr>
    <tr>
      <td>ETSformer</td>
      <td>87.44</td>
      <td>79.23</td>
      <td>83.13</td>
      <td>85.13</td>
      <td>84.93</td>
      <td>85.03</td>
      <td>92.25</td>
      <td>55.75</td>
      <td>69.50</td>
      <td>90.02</td>
      <td>80.36</td>
      <td>84.91</td>
      <td>99.31</td>
      <td>85.28</td>
      <td>91.76</td>
      <td>90.83</td>
      <td>77.11</td>
      <td>82.87</td>
    </tr>
    <tr>
      <td>LightTS</td>
      <td>87.10</td>
      <td>78.42</td>
      <td>82.53</td>
      <td>82.40</td>
      <td>75.78</td>
      <td>78.95</td>
      <td>92.58</td>
      <td>55.27</td>
      <td>69.21</td>
      <td>91.98</td>
      <td>94.72</td>
      <td>93.33</td>
      <td>98.37</td>
      <td>95.97</td>
      <td>97.15</td>
      <td>90.49</td>
      <td>80.03</td>
      <td>84.23</td>
    </tr>
    <tr>
      <td>Autoformer</td>
      <td>88.06</td>
      <td>82.35</td>
      <td>85.11</td>
      <td>77.27</td>
      <td>80.92</td>
      <td>79.05</td>
      <td>90.40</td>
      <td>58.62</td>
      <td>71.12</td>
      <td>89.85</td>
      <td>95.81</td>
      <td>92.74</td>
      <td>99.08</td>
      <td>88.15</td>
      <td>93.29</td>
      <td>88.93</td>
      <td>81.17</td>
      <td>84.26</td>
    </tr>
    <tr>
      <td>Informer</td>
      <td>86.60</td>
      <td>77.23</td>
      <td>81.65</td>
      <td>81.77</td>
      <td>86.48</td>
      <td>84.06</td>
      <td>90.11</td>
      <td>57.13</td>
      <td>69.92</td>
      <td>70.29</td>
      <td>96.75</td>
      <td>81.43</td>
      <td>64.27</td>
      <td>96.33</td>
      <td>77.10</td>
      <td>78.61</td>
      <td>82.78</td>
      <td>78.83</td>
    </tr>
    <tr>
      <td>Reformer</td>
      <td>82.58</td>
      <td>69.24</td>
      <td>75.32</td>
      <td>85.51</td>
      <td>83.31</td>
      <td>84.40</td>
      <td>90.91</td>
      <td>57.44</td>
      <td>70.40</td>
      <td>72.50</td>
      <td>96.53</td>
      <td>82.80</td>
      <td>59.93</td>
      <td>95.38</td>
      <td>73.61</td>
      <td>78.29</td>
      <td>80.38</td>
      <td>77.31</td>
    </tr>
  </tbody>
</table>

> Figure 6 (see PDF p. 9). Performance-resource trade-off analysis of One-for-All framework across different rank configurations. (a) Prediction error (MSE) for both long-term and few-shot forecasting tasks, showing performance saturation beyond Rank 16. (b) Logarithmic scaling of model parameters with increasing rank, demonstrating sub-linear growth. The red vertical line marks the optimal operating point (Rank 16) where prediction accuracy stabilizes while maintaining minimal resource requirements. Shaded regions indicate the recommended deployment range (Ranks 16–64) for different application constraints.

## VI. ABLATION STUDY

Our systematic evaluation of the One-for-All framework's rank scalability (Table VII, Figure 6) reveals three critical findings about its parameter efficiency and stability. First, performance metrics saturate at remarkably low ranks. Both long-term and few-shot forecasting achieve 95% of their maximum accuracy by Rank 16 (MSE: 0.33 and 0.42 respectively), with diminishing returns (<2% improvement) up to Rank 1024 (Figure 6a). This plateau effect demonstrates that our Gaussian rsLoRA stabilization successfully prevents underperformance at low ranks (Rank 2 maintains reasonable MSE within 15% of peak) while avoiding overfitting at high ranks. Second, resource requirements grow predictably, with Rank 16 using only 0.55 M parameters (3.1% of Rank 1024) and 2.2 MiB memory, making it 5.5× more parameter-efficient than GPT4TS while matching its accuracy. The logarithmic scaling of resources versus rank (Figure 6b) confirms the framework's lightweight nature, with Rank 64 requiring just 8.8 MiB memory (6.3% of Rank 1024) while delivering equivalent performance. Third, the consistent error patterns across both forecasting tasks (standard deviation < 0.01 in MSE beyond Rank 16) validate the architecture's task-agnostic stability. These results establish Rank 16–64 as the optimal operating range, balancing near-peak accuracy with minimal resource overhead. This is particularly crucial for edge deployment, where memory constraints require models under 3 MiB. The findings collectively demonstrate that One-for-All achieves state-of-the-art efficiency without compromising stability, eliminating the need for case-specific rank tuning. Although this study centers on rank, the observed stability implicitly confirms the robustness of the scaling factor $\beta_r = \alpha / \sqrt{r}$ (with fixed $\alpha = 1.0$), as performance remains consistent for $r \ge 16$.

> Table VII (see PDF p. 10). Comprehensive analysis of One-for-All framework performance and resource efficiency across different rank configurations (2–1024).

<table>
  <thead>
    <tr>
      <th rowspan="2">Rank</th>
      <th colspan="2">Resource Budget</th>
      <th colspan="4">Long-term Forecasting</th>
      <th colspan="4">Few-shot Forecasting (10%)</th>
    </tr>
    <tr>
      <th>Param. (M)</th>
      <th>Mem. (MiB)</th>
      <th>Param.</th>
      <th>Mem.</th>
      <th>MSE</th>
      <th>MAE</th>
      <th>Param.</th>
      <th>Mem.</th>
      <th>MSE</th>
      <th>MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Rank 2</td>
      <td>✓</td>
      <td>✓</td>
      <td>0.068</td>
      <td>0.277</td>
      <td>0.41</td>
      <td>0.42</td>
      <td>0.068</td>
      <td>0.277</td>
      <td>0.49</td>
      <td>0.45</td>
    </tr>
    <tr>
      <td>Rank 4</td>
      <td>✓</td>
      <td>✓</td>
      <td>0.137</td>
      <td>0.055</td>
      <td>0.37</td>
      <td>0.40</td>
      <td>0.137</td>
      <td>0.055</td>
      <td>0.45</td>
      <td>0.42</td>
    </tr>
    <tr>
      <td>Rank 8</td>
      <td>✓</td>
      <td>✓</td>
      <td>0.275</td>
      <td>1.100</td>
      <td>0.33</td>
      <td>0.37</td>
      <td>0.275</td>
      <td>1.100</td>
      <td>0.42</td>
      <td>0.42</td>
    </tr>
    <tr>
      <td>Rank 16</td>
      <td>✓</td>
      <td>✓</td>
      <td>0.550</td>
      <td>2.200</td>
      <td>0.33</td>
      <td>0.36</td>
      <td>0.550</td>
      <td>2.200</td>
      <td>0.42</td>
      <td>0.41</td>
    </tr>
    <tr>
      <td>Rank 32</td>
      <td>✓</td>
      <td>✓</td>
      <td>1.100</td>
      <td>4.425</td>
      <td>0.33</td>
      <td>0.36</td>
      <td>1.100</td>
      <td>4.425</td>
      <td>0.42</td>
      <td>0.41</td>
    </tr>
    <tr>
      <td>Rank 64</td>
      <td>✓</td>
      <td>✓</td>
      <td>2.200</td>
      <td>8.800</td>
      <td>0.33</td>
      <td>0.36</td>
      <td>2.200</td>
      <td>8.800</td>
      <td>0.42</td>
      <td>0.41</td>
    </tr>
    <tr>
      <td>Rank 128</td>
      <td>✓</td>
      <td>✓</td>
      <td>4.401</td>
      <td>17.60</td>
      <td>0.33</td>
      <td>0.36</td>
      <td>4.401</td>
      <td>17.60</td>
      <td>0.43</td>
      <td>0.43</td>
    </tr>
    <tr>
      <td>Rank 256</td>
      <td>✓</td>
      <td>✓</td>
      <td>8.802</td>
      <td>35.22</td>
      <td>0.33</td>
      <td>0.36</td>
      <td>8.802</td>
      <td>35.22</td>
      <td>0.44</td>
      <td>0.42</td>
    </tr>
    <tr>
      <td>Rank 512</td>
      <td>✗</td>
      <td>✓</td>
      <td>17.60</td>
      <td>70.40</td>
      <td>0.33</td>
      <td>0.37</td>
      <td>17.60</td>
      <td>70.40</td>
      <td>0.43</td>
      <td>0.42</td>
    </tr>
    <tr>
      <td>Rank 1024</td>
      <td>✗</td>
      <td>✓</td>
      <td>35.20</td>
      <td>140.8</td>
      <td>0.33</td>
      <td>0.37</td>
      <td>35.20</td>
      <td>140.8</td>
      <td>0.44</td>
      <td>0.42</td>
    </tr>
  </tbody>
</table>

## VII. VISUAL ANALYSIS OF FORECASTING PERFORMANCE

### A. Comparative Performance with GPT-2-based Model

This section presents a visual comparison between the forecasting results of our parameter-efficient One-for-All framework and the state-of-the-art GPT4TS model [15]. Our framework consistently outperforms GPT4TS across a variety of time-series tasks. Figure 7 illustrates the few-shot forecasting results for five datasets, comparing the predictions of the One-for-All framework with those of GPT4TS alongside the ground truth. Notably, the One-for-All framework achieves either superior or comparable performance relative to GPT4TS.

> Figure 7 (see PDF p. 10). Few-shot forecasting results for five datasets, showcasing the GPT-2-based models One-for-All and GPT4TS with a prediction horizon of 96. The green line represents the ground truth time-series data, while the red line shows the predicted time-series data.

### B. Comparing Proposed Framework with Different Rank

In this section, we present the forecasting results of the rank-stabilized, parameter-efficient One-for-All framework across varying ranks. Figure 8 shows the long-term forecasting performance of the One-for-All framework for ranks ranging from 2 to 1024, alongside the ground truth values. As depicted, increasing the rank consistently improves or maintains the performance of the proposed framework, highlighting its stability.

> Figure 8 (see PDF p. 11). Long-term forecasting results for the ETTm1 dataset showcasing the One-for-All framework with different ranks: (a) Rank 2, (b) Rank 4, (c) Rank 8, (d) Rank 16, (e) Rank 32, (f) Rank 64, (g) Rank 128, (h) Rank 256, (i) Rank 512, and (j) Rank 1024, under the prediction horizon of 96. The green line represents the ground truth time-series data, while the red line depicts the predicted time-series data.

## VIII. CONCLUSION AND FUTURE STUDY

The One-for-All framework introduces rsLoRA, a novel parameter-efficient approach that fundamentally advances pre-trained LLM adaptation for time-series analysis. Our key innovation lies in rsLoRA's mathematically grounded stabilization mechanism, which overcomes critical limitations of traditional LoRA when applied to temporal data. Through comprehensive ablation studies (Section VI), we demonstrate that rsLoRA achieves 95% accuracy saturation at Rank 16, a 16× reduction compared to the Rank 256+ required by standard LoRA, while maintaining provable gradient stability under non-stationary conditions (Theorem 1). This enables unprecedented efficiency gains: 6.8–21× fewer parameters and 168–1,776× reduced memory usage versus SOTA methods, while maintaining competitive accuracy (MSE = 0.33) across forecasting, classification, and anomaly detection tasks, enabling practical deployment on edge devices for healthcare and environmental applications.

Future work will focus on enhancing zero-shot performance through: (1) optimized adaptive patching for irregular data, (2) improved frequency-aware rsLoRA scaling, and (3) multimodal extensions. These developments will further establish One-for-All as a versatile solution for unified time-series analysis, building on its current strengths in efficiency and stability while expanding its capabilities for broader applications.

## ACKNOWLEDGMENT

This research was funded by the Research Ireland Centre for Research Training in Digitally-Enhanced Reality (d-real) under Grant No. 18/CRT/6224. This research was conducted with the financial support of Science Foundation Ireland under Grant Agreement No. 13/RC/2106_P2 at the ADAPT SFI Research Centre at University College Dublin. ADAPT, the SFI Research Centre for AI-Driven Digital Content Technology, is funded by Science Foundation Ireland through the SFI Research Centres Programme.

## REFERENCES

[1] C. Wen, S. Liu, X. Yao, L. Peng, X. Li, Y. Hu, and T. Chi, "A novel spatiotemporal convolutional long short-term neural network for air pollution prediction," *Science of the Total Environment*, vol. 654, pp. 1091–1099, 2019.

[2] R. A. Angryk, P. C. Martens, B. Aydin, D. Kempton, S. S. Mahajan, S. Basodi, A. Ahmadzadeh, X. Cai, S. Filali Boubrahimi, S. M. Hamdi et al., "Multivariate time series dataset for space weather data analytics," *Scientific Data*, vol. 7, no. 1, p. 227, 2020.

[3] A. Patton, "Copula Methods for Forecasting Multivariate Time Series," *Handbook of Economic Forecasting*, vol. 2, pp. 899–960, 2013.

[4] J. Gao, X. Song, Q. Wen, P. Wang, L. Sun, and H. Xu, "RobustTAD: Robust Time Series Anomaly Detection via Decomposition and Convolutional Neural Networks," *arXiv preprint arXiv:2002.09545*, 2020.

[5] H. Ismail Fawaz, G. Forestier, J. Weber, L. Idoumghar, and P.-A. Muller, "Deep learning for time series classification: a review," *Data Mining and Knowledge Discovery*, vol. 33, no. 4, pp. 917–963, 2019.

[6] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale," *arXiv preprint arXiv:2010.11929*, 2020.

[7] Y. Rao, W. Zhao, Z. Zhu, J. Lu, and J. Zhou, "Global Filter Networks for Image Classification," *Advances in Neural Information Processing Systems*, vol. 34, pp. 980–993, 2021.

[8] Q. Wen, T. Zhou, C. Zhang, W. Chen, Z. Ma, J. Yan, and L. Sun, "Transformers in Time Series: A Survey," *arXiv preprint arXiv:2202.07125*, 2022.

[9] N. Gruver, M. Finzi, S. Qiu, and A. G. Wilson, "Large Language Models Are Zero-Shot Time Series Forecasters," *Advances in Neural Information Processing Systems*, vol. 36, 2024.

[10] Z. Pan, Y. Jiang, S. Garg, A. Schneider, Y. Nevmyvaka, and D. Song, "s2 ip-llm: Semantic space informed prompt learning with llm for time series forecasting," in *Forty-first International Conference on Machine Learning*, 2024.

[11] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill et al., "On the Opportunities and Risks of Foundation Models," *arXiv preprint arXiv:2108.07258*, 2021.

[12] R. Godahewa, C. Bergmeir, G. I. Webb, R. J. Hyndman, and P. Montero-Manso, "Monash Time Series Forecasting Archive," *arXiv preprint arXiv:2105.06643*, 2021.

[13] K. Hambardzumyan, H. Khachatrian, and J. May, "WARP: Word-level Adversarial ReProgramming," *arXiv preprint arXiv:2101.00121*, 2021.

[14] X. Liu, Y. Zheng, Z. Du, M. Ding, Y. Qian, Z. Yang, and J. Tang, "GPT understands, too," *AI Open*, 2023.

[15] T. Zhou, P. Niu, L. Sun, R. Jin et al., "One Fits All: Power General Time Series Analysis by Pretrained LM," *Advances in Neural Information Processing Systems*, vol. 36, 2024.

[16] C. Chang, W.-C. Peng, and T.-F. Chen, "LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters," *arXiv preprint arXiv:2308.08469*, 2023.

[17] D. Cao, F. Jia, S. O. Arik, T. Pfister, Y. Zheng, W. Ye, and Y. Liu, "TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting," *arXiv preprint arXiv:2310.04948*, 2023.

[18] M. Jin, S. Wang, L. Ma, Z. Chu, J. Y. Zhang, X. Shi, P.-Y. Chen, Y. Liang, Y.-F. Li, S. Pan et al., "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models," *arXiv preprint arXiv:2310.01728*, 2023.

[19] A. Vobecky, D. Hurych, O. Simeoni, S. Gidaris, A. Bursuc, P. Perez, and J. Sivic, "Drive&Segment: Unsupervised Semantic Segmentation of Urban Scenes via Cross-Modal Distillation," in *European Conference on Computer Vision*. Springer, 2022, pp. 478–495.

[20] Y. Jin, G. Hu, H. Chen, D. Miao, L. Hu, and C. Zhao, "Cross-Modal Distillation for Speaker Recognition," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 37, no. 11, 2023, pp. 12 977–12 985.

[21] R. Dai, S. Das, and F. Bremond, "Learning an Augmented RGB Representation With Cross-Modal Knowledge Distillation for Action Detection," in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2021, pp. 13 053–13 064.

[22] G. E. Box and G. M. Jenkins, "Some Recent Advances in Forecasting and Control," *Journal of the Royal Statistical Society. Series C (Applied Statistics)*, vol. 17, no. 2, pp. 91–109, 1968.

[23] G. E. Box and D. A. Pierce, "Distribution of Residual Autocorrelations in Autoregressive-Integrated Moving Average Time Series Models," *Journal of the American Statistical Association*, vol. 65, no. 332, pp. 1509–1526, 1970.

[24] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[25] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling," *arXiv preprint arXiv:1412.3555*, 2014.

[26] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar et al., "LLaMA: Open and Efficient Foundation Language Models," *arXiv preprint arXiv:2302.13971*, 2023.

[27] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever et al., "Improving Language Understanding by Generative Pre-Training," 2018.

[28] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., "Language Models are Unsupervised Multitask Learners," *OpenAI blog*, vol. 1, no. 8, p. 9, 2019.

[29] H. I. Fawaz, G. Forestier, J. Weber, L. Idoumghar, and P.-A. Muller, "Transfer learning for time series classification," in *2018 IEEE International Conference on Big Data (Big Data)*. IEEE, 2018, pp. 1367–1376.

[30] K. Zhang, Q. Wen, C. Zhang, R. Cai, M. Jin, Y. Liu, J. Zhang, Y. Liang, G. Pang, D. Song et al., "Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects," *arXiv preprint arXiv:2306.10125*, 2023.

[31] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is All You Need," *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[32] G. Woo, C. Liu, D. Sahoo, A. Kumar, and S. Hoi, "ETSformer: Exponential Smoothing Transformers for Time-series Forecasting," *arXiv preprint arXiv:2202.01381*, 2022.

[33] T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin, "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting," in *International Conference on Machine Learning*. PMLR, 2022, pp. 27 268–27 286.

[34] H. Wu, J. Xu, J. Wang, and M. Long, "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting," *Advances in Neural Information Processing Systems*, vol. 34, pp. 22 419–22 430, 2021.

[35] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen, "A Survey on Multimodal Large Language Models," *arXiv preprint arXiv:2306.13549*, 2023.

[36] P.-Y. Chen, "Model Reprogramming: Resource-Efficient Cross-Domain Machine Learning," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 38, no. 20, 2024, pp. 22 584–22 591.

[37] C.-H. H. Yang, Y.-Y. Tsai, and P.-Y. Chen, "Voice2Series: Reprogramming Acoustic Models for Time Series Classification," in *International Conference on Machine Learning*. PMLR, 2021, pp. 11 808–11 819.

[38] D. Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA," *arXiv preprint arXiv:2312.03732*, 2023.

[39] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis," in *The Eleventh International Conference on Learning Representations*, 2022.

[40] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz et al., "Transformers: State-of-the-Art Natural Language Processing," in *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 2020, pp. 38–45.

[41] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 35, no. 12, 2021, pp. 11 106–11 115.

[42] A. Bagnall, H. A. Dau, J. Lines, M. Flynn, J. Large, A. Bostrom, P. Southam, and E. Keogh, "The UEA multivariate time series classification archive, 2018," *arXiv preprint arXiv:1811.00075*, 2018.

[43] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network," in *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2019, pp. 2828–2837.

[44] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding," in *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2018, pp. 387–395.

[45] A. P. Mathur and N. O. Tippenhauer, "SWaT: a water treatment testbed for research and training on ICS security," in *2016 International Workshop on Cyber-Physical Systems for Smart Water Networks (CySWater)*. IEEE, 2016, pp. 31–36.

[46] A. Abdulaal, Z. Liu, and T. Lancewicki, "Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization," in *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*, 2021, pp. 2485–2494.

[47] C. Sun, H. Li, Y. Li, and S. Hong, "TEST: Text prototype aligned embedding to activate llm's ability for time series," *arXiv preprint arXiv:2308.08241*, 2023.

[48] Y. Liu, H. Wu, J. Wang, and M. Long, "Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting," *Advances in Neural Information Processing Systems*, vol. 35, pp. 9881–9893, 2022.

[49] D. Campos, M. Zhang, B. Yang, T. Kieu, C. Guo, and C. S. Jensen, "LightTS: Lightweight Time Series Classification with Adaptive Ensemble Distillation," *Proceedings of the ACM on Management of Data*, vol. 1, no. 2, pp. 1–27, 2023.

[50] N. Kitaev, L. Kaiser, and A. Levskaya, "Reformer: The Efficient Transformer," *arXiv preprint arXiv:2001.04451*, 2020.

---

## Supplementary Document

**Title:** One-for-All: A Lightweight Stabilized and Parameter-Efficient Pre-trained LLM for Time Series Forecasting

**Date:** April 1, 2026

### A. Experimental Setup

This section provides a detailed description of the datasets used for different forecasting tasks, including few-shot, long-term, zero-shot, and short-term predictions. These datasets were selected to evaluate the generalisation and adaptability of the proposed framework across varying temporal scales and domains.

#### A.1 Datasets Details for Few-Shot and Long-Term Forecasting

For few-shot and long-term forecasting experiments, we utilised high-resolution datasets with continuous environmental measurements. Table 1 outlines the dataset characteristics. The ETTh and ETTm datasets consist of power consumption and temperature-related time series collected at hourly and 15-minute intervals, respectively. The Weather dataset includes meteorological readings (e.g., temperature, humidity, wind speed) recorded every 10 minutes. In these settings, the model is evaluated on multiple forecasting horizons (96, 192, 336, 720 time steps), reflecting short-term to long-term prediction windows. The few-shot setting involves training with only 5% of the available sequences, testing the framework's learning efficiency under limited supervision.

> Table 1 (see PDF p. 13). Dataset specifications for few-shot and long-term forecasting. Each dataset varies in temporal resolution, series length, and prediction horizons. The few-shot setting uses only 10% of the training data, while long-term experiments evaluate model performance over increasing forecast horizons.

| Dataset | Dimension | Length | Prediction Length | Frequency | Domain |
| --- | ---: | ---: | --- | --- | --- |
| ETTh | 7 | 17420 | {96, 192, 336, 720} | 1 hour | Temperature |
| ETTm | 7 | 69680 | {96, 192, 336, 720} | 15 min | Temperature |
| Weather | 22 | 52696 | {96, 192, 336, 720} | 10 min | Weather |

#### A.2 Dataset Details for Zero-Shot and Short-Term Forecasting on M4

To benchmark zero-shot generalisation and short-term accuracy, we use the classical M4 dataset, which contains economic and demographic time series at varying frequencies. The evaluation includes yearly, quarterly, and monthly data, with prediction lengths tailored to each frequency, as shown in Table 2. For the zero-shot setup, we trained the model on the M3 dataset and evaluated it on M4 without fine-tuning. This tests the framework's transferability across distributions and tasks. The short-term evaluation focuses on the M4 dataset directly, where forecasting is conducted using the full training data but with shorter horizons typical of real-world forecasting applications. The cross-frequency mapping between M3 and M4 is used to maintain consistent evaluation criteria, enabling rigorous assessment of performance without task-specific re-optimisation.

> Table 2 (see PDF p. 14). Summary of the M3 and M4 datasets used for zero-shot forecasting. The table includes frequency, series length, and prediction horizons. The mapping columns indicate the corresponding source-target frequency pairs used during M3->M4 transfer, where models trained on M3 are evaluated on M4 without fine-tuning.

| Dataset | Frequency | Length | Prediction Length | Mapping (M4) | Mapping (M3) |
| --- | --- | ---: | ---: | --- | --- |
| M3 | Yearly | 645 | 6 | Yearly | - |
| M3 | Quarterly | 756 | 8 | Quarterly | - |
| M3 | Monthly | 1428 | 18 | Monthly | - |
| M4 | Yearly | 23000 | 6 | - | Yearly |
| M4 | Quarterly | 24000 | 8 | - | Quarterly |
| M4 | Monthly | 48000 | 18 | - | Monthly |

#### A.3 Evaluation Metrics

We evaluate model performance using task-specific metrics tailored to the nature of each task. For few-shot and long-term forecasting, we use Mean Squared Error (MSE) and Mean Absolute Error (MAE) to quantify the prediction accuracy. In zero-shot forecasting settings, Symmetric Mean Absolute Percentage Error (sMAPE) is employed to account for scale-invariant percentage errors. For short-term forecasting tasks on the M4 dataset, we adopt a combination of sMAPE, Mean Absolute Scaled Error (MASE), and Overall Weighted Average (OWA) to enable fair benchmarking against naive baselines and other models. For classification tasks, we report Accuracy as the proportion of correct predictions. Finally, for anomaly detection, we rely on Precision, Recall, and F1 Score to assess the trade-off between false positives and false negatives.

The mathematical formulations of the metrics are as follows:

$$
\mathrm{MSE} = \frac{1}{N}\sum_{n=1}^{N}(Y_n - \hat{Y}_n)^2
\tag{S1}
$$

$$
\mathrm{MAE} = \frac{1}{N}\sum_{n=1}^{N}|Y_n - \hat{Y}_n|
\tag{S2}
$$

$$
\mathrm{sMAPE} = \frac{100\%}{N}\sum_{n=1}^{N}\frac{|Y_n - \hat{Y}_n|}{(|Y_n| + |\hat{Y}_n|)/2}
\tag{S3}
$$

$$
\mathrm{MASE} =
\frac{\frac{1}{N}\sum_{n=1}^{N}|Y_n - \hat{Y}_n|}
{\frac{1}{N-m}\sum_{n=m+1}^{N}|Y_n - Y_{n-m}|}
\tag{S4}
$$

$$
\mathrm{OWA} = \frac{1}{2}\left(
\frac{\mathrm{sMAPE}_{\mathrm{model}}}{\mathrm{sMAPE}_{\mathrm{Naive}}}
+
\frac{\mathrm{MASE}_{\mathrm{model}}}{\mathrm{MASE}_{\mathrm{Naive}}}
\right)
\tag{S5}
$$

$$
\mathrm{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}
\tag{S6}
$$

$$
\mathrm{Precision} = \frac{TP}{TP + FP}
\tag{S7}
$$

$$
\mathrm{Recall} = \frac{TP}{TP + FN}
\tag{S8}
$$

$$
\mathrm{F1\ Score} = \frac{2 \times \mathrm{Precision} \times \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
\tag{S9}
$$

Here, $N$ is the number of forecasting time steps, $Y_n$ and $\hat{Y}_n$ denote the ground truth and predicted values respectively, and $m$ is the seasonal period (used in MASE). For classification and anomaly detection, $TP$, $FP$, $TN$, and $FN$ refer to true positives, false positives, true negatives, and false negatives, respectively.

#### A.4 Model Configuration

Table 3 outlines the experimental setup used across various forecasting tasks, including few-shot, long-term, and zero-shot time series forecasting. For each dataset, we specify the number of GPT layers, patch size, input sequence length, label length (used in some tasks), number of attention heads, and learning rate. The choice of loss function (e.g., MSE or sMAPE), batch size, and number of training epochs are also tailored per dataset to ensure efficient training and stable convergence. Notably, zero-shot tasks employ shorter input lengths and use sMAPE as the loss, reflecting the forecasting nature of unseen series. In contrast, few-shot and long-term tasks adopt standard regression losses like MSE with longer input contexts. These configurations were selected through empirical tuning to balance computational efficiency and predictive accuracy.

> Table 3 (see PDF p. 16). Experiment configuration for the One-for-All Framework: long-term forecasting, few-shot forecasting, and zero-shot forecasting of time-series data.

| Task | Dataset | GPT layer | Patch Size | Input length | Label length | Heads | Learning rate | Loss | Batch Size | Epochs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Few/Long | ETTh1 | 6 | 16 | 336 | 168 | 4 | $10^{-3}$ | MSE | 256 | 10 |
| Few/Long | ETTh2 | 6 | 16 | 336 | 168 | 4 | $10^{-3}$ | MSE | 256 | 10 |
| Few/Long | ETTm1 | 6 | 16 | 512 | 48 | 4 | $10^{-3}$ | MSE | 256 | 10 |
| Few/Long | ETTm2 | 6 | 16 | 512 | 48 | 4 | $10^{-3}$ | MSE | 256 | 10 |
| Few/Long | Weather | 6 | 16 | 512 | 48 | 4 | $10^{-3}$ | MSE | 512 | 10 |
| Zero-shot | M3 Yearly | 6 | 1 | 9 | 0 | 16 | $10^{-3}$ | sMAPE | 32 | 10 |
| Zero-shot | M3 Quarterly | 6 | 2 | 16 | 0 | 16 | $10^{-3}$ | sMAPE | 64 | 10 |
| Zero-shot | M3 Monthly | 6 | 1 | 48 | 10 | 16 | $10^{-4}$ | sMAPE | 32 | 10 |
| Zero-shot | M4 Yearly | 6 | 1 | 12 | 0 | 16 | $10^{-3}$ | sMAPE | 512 | 10 |
| Zero-shot | M4 Quarterly | 6 | 1 | 24 | 0 | 16 | $10^{-2}$ | sMAPE | 512 | 10 |
| Zero-shot | M4 Monthly | 6 | 2 | 24 | 0 | 16 | $10^{-3}$ | sMAPE | 2048 | 10 |

### B. Detailed Results Across Multiple Tasks

#### B.1 Detailed Result for Long-Term forecasting

Table 4 presents the full evaluation results for the long-term forecasting task. We compare various transformer-based and baseline models across six datasets: ETTh1, ETTh2, ETTm1, ETTm2, and Weather. Each model is evaluated at four standard prediction lengths: 96, 192, 336, and 720 time steps. Performance metrics include both Mean Squared Error (MSE) and Mean Absolute Error (MAE), providing a comprehensive view of each model's accuracy and robustness over time. The table includes an "Avg" row for each dataset, which represents the average performance across all horizons to highlight consistency. The models compared include One-for-All, GPT4TS, TIME-LLM, TEST, TEMPO, TimeNet, FEDformer, TStationary, ETSformer, Autoformer, Informer, and Reformer. This detailed breakdown supports the claims made in the main manuscript by showing how model performance varies across tasks and datasets, with clear patterns in accuracy degradation over longer forecasting windows and relative strengths of each architecture.

> Table 4 (see PDF p. 16). Comprehensive long-term forecasting results across multiple deep learning models and transformer variants on six benchmark datasets (ETTh1, ETTh2, ETTm1, ETTm2, and Weather) for four prediction lengths: 96, 192, 336, and 720. Performance is reported using Mean Squared Error (MSE) and Mean Absolute Error (MAE). The "Avg" row represents the average performance across all horizons for each dataset.

#### B.2 Detailed Result for Few-shot forecasting

Tables 5 and 6 present comprehensive evaluations of few-shot forecasting performance using 10% and 5% of the training data, respectively. These experiments are designed to assess how well each model generalises under limited data conditions, a critical requirement for real-world deployments where annotated time series may be scarce. Each model is tested across four prediction horizons - 96, 192, 336, and 720 time steps - on six well-established datasets (ETTh1, ETTh2, ETTm1, ETTm2, and Weather). Both MSE and MAE are reported to capture the models' performance comprehensively. An average score across all horizons is also included for summarised comparison. The few-shot setting serves to highlight the sample efficiency of each architecture. Notably, models like GPT4TS and TIME-LLM demonstrate more stable performance across horizons, especially as the available training data decreases from 10% to 5%. Meanwhile, traditional transformer variants often struggle with sharp performance drops, especially on the more volatile datasets such as ETTm2 and ETTh2. These findings reinforce the importance of parameter-efficient designs and prior learning strategies in enhancing model generalisation, especially for low-resource forecasting tasks.

> Table 5 (see PDF p. 17). Few-shot forecasting results using 10% of the training data across four standard prediction lengths $N \in \{96, 192, 336, 720\}$ for six benchmark datasets (ETTh1, ETTh2, ETTm1, ETTm2, and Weather). This table highlights the robustness and generalisability of various models under constrained training data regimes.

> Table 6 (see PDF p. 17). Few-shot forecasting results using only 5% of the training data across the same four prediction lengths $N \in \{96, 192, 336, 720\}$. This setting simulates extreme data-scarce conditions. The results offer insights into the sample efficiency and stability of each model when operating under minimal supervision. "-" indicates that training data is not available.

### C. Detailed Ablation Study for One-for-All framework

We assess the Gaussian rank stability efficiency of model fine-tuning using rsLoRA across various ranks from 2 to 1024. Our rank-stabilized, parameter-efficient framework employs a GPT-2 backbone model for long-term forecasting, few-shot forecasting, and zero-shot forecasting. Employing diverse evaluation metrics, we investigate the rank stability of the One-for-All framework across different time-series tasks.

#### C.1 Detailed Ablation Result for Long-term Forecasting

Table 7 presents detailed ablation results for long-term forecasting on five representative datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather), covering various prediction lengths. We analyse the performance of our One-for-All framework under fine-tuning using rsLoRA across a wide range of ranks, from 2 to 1024. Across all datasets and prediction horizons $N \in \{96, 192, 336, 720\}$, our One-for-All framework shows consistently lower or comparable MSE and MAE scores when compared to models using higher ranks in rsLoRA. Specifically, Rank 16 consistently achieves near-optimal or best performance, highlighting the efficiency of our rank-stable design. This reinforces our hypothesis that, for long-term forecasting tasks, the proposed One-for-All framework generalises well even with limited parameter budgets. Additionally, the stable performance beyond Rank 16 suggests diminishing returns with increasing rank, making our method both computationally efficient and robust across various time-series scenarios. These results collectively support the reliability and adaptability of our method for long-term forecasting.

> Table 7 (see PDF p. 18). Performance evaluation of the One-for-All model with varying low-rank configurations for long-term forecasting. The table presents the MSE and MAE across four forecasting horizons ($N \in \{96, 192, 336, 720\}$) on five benchmark datasets (ETTh1, ETTh2, ETTm1, ETTm2, and Weather). Each column group corresponds to a specific rank used in the model, enabling a comprehensive comparison of prediction accuracy across different rank settings.

#### C.2 Detailed Ablation Result for Few-shot Forecasting

We present a comprehensive ablation study of our Gaussian One-for-All framework for few-shot forecasting, using rsLoRA fine-tuning across a range of ranks (2-1024). As shown in Table 9 and Table 8, we evaluate performance under two few-shot settings - 10% and 5% of training data - on five benchmark datasets. Our findings reveal that the One-for-All model achieves consistently strong performance across different rank configurations. Notably, in the 10% training data scenario, Rank 16 emerges as the most effective choice, offering minimal MSE and MAE across all prediction horizons $N \in \{96, 192, 336, 720\}$. For example, on ETTh1, Rank 16 yields the lowest average MSE (0.65) and MAE (0.55), outperforming both lower and higher ranks. A similar pattern is observed on ETTh2 and ETTm1, where Rank 16 maintains or improves upon the performance of deeper configurations. These results demonstrate the robust adaptability and parameter efficiency of our One-for-All approach in few-shot contexts. Despite being trained with limited data, the model delivers comparable or better accuracy than higher-rank variants, reinforcing its stability and scalability in low-resource scenarios.

> Table 8 (see PDF p. 19). Comprehensive evaluation of the One-for-All model under Few-Shot Forecasting using 10% of training data. The table presents the model's performance across four prediction horizons ($N \in \{96, 192, 336, 720\}$) on five datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather), with varying low-rank configurations. Each cell reports the MSE and MAE metrics. Bold values indicate the best performance for each horizon and dataset across all ranks. The results highlight the impact of rank selection on model efficiency and accuracy under constrained data availability.

> Table 9 (see PDF p. 20). Comprehensive evaluation of the One-for-All model under Few-Shot Forecasting using 5% of training data. The table presents the model's performance across four prediction horizons ($N \in \{96, 192, 336, 720\}$) on five datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather), with varying low-rank configurations. Each cell reports the MSE and MAE metrics. Bold values indicate the best performance for each horizon and dataset across all ranks. The results highlight the impact of rank selection on model efficiency and accuracy under constrained data availability. "-" indicates that training data is not available.

#### C.3 Detailed Ablation Result for Zero-shot Forecasting

Table 10 presents the complete ablation results for zero-shot forecasting under the One-for-All framework. Similar to the few-shot setting, performance steadily improves with increasing rank, indicating the benefit of richer representations. The improvements are particularly noticeable in cross-domain evaluations, such as M3->M4 and M4->M3, highlighting the robustness of the proposed rank-stable design. Notably, the best performance is achieved with moderate-to-high ranks (e.g., Rank 16 and Rank 64), after which the gains plateau or vary slightly. This confirms the stability of our approach and its adaptability across different dataset granularities and forecasting horizons.

> Table 10 (see PDF p. 20). Zero-shot forecasting performance of the One-for-All model across varying LoRA ranks on the M3 and M4 datasets. The table reports sMAPE scores for different frequency categories (Yearly, Quarterly, Monthly) in both M3->M4 and M4->M3 transfer settings. The highlighted values correspond to Rank 16, used as the baseline configuration. Lower sMAPE indicates better performance.

| Variant | Metric | Rank 2 | Rank 4 | Rank 8 | Rank 16 | Rank 32 | Rank 64 | Rank 128 | Rank 256 | Rank 512 | Rank 1024 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M3 -> M4 | Yearly | 13.80 | 13.67 | 13.60 | 13.53 | 13.54 | 13.51 | 13.41 | 13.69 | 13.57 | 13.73 |
| M3 -> M4 | Quarterly | 11.31 | 11.03 | 10.86 | 11.00 | 10.92 | 11.09 | 10.89 | 11.12 | 10.91 | 11.45 |
| M3 -> M4 | Monthly | 15.52 | 15.28 | 15.16 | 15.28 | 15.39 | 14.90 | 15.07 | 14.94 | 14.85 | 14.95 |
| M4 -> M3 | Yearly | 22.77 | 22.25 | 20.75 | 19.23 | 19.11 | 18.01 | 18.16 | 17.87 | 17.61 | 17.44 |
| M4 -> M3 | Quarterly | 11.92 | 11.83 | 11.57 | 11.77 | 11.66 | 11.01 | 11.66 | 11.28 | 11.47 | 11.20 |
| M4 -> M3 | Monthly | 16.05 | 15.34 | 15.23 | 14.30 | 14.53 | 14.94 | 14.67 | 14.83 | 14.86 | 14.73 |
| Avg | sMAPE | 15.23 | 14.90 | 14.53 | 14.19 | 14.19 | 13.91 | 13.98 | 13.96 | 13.88 | 13.92 |

#### C.4 Detailed Ablation Result for Short-term Forecasting Task

The complete ablation results for short-term forecasting are presented in Table 11. The analysis is conducted across multiple ranks using three key metrics: SMAPE, MASE, and OWA. The results span different granularities (Yearly, Quarterly, Monthly, and Others) of the M4 dataset. We observe that our One-for-All framework demonstrates consistent improvements at selected ranks across all metrics, particularly at Rank 1024. The stability of performance across a broad range of ranks reinforces the effectiveness and generalisability of the proposed rank-stable design in short-horizon settings.

> Table 11 (see PDF p. 21). Short-term forecasting performance on the M4 dataset across various ranks, evaluated using three performance metrics: SMAPE, MASE, and OWA. The results are presented for different data frequencies (Yearly, Quarterly, Monthly, and Others) with the average performance shown in the "Avg" column. The table summarizes the impact of different rank configurations on model performance for each metric. Lower values indicate better forecasting accuracy.

Markdown split below into three metric blocks for readability; values are unchanged from the PDF table.

**SMAPE**

| Rank | Yearly | Quarterly | Monthly | Others | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rank 2 | 15.28 | 10.48 | 12.96 | 5.29 | 12.51 |
| Rank 4 | 14.29 | 10.51 | 13.11 | 5.37 | 12.37 |
| Rank 8 | 15.21 | 10.50 | 13.02 | 5.34 | 12.53 |
| Rank 16 | 14.79 | 10.36 | 12.96 | 5.34 | 12.38 |
| Rank 32 | 15.05 | 10.49 | 13.36 | 5.45 | 12.66 |
| Rank 64 | 15.34 | 10.46 | 12.99 | 5.29 | 12.54 |
| Rank 128 | 15.17 | 10.56 | 13.05 | 5.23 | 12.55 |
| Rank 256 | 15.97 | 10.38 | 13.15 | 5.56 | 12.75 |
| Rank 512 | 14.77 | 10.44 | 13.26 | 5.29 | 12.53 |
| Rank 1024 | 14.66 | 10.52 | 13.10 | 5.49 | 12.46 |

**MASE**

| Rank | Yearly | Quarterly | Monthly | Others | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rank 2 | 3.71 | 1.23 | 0.96 | 3.59 | 1.79 |
| Rank 4 | 3.34 | 1.21 | 0.96 | 3.66 | 1.70 |
| Rank 8 | 3.76 | 1.24 | 0.96 | 3.60 | 1.80 |
| Rank 16 | 3.59 | 1.22 | 0.96 | 3.57 | 1.76 |
| Rank 32 | 3.68 | 1.23 | 0.99 | 3.63 | 1.80 |
| Rank 64 | 3.67 | 1.22 | 0.96 | 3.53 | 1.77 |
| Rank 128 | 3.72 | 1.24 | 0.96 | 3.49 | 1.79 |
| Rank 256 | 4.07 | 1.22 | 0.97 | 3.61 | 1.87 |
| Rank 512 | 3.49 | 1.22 | 0.99 | 3.50 | 1.74 |
| Rank 1024 | 3.53 | 1.24 | 0.96 | 3.61 | 1.75 |

**OWA**

| Rank | Yearly | Quarterly | Monthly | Others | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rank 2 | 0.93 | 0.92 | 0.90 | 1.12 | 0.93 |
| Rank 4 | 0.85 | 0.92 | 0.90 | 1.14 | 0.90 |
| Rank 8 | 0.93 | 0.92 | 0.90 | 1.13 | 0.93 |
| Rank 16 | 0.90 | 0.91 | 0.90 | 1.21 | 0.91 |
| Rank 32 | 0.92 | 0.92 | 0.92 | 1.14 | 0.93 |
| Rank 64 | 0.93 | 0.92 | 0.90 | 1.11 | 0.92 |
| Rank 128 | 0.93 | 0.93 | 0.90 | 1.10 | 0.93 |
| Rank 256 | 1.00 | 0.91 | 0.91 | 1.15 | 0.96 |
| Rank 512 | 0.89 | 0.91 | 0.92 | 1.11 | 0.91 |
| Rank 1024 | 0.89 | 0.93 | 0.90 | 1.14 | 0.91 |

#### C.5 Detailed Ablation Result for Classification Task

Table 12 provides a comprehensive ablation study for the classification task across multiple datasets, evaluating performance at different ranks. The One-for-All framework demonstrates robust improvements in accuracy as the rank increases, with consistent gains observed particularly in complex and cross-domain datasets such as Japanese Vowels. These results highlight the rank-stable nature of the proposed framework, where higher ranks enable better representational capacity without sacrificing generalisability. The average performance plateaus at higher ranks, suggesting effective convergence and stability across diverse classification challenges.

> Table 12 (see PDF p. 21). Detailed accuracy analysis of the One-for-All model with varying ranks (Rank 2 to Rank 1024) for classification tasks across multiple datasets. The accuracy for each dataset is reported as a percentage, showing the model's performance at different rank configurations. The table includes results for datasets such as Ethanol, Face Detection, Heartbeat, Japanese Vowels, SCP1, and SCP2.

| Dataset | Rank 2 | Rank 4 | Rank 8 | Rank 16 | Rank 32 | Rank 64 | Rank 128 | Rank 256 | Rank 512 | Rank 1024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ethanol | 28% | 30% | 32% | 32% | 30% | 29% | 30% | 31% | 30% | 32% |
| Face Detection | 56% | 62% | 65% | 66% | 66% | 66% | 66% | 66% | 68% | 67% |
| Heartbeat | 72% | 75% | 76% | 75% | 75% | 76% | 76% | 76% | 76% | 75% |
| Japanese Vowels | 46% | 81% | 95% | 97% | 97% | 97% | 98% | 97% | 98% | 97% |
| SCP1 | 83% | 89% | 91% | 91% | 91% | 92% | 91% | 92% | 91% | 93% |
| SCP2 | 50% | 55% | 60% | 57% | 53% | 57% | 55% | 58% | 57% | 55% |

#### C.6 Detailed Ablation Result for Anomaly Detection Task

The detailed ablation results for the anomaly detection task across different datasets and ranks are presented in Table 13. The One-for-All framework demonstrates robust performance across ranks, particularly at moderate and higher ranks. Notably, the average F1-score improves or remains stable, suggesting the rank-based tuning contributes positively to both generalization and sensitivity to rare anomalous events.

> Table 13 (see PDF p. 22). Detailed F1-score (as %) analysis of the One-for-All model with varying ranks (Rank 2 to Rank 1024) for anomaly detection tasks across multiple datasets. The F1-score, representing the harmonic mean of precision and recall, is reported for datasets such as MSL, PSM, SMAP, SMD, and SWAT. The "Avg" row provides the average F1-score across all datasets for each rank configuration. The table highlights the model's performance in anomaly detection at different ranks.

| Dataset | Rank 2 | Rank 4 | Rank 8 | Rank 16 | Rank 32 | Rank 64 | Rank 128 | Rank 256 | Rank 512 | Rank 1024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MSL | 80% | 83% | 82% | 82% | 82% | 82% | 82% | 82% | 82% | 82% |
| PSM | 96% | 96% | 97% | 97% | 97% | 97% | 97% | 97% | 94% | 94% |
| SMAP | 66% | 67% | 67% | 67% | 67% | 67% | 67% | 67% | 67% | 67% |
| SMD | 82% | 82% | 83% | 84% | 84% | 84% | 84% | 84% | 84% | 84% |
| SWAT | 88% | 92% | 92% | 92% | 92% | 92% | 92% | 92% | 92% | 92% |
| Avg | 82% | 84% | 84% | 84% | 84% | 84% | 84% | 84% | 84% | 84% |

### D. Parameter and Models Size Comparison for One-for-All Across Long/Few-shot Forecasting Tasks

In our study, we delve into the intricacies of fine-tuning our One-for-All framework with the Gaussian rsLoRA framework, across a spectrum of ranks ranging from 2 to 1024. Our comparative analysis focuses on two pivotal aspects: the number of trainable parameters and the memory size of the models. For our evaluation metrics, we present the total count of trainable parameters in million (M) and the models' memory size in megabytes (MiB). This systematic approach allows for a comprehensive understanding of the resource-efficient frameworks. The detailed results for long/few-shot forecasting are provided in Table 14, Table 15, and Table 16. In our exploration, we analyze the total parameter requirements and model sizes across various ranks ranging from rank 2 to rank 1024. As expected, we observe an increase in the number of trainable parameters and model size with higher ranks. However, even with this increase, the total number of parameters and model memory requirements remain significantly lower compared to recently developed pre-trained GPT2-based models such as GPT4TS. Notably, for models with higher ranks (e.g., rank 1024), the model memory size is only 41% compared to the GPT4TS model. Therefore, the resource-efficient One-for-All framework proves to be exceptionally beneficial for long/few-shot forecasting tasks.

> Table 14 (see PDF p. 23). Efficiency analysis of the One-for-All model with varying ranks (Rank 2 to Rank 1024) for few-shot forecasting and long-term forecasting tasks using the ETTh1 and ETTh2 datasets. The table presents the trainable parameters (in millions), the percentage of total model parameters, and the memory required to save the trained model (in MiB) for each rank across both forecasting tasks. The results highlight the impact of increasing model rank on parameter count, memory usage, and the overall efficiency of the model, showcasing the trade-offs between model size and performance.

> Table 15 (see PDF p. 23). Efficiency analysis of the One-for-All model with varying ranks (Rank 2 to Rank 1024) for few-shot forecasting and long-term forecasting tasks using the ETTm1 and ETTm2 datasets. The table presents the trainable parameters (in millions), the percentage of total model parameters, and the memory required to save the trained model (in MiB) for each rank across both forecasting tasks. The results highlight the impact of increasing model rank on parameter count, memory usage, and the overall efficiency of the model, showcasing the trade-offs between model size and performance.

> Table 16 (see PDF p. 24). Efficiency analysis of the One-for-All model with varying ranks (Rank 2 to Rank 1024) for few-shot forecasting and long-term forecasting tasks using the Weather datasets. The table presents the trainable parameters (in millions), the percentage of total model parameters, and the memory required to save the trained model (in MiB) for each rank across both forecasting tasks. The results highlight the impact of increasing model rank on parameter count, memory usage, and the overall efficiency of the model, showcasing the trade-offs between model size and performance.

### E. Parameter Comparison of State-of-the-Art Models Across Long/Few-shot Forecasting Tasks

To ensure a fair and rigorous comparison with the One-for-All framework, we systematically examine the total number of trainable parameters and model sizes of various state-of-the-art models. All experiments are conducted under the same settings as TimesNet to maintain consistency across evaluations. The results for few-shot and long-term forecasting are summarised in Table 17. This standardised evaluation allows us to comprehensively and impartially assess the efficiency and scalability of the One-for-All framework in relation to existing approaches.

> Table 17 (see PDF p. 24). Efficiency analysis of state-of-the-art models in few-shot and long-term forecasting tasks on the ETTh1 and ETTh2 datasets. The table provides the trainable parameters (in millions), the percentage of total parameters, and the memory required for saving the trained model (in MiB) across various forecasting tasks. The models compared include GPT4TS, TIME-LLM (LLaMA-7B), TEST (GPT-2 Medium), TEMPO (GPT-2 Medium), TimesNet, FEDformer, Stationary, ETSformer, Autoformer, Informer, and Reformer, with performance metrics shown for different ranks of forecasting. This analysis helps in understanding the trade-offs between model size, efficiency, and forecasting accuracy.
