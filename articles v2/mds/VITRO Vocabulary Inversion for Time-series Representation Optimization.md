# VITRO: Vocabulary Inversion for Time-series Representation Optimization

Filippos Bellos  
University of Michigan  
Ann Arbor, MI, USA  
Email: `fbellos@umich.edu`

Nam H. Nguyen  
Capital One  
McLean, VA, USA  
Email: `nam.nguyen@capitalone.com`

Jason J. Corso  
University of Michigan  
Ann Arbor, MI, USA  
Email: `jjcorso@umich.edu`

*arXiv:2412.17921v1 [cs.LG], December 23, 2024.*

**Abstract.** Although LLMs have demonstrated remarkable capabilities in processing and generating textual data, their pre-trained vocabularies are ill-suited for capturing the nuanced temporal dynamics and patterns inherent in time series. The discrete, symbolic nature of natural language tokens, which these vocabularies are designed to represent, does not align well with the continuous, numerical nature of time series data. To address this fundamental limitation, we propose VITRO. Our method adapts textual inversion optimization from the vision-language domain in order to learn a new time series per-dataset vocabulary that bridges the gap between the discrete, semantic nature of natural language and the continuous, numerical nature of time series data. We show that learnable time series-specific pseudo-word embeddings represent time series data better than existing general language model vocabularies, with VITRO-enhanced methods achieving state-of-the-art performance in long-term forecasting across most datasets.

**Index Terms.** Multivariate Time Series, Large Language Models, Forecasting, Optimization, Textual Inversion.

## I. Introduction

Large Language Models (LLMs) have transformed natural language processing (NLP), excelling in traditional NLP tasks like text generation but also showing promise in tasks that require complex and structured reasoning [1, 2]. Their impact has extended beyond NLP, contributing to rapid advancements in computer vision and other signal processing applications through the development of multimodal models that can process and integrate information from various modalities such as text, images, and audio. This versatility has naturally led the community to explore their potential in time series forecasting, a fundamental capability in numerous real-world dynamic systems [3] including energy load management [4], climate modelling [5], traffic forecasting [6], etc. Traditionally, these forecasting tasks have required extensive domain expertise and task-specific model designs, an approach that stands in contrast to LLMs, which demonstrate strong performance across diverse tasks with minimal examples, often in few-shot or zero-shot scenarios [7, 8]. This contrast underscores the need to consider if and how the pre-trained knowledge and generalization capabilities of LLMs can be fully harnessed to perform accurate time series forecasting without fine-tuning the underlying model.

Time-LLM [9] and TEST [10] attempt to address this challenge by reprogramming the input time series into text prototype representations and using textual prompts to provide additional context. Crucially, these methods enable the LLM to perform time series forecasting while keeping the pre-trained model completely frozen, thus fully leveraging the model's pre-trained capabilities. Other methods, such as OFA [11] and S2IP-LLM [12], also investigate the use of pre-trained LLMs for time-series forecasting. However, they require partial fine-tuning of the underlying language model to achieve good performance, potentially limiting their ability to fully exploit the LLM's pre-trained knowledge.

> Fig. 1 (see PDF p. 1). VITRO optimizes learnable pseudo-word embeddings $v_i$ for each time series instance $X_i$ and a shared dataset embedding $s$ to construct a new data-centric time series vocabulary tailored for forecasting. Time series are normalized, patched, and embedded. These patch embeddings $E_i$ serve as prompts to guide the optimization of pseudo-words. The composite representation, including statistical features $e_{stats}$, is fed into a frozen LLM, whose output is projected to generate forecasts $\hat{Y}_i$.

Despite the promise in these methods, they are still limited by the existing LLM vocabulary they rely on, which fails to capture the nuanced patterns and characteristics specific to time series data. This limitation naturally raises the question: *Is there a better way to represent time series data than using the general-purpose vocabulary of LLMs, in order to leverage the inherent capabilities of LLMs for effective time series forecasting?*

To address this question, we propose VITRO, a new method that, as depicted in Fig. 1, constructs a time series specific vocabulary by learning unique pseudo-words for each time series instance in a dataset, inspired by the concept of textual inversion [13]. In addition, VITRO optimizes a shared embedding, able to capture the domain specific dataset information, which in many real-world applications may not always be available or informative.

Intuitively, this method bridges the gap between LLMs and time series data by creating a vocabulary that encodes time series information in a format interpretable by the language model, while at the same time capturing the subtle variations in time series.

We demonstrate that VITRO can be leveraged across different forecasting approaches and LLM architectures, showcasing its potential for broad application in the field of time series forecasting. Quantitative experiments show state-of-the-art performance for the methods that leverage VITRO, while qualitative analysis reveals that our learned vocabulary exhibits distinct patterns in attention weights and embedding distributions, indicating successful specialization for time series tasks.

## II. Method

### A. Problem Formulation and Overview

We formulate our problem of Vocabulary Inversion for Time Series Representation Optimization as follows. Let $X \in \mathbb{R}^{N \times T}$ denote the time series data consisting of $N$ different 1-dimensional variables across a lookback window of $T$ time steps, where the $i$-th series is denoted as $X_i \in \mathbb{R}^{1 \times T}$. We aim to learn a new time series data-centric vocabulary that will allow a large language model $f(\cdot)$ to better understand the input time series in order to more accurately predict the next $\tau$ time steps based on the input window $T$. Let $Y \in \mathbb{R}^{N \times \tau}$ denote the ground truth values for the next $\tau$ time steps, and $\hat{Y} \in \mathbb{R}^{N \times \tau}$ represent the corresponding predictions. The overall objective is to minimize the mean square errors between $Y$ and $\hat{Y}$, defined as:

$$
\frac{1}{\tau}\sum_{t=1}^{\tau}\left\|Y_t - \hat{Y}_t\right\|_F^2
\tag{1}
$$

To solve this problem, we are inspired by the approach of textual inversion [13] from the text-to-image diffusion model literature - a simple yet powerful technique in low-shot image generation that learns a common concept in given images as a single token in text embedding space. Motivated by its success, we aim to learn different concepts-representations of time series data as text embeddings and use them to develop a new time series forecasting vocabulary, which we hypothesize will represent time series data better than the existing general natural language model vocabulary.

Our method consists of two stages. The first stage optimizes a specialized vocabulary tailored for time series forecasting. This stage captures patterns across an entire dataset, creating a rich vocabulary that reflects the temporal dynamics and variations inherent in the time series. The primary goal here is to establish a comprehensive representation rather than immediate forecasting accuracy. In the second stage, we use this specialized vocabulary for the actual forecasting tasks. This stage benefits from the broad context learned in the first stage, applying it to individual time series instances to enhance forecasting performance. The two-stage approach ensures that our vocabulary is informed by the full dataset. It allows us to first establish a strong foundational representation before focusing on specific forecasting tasks.

### B. Stage 1: Vocabulary Inversion for Time Series

LLMs begin with a text processing step where each word or sub-word in an input string is converted to a token, which is an index in some pre-defined dictionary. Each token is then linked to a unique embedding vector that can be retrieved through an index-based embedding lookup.

We choose this embedding space as the target for vocabulary inversion. Specifically, let a time series dataset $D = \{X_1, X_2, \ldots, X_n\}$, where $n$ represents the number of time series instances in a dataset and each time series instance $X_i \in \mathbb{R}^{1 \times T}$ represents a time series segment of length $T$ (lookback window). For this dataset, we designate a set of placeholder strings, $P^* = \{P_1^*, P_2^*, \ldots, P_n^*\}$, where each $P_i^*$ represents a unique time series instance $X_i$. Additionally, we introduce a placeholder $S^*$ shared per dataset which represents the entire dataset-domain information.

Concretely, our approach involves an iterative optimization process for:

- A corresponding embedding $v_i$ for each $P_i^*$, effectively expanding the LLM's vocabulary with $n$ new "words" that encode time series information.
- A corresponding embedding $s$ for $S^*$, that encodes the domain information for $D$ as a whole.

We start by randomly initializing each pseudo-word embedding $v_i$ and associating the placeholder $P_i^*$ to it. Similarly, we initialize the shared embedding $s$ and associate it with $S^*$. The optimization process then runs for a fixed number of iterations, optimizing these embeddings to better represent the time series forecasting data. To condition the generation process, we utilize a small set of text prompt templates containing these placeholders, such as "The time series is $P_i^*$" or "Forecast the next steps of $P_i^*$".

As shown in Fig. 1, we essentially intervene in the LLM's embedding process. This allows us to "inject" a rich set of time series concepts into the LLM's vocabulary, each reflecting a specific instance in our dataset. The same happens with the shared embedding, which "injects" general domain information.

**1) Model Pipeline:** Following convention, for each input time series $X_i \in \mathbb{R}^{T}$, we first apply reversible instance normalization (RevIN) [14] to mitigate distribution shift: $\tilde{X}_i = \mathrm{RevIN}(X_i)$. We then divide $\tilde{X}_i$ into $P$ overlapping or non-overlapping patches of length $L_p$: $X_{P,i} \in \mathbb{R}^{P \times L_p}$ [15], where

$$
P = \left\lfloor \frac{T - L_p}{S} \right\rfloor + 2,
$$

and $S$ is the horizontal sliding stride.

To obtain the final embeddings, we apply a linear transformation to each patch: $E_i = W_e X_{P,i} + b_e$, where $E_i \in \mathbb{R}^{P \times d}$, $W_e \in \mathbb{R}^{d \times L_p}$ is a learnable weight matrix, $b_e \in \mathbb{R}^{d}$ is a learnable bias vector, and $d$ is the embedding dimension of the target LLM.

**Patch Embeddings as Prompts.** We leverage the patch embeddings $E_i$, as a composite prompt to guide the LLM's processing of time series data and the optimization of pseudo-words $v_i$, and shared embedding $s$. This approach is inspired by recent advancements showing that non-textual data modalities can be effectively integrated as prefixes in prompts to facilitate reasoning [16]. In our case, the patch embeddings $E_i$ serve as a numerical representation of the time series, while the pseudo-words $v_i$ and shared embedding $s$ provide learnable, text-like anchors for the model.

For each pseudo-word $P_i^*$, we learn a corresponding embedding vector $v_i \in \mathbb{R}^{d}$. We pass $P_i^*$ through the LLM's tokenizer to obtain a token representation which we associate in the LLM's embedding lookup table with our learnable embedding $v_i$. Accordingly, we associate the shared word $S^*$ with the learned shared embedding $s$ in the embedding lookup process.

We then concatenate $E_i$, $v_i$ and $s$ along with certain statistics we calculate for $X_i$, and we feed them through the frozen LLM to obtain the last hidden layer output $h_i \in \mathbb{R}^{h}$:

$$
h_i = f([E_i; v_i; s; e_{stats}]),
$$

where $f(\cdot)$ denotes the frozen LLM and `;` denotes concatenation.

The last hidden layer output $h_i$ is passed through a learnable linear layer $g(\cdot)$ to generate the forecasted values $\hat{Y}_i$:

$$
\hat{Y}_i = g(h_i) = W h_i + b,
$$

where $W \in \mathbb{R}^{\tau \times h}$ and $b \in \mathbb{R}^{\tau}$ are the learnable weights and bias.

**2) Optimization Objective:** Our optimization objective is to minimize the loss $\mathcal{L}$ between the forecasted values $\hat{Y}_i$ and the ground truth future values $Y_i$ for each time series instance $X_i$:

$$
\mathcal{L}_{MSE}
=
\frac{1}{\tau}
\left\|
Y_i -
\left(
g\left(
f\left(
E_i;
\text{``The time series is }[P_i^*]\text{, The dataset is }[S^*]\text{''};
\mathrm{Statistics}(X_i)
\right)
\right)
\right)
\right\|_F^2
\tag{2}
$$

We optimize the shared embedding $s$, pseudo-word embeddings $V = v_1, v_2, \ldots, v_n$ and all other learnable parameters $\theta$ (including $W_e$, $b_e$, $W$ and $b$) to minimize the total loss:

$$
s^*, V^*, \theta^*
=
\arg\min_{s,V,\theta}
\sum_{i=1}^{n} \mathcal{L}_{MSE}
\tag{3}
$$

### C. Stage 2: Time Series Forecasting with Learned Vocabulary

Stage 2 of our method focuses on leveraging the learned vocabulary $V$ for time series forecasting. We present two approaches using different LLM architectures (i.e. GPT2 [17] and LLaMa [18]) demonstrating that the effectiveness of VITRO is not restricted to one type of LLM or LLM-based method: a similarity-based selection method that directly utilizes the vocabulary, and TimeLLM's [9] attention-based approach that allows us to assess VITRO's benefits compared to the standard LLM vocabulary within the TimeLLM method, which is the current state of the art method that utilizes a frozen pretrained LLM.

**Similarity-based Selection (Sim).** For computational efficiency, instead of using the word embeddings from the full vocabulary $V$, we first derive a reduced set of core lexicon embeddings $C$ using a linear mapping function $h(\cdot)$:

$$
C = h(V) = W_v V + b_v,
$$

where $C \in \mathbb{R}^{n' \times d}$, $n'$ is the number of core lexicon embeddings with $n' \le n$, $W_v \in \mathbb{R}^{n' \times n}$ and $b_v \in \mathbb{R}^{n'}$ are learnable parameters. For each patch embedding $E_i$ and each core lexicon embedding $c_m \in C$, we compute the cosine similarity:

$$
\mathrm{sim}(E_i, c_m) = \frac{E_i \cdot c_m}{\|E_i\|\|c_m\|}.
$$

From these $n'$ core lexicon embeddings, we then select the top $k$ embeddings with the highest similarity scores, where $k < n'$. This similarity-based ranking and selection ensures that we identify the most relevant core lexicon embeddings for each patch, while maintaining computational efficiency by operating in a reduced similarity space ($n'$ instead of $n$).

We then form an augmented embedding for each patch:

$$
\hat{e}_i = [E_i; c_1; c_2; \ldots; c_k; s; e_{stats}],
$$

that will serve as input to the frozen pre-trained LLM, which in this case is GPT2.

**Attention-based approach.** As mentioned, we also employ TimeLLM's [9] attention-based approach, demonstrating the improved results (in Section III) when using our optimized vocabulary over the existing one. This method involves a multi-head cross-attention mechanism between patch embeddings and our optimized vocabulary, allowing the model to dynamically select relevant information.

Concretely, we employ a multi-head cross-attention layer. For each head $h = \{1, \ldots, H\}$, we define the Query matrices as $Q_h^{(i)} = E_i W_h^Q$, the Key matrices as $K_h^{(i)} = C W_h^K$ and the Value matrices as $V_h^{(i)} = C W_h^V$, where $W_h^Q, W_h^K, W_h^V \in \mathbb{R}^{d \times d_h}$, and $d_h = d / H$.

The attention operation for each head is:

$$
Z_h^{(i)} = \mathrm{Softmax}\left(\frac{Q_h^{(i)} K_h^{(i)\top}}{\sqrt{d_h}}\right)V_h^{(i)}.
$$

Aggregating $Z_h^{(i)} \in \mathbb{R}^{P \times d_h}$ across all heads yields $Z^{(i)} \in \mathbb{R}^{P \times d}$, which is then passed through the frozen LLM, which in this case is Llama-7B, along with the optimized shared embedding that encapsulates the domain information, and the calculated statistics.

## III. Experiments

In our experimental evaluation, we compare the effectiveness of VITRO vocabularies against the general natural language LLM vocabularies within our Stage 2 framework, demonstrating the advantages of our optimized time series representation. We benchmark VITRO-enhanced methods against other LLM-based methods and traditional time series forecasting baselines on the task of long-term forecasting. We also provide a qualitative analysis of VITRO and existing LLM vocabularies. For all baselines, we adhere to the experimental configurations outlined by [19], utilizing their unified pipeline[^1].

**Baselines.** The baselines include the best performing approaches based on LLMs, i.e., Time-LLM [9] whose method is used as variant of our stage 2, and S2IP-LLM [12] even though this method partly finetunes the backbone model. We also include the best performing Transformer-based and non-Transformer methods, i.e. PatchTST [15] and Dlinear [20].

### A. Long-term Forecasting

**Setup.** We evaluate the effectiveness of VITRO across 7 public datasets: Weather, Electricity, Traffic, and four ETT datasets (i.e., ETTh1, ETTh2, ETTm1, and ETTm2), which have been widely adopted as benchmarking datasets for long-term forecasting models. The input time series length is 512 and we evaluate the performance on four different horizons `{96, 192, 336, 720}`. The evaluation metrics include the mean square error (MSE) and the mean absolute error (MAE).

**TABLE I.** Long-term forecasting results for `{96, 192, 336, 720}` horizons. A lower value indicates a better performance. All results are averaged from four forecasting horizons `{96, 192, 336, 720}`. Arrows `&darr;` indicate positive impact of VITRO compared to existing LLM vocabulary. Bold: best results. Underlined: second best. The authors reproduced TimeLLM, S2IP-LLM, PatchTST, Dlinear results using their official open-source implementations.

<table>
  <thead>
    <tr>
      <th>Methods</th>
      <th colspan="2">VITRO-Sim</th>
      <th colspan="2">Sim</th>
      <th colspan="2">VITRO-TimeLLM</th>
      <th colspan="2">TimeLLM</th>
      <th colspan="2">S<sup>2</sup>IP-LLM</th>
      <th colspan="2">PatchTST</th>
      <th colspan="2">Dlinear</th>
    </tr>
    <tr>
      <th>Metric</th>
      <th>MSE</th>
      <th>MAE</th>
      <th>MSE</th>
      <th>MAE</th>
      <th>MSE</th>
      <th>MAE</th>
      <th>MSE</th>
      <th>MAE</th>
      <th>MSE</th>
      <th>MAE</th>
      <th>MSE</th>
      <th>MAE</th>
      <th>MSE</th>
      <th>MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>ETTh1</strong></td>
      <td><strong>0.412</strong> &darr;</td>
      <td><strong>0.430</strong> &darr;</td>
      <td>0.442</td>
      <td>0.449</td>
      <td><u>0.416</u> &darr;</td>
      <td><u>0.437</u> &darr;</td>
      <td>0.437</td>
      <td>0.450</td>
      <td>0.425</td>
      <td>0.440</td>
      <td>0.444</td>
      <td>0.453</td>
      <td>0.418</td>
      <td>0.439</td>
    </tr>
    <tr>
      <td><strong>ETTh2</strong></td>
      <td><u>0.351</u> &darr;</td>
      <td><strong>0.393</strong> &darr;</td>
      <td>0.370</td>
      <td>0.402</td>
      <td><strong>0.349</strong> &darr;</td>
      <td><u>0.395</u> &darr;</td>
      <td>0.360</td>
      <td>0.400</td>
      <td>0.358</td>
      <td>0.403</td>
      <td>0.381</td>
      <td>0.411</td>
      <td>0.502</td>
      <td>0.481</td>
    </tr>
    <tr>
      <td><strong>ETTm1</strong></td>
      <td>0.353 &darr;</td>
      <td><strong>0.380</strong> &darr;</td>
      <td>0.365</td>
      <td>0.388</td>
      <td><u>0.352</u> &darr;</td>
      <td>0.387 &darr;</td>
      <td>0.367</td>
      <td>0.396</td>
      <td><strong>0.347</strong></td>
      <td><u>0.382</u></td>
      <td>0.363</td>
      <td>0.391</td>
      <td>0.357</td>
      <td>0.389</td>
    </tr>
    <tr>
      <td><strong>ETTm2</strong></td>
      <td><strong>0.260</strong> &darr;</td>
      <td><u>0.323</u> &darr;</td>
      <td>0.284</td>
      <td>0.332</td>
      <td>0.263 &darr;</td>
      <td><strong>0.321</strong> &darr;</td>
      <td>0.264</td>
      <td>0.325</td>
      <td><u>0.261</u></td>
      <td>0.326</td>
      <td>0.267</td>
      <td>0.325</td>
      <td>0.275</td>
      <td>0.340</td>
    </tr>
    <tr>
      <td><strong>Weather</strong></td>
      <td>0.230 &darr;</td>
      <td>0.268 &darr;</td>
      <td>0.233</td>
      <td>0.273</td>
      <td><strong>0.225</strong> &darr;</td>
      <td><strong>0.263</strong> &darr;</td>
      <td><u>0.227</u></td>
      <td>0.265</td>
      <td>0.229</td>
      <td>0.267</td>
      <td><strong>0.225</strong></td>
      <td><u>0.264</u></td>
      <td>0.248</td>
      <td>0.300</td>
    </tr>
    <tr>
      <td><strong>Electricity</strong></td>
      <td><strong>0.161</strong> &darr;</td>
      <td><u>0.258</u> &darr;</td>
      <td><u>0.165</u></td>
      <td>0.261</td>
      <td>0.166 &darr;</td>
      <td>0.267 &darr;</td>
      <td>0.168</td>
      <td>0.270</td>
      <td>0.167</td>
      <td>0.263</td>
      <td><strong>0.161</strong></td>
      <td><strong>0.252</strong></td>
      <td>0.166</td>
      <td>0.263</td>
    </tr>
    <tr>
      <td><strong>Traffic</strong></td>
      <td><u>0.399</u> &darr;</td>
      <td><u>0.276</u> &darr;</td>
      <td>0.402</td>
      <td>0.279</td>
      <td>0.408 &darr;</td>
      <td>0.306 &darr;</td>
      <td>0.410</td>
      <td>0.310</td>
      <td>0.418</td>
      <td>0.303</td>
      <td><strong>0.390</strong></td>
      <td><strong>0.263</strong></td>
      <td>0.433</td>
      <td>0.295</td>
    </tr>
  </tbody>
</table>

**Results.** Our results are shown in TABLE I. When we replace the existing general-purpose vocabulary with VITRO's learned vocabulary in both the Time-LLM approach and our similarity-based method, we observe consistent improvements across all 7 datasets tested for both MSE and MAE metrics. The impact is particularly pronounced for the ETTh1, ETTh2 and ETTm1 datasets. When compared to state-of-the-art methods, our VITRO-enhanced approaches consistently outperform across most datasets. Specifically, for the MAE metric our methods outperform the LLM-based method (S2IP-LLM) in all 7 datasets while for MSE in 6 out of 7 datasets. Comparing against the transformer based method (PatchTST) VITRO performs better in 5 out of 7 datasets for the MAE metric, 4 out of 7 for MSE while achieving the same result for the same metric in 2 datasets (i.e. Electricity, Weather). Finally, we outperform the non-transformer method (Dlinear) in all datasets for both metrics.

## IV. Qualitative Analysis

Figures 2 and 3 reveal the impact of the specialized nature of the new vocabulary for time series tasks, contrasting with the general-purpose characteristics of existing vocabularies.

In Fig. 3, the heatmaps, generated by the attention-based approach of stage 2 (TimeLLM approach), for the VITRO vocabulary show distinct horizontal striping patterns. This suggests that certain vocabulary elements are consistently more important across different parts of the input sequence, indicating that our vocabulary has captured some general features and underlying structures in time series data applicable across various time steps. In contrast, the existing vocabulary's weights show a more uniform distribution, suited for more general language tasks. Figure 2's PCA and TSNE visualizations further support this distinction: the new vocabulary forms a U-shaped manifold, suggesting a robust and structured embedding space, which indicates a specialized representation of time series concepts, while the existing vocabulary reveals a diffuse, circular distribution typical of general-purpose language embeddings.

> Fig. 2 (see PDF p. 4). PCA and t-SNE visualizations of VITRO and existing general-purpose vocabulary embedding space.

> Fig. 3 (see PDF p. 4). VITRO and LLM existing vocabularies heatmaps. Each row corresponds to a word in the vocabulary, the y-axis represents the index of the word, and the x-axis denotes the embedding dimensions. Brighter colors indicate higher values.

## V. Conclusion and Future Work

VITRO demonstrates significant potential in enhancing LLMs for time series forecasting by learning a time series data-centric vocabulary through vocabulary inversion. Our results consistently show that time series forecasting accuracy can be improved by replacing the LLM's general-purpose vocabulary with our VITRO-optimized one. However, as an iterative optimization-based method, VITRO's computational cost may limit its application in larger datasets. Future research directions will explore further optimization of the vocabulary learning process, extending VITRO to other time series tasks beyond forecasting, and integrating VITRO with other LLM-based methods (e.g. S2IP-LLM).

## References

[1] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou, "Chain-of-thought prompting elicits reasoning in large language models," in *Proceedings of the 36th International Conference on Neural Information Processing Systems*, Red Hook, NY, USA, 2024, NIPS '22, Curran Associates Inc.

[2] Filippos Bellos, Yayuan Li, Wuao Liu, and Jason Corso, "Can large language models reason about goal-oriented tasks?," in *Proceedings of the First edition of the Workshop on the Scaling Behavior of Large Language Models (SCALE-LLM 2024)*, St. Julian's, Malta, Mar. 2024, pp. 24-34, Association for Computational Linguistics.

[3] Ming Jin, Huan Yee Koh, Qingsong Wen, Daniele Zambon, Cesare Alippi, Geoffrey I. Webb, Irvin King, and Shirui Pan, "A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection," *IEEE transactions on pattern analysis and machine intelligence*, vol. PP, 2023.

[4] Hengbo Liu, Ziqing Ma, Linxiao Yang, Tian Zhou, Rui Xia, Yi Wang, Qingsong Wen, and Liang Sun, "Sadi: A self-adaptive decomposed interpretable framework for electric load forecasting under extreme events," in *IEEE International Conference on Acoustics, Speech and Signal Processing*, 2023.

[5] Stephen H. Schneider and Robert E Dickinson, "Climate modeling," *Reviews of Geophysics*, vol. 12, no. 3, pp. 447-493, 1974.

[6] Yihong Tang, Ao Qu, Andy H. F. Chow, William H. K. Lam, Sze Chun Wong, and Wei Ma, "Domain adversarial spatial-temporal network: A transferable framework for short-term traffic forecasting across cities," *CoRR*, vol. abs/2202.03630, 2022.

[7] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei, "Language models are few-shot learners," in *Advances in Neural Information Processing Systems 33, NeurIPS 2020, December 6-12, 2020, Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, Eds.*, 2020.

[8] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa, "Large language models are zero-shot reasoners," *Advances in neural information processing systems*, vol. 35, pp. 22199-22213, 2022.

[9] Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, and Qingsong Wen, "Time-LLM: Time series forecasting by reprogramming large language models," in *The Twelfth International Conference on Learning Representations*, 2024.

[10] Chenxi Sun, Hongyan Li, Yaliang Li, and Shenda Hong, "TEST: Text prototype aligned embedding to activate LLM's ability for time series," in *The Twelfth International Conference on Learning Representations*, 2024.

[11] Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, and Rong Jin, "One fits all: Power general time series analysis by pretrained LM," in *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.

[12] Zijie Pan, Yushan Jiang, Sahil Garg, Anderson Schneider, Yuriy Nevmyvaka, and Dongjin Song, "$S^2$IP-LLM: Semantic space informed prompt learning with LLM for time series forecasting," in *Forty-first International Conference on Machine Learning*, 2024.

[13] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-Or, "An image is worth one word: Personalizing text-to-image generation using textual inversion," in *The Eleventh International Conference on Learning Representations*, 2023.

[14] Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo, "Reversible instance normalization for accurate time-series forecasting against distribution shift," in *International Conference on Learning Representations*, 2022.

[15] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam, "A time series is worth 64 words: Long-term forecasting with transformers," in *International Conference on Learning Representations*, 2023.

[16] Maria Tsimpoukelli, Jacob Menick, Serkan Cabi, S. M. Ali Eslami, Oriol Vinyals, and Felix Hill, "Multi-modal few-shot learning with frozen language models," in *Advances in Neural Information Processing Systems*, A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, Eds., 2021.

[17] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al., "Language models are unsupervised multitask learners," *OpenAI blog*, vol. 1, no. 8, pp. 9, 2019.

[18] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al., "Llama: Open and efficient foundation language models," *arXiv preprint arXiv:2302.13971*, 2023.

[19] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long, "Timesnet: Temporal 2d-variation modeling for general time series analysis," in *International Conference on Learning Representations*, 2023.

[20] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu, "Are transformers effective for time series forecasting?," in *Proceedings of the AAAI Conference on Artificial Intelligence*, 2023.

[^1]: `https://github.com/thuml/Time-Series-Library`
