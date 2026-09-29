# S²IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting

Zijie Pan^1, Yushan Jiang^1, Sahil Garg^2, Anderson Schneider^2, Yuriy Nevmyvaka^2, and Dongjin Song^1

^1 School of Computing, University of Connecticut, Storrs, USA  
^2 Department of Machine Learning Research, Morgan Stanley, New York, USA

Correspondence to: Yuriy Nevmyvaka `<yuriy.nevmyvaka@morganstanley.com>`, Dongjin Song `<dongjin.song@uconn.edu>`.

*Proceedings of the 41st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).*

**Abstract.** Recently, there has been a growing interest in leveraging pre-trained large language models (LLMs) for various time series applications. However, the semantic space of LLMs, established through the pre-training, is still underexplored and may help yield more distinctive and informative representations to facilitate time series forecasting. To this end, we propose Semantic Space Informed Prompt learning with LLM (S²IP-LLM) to align the pre-trained semantic space with time series embedding space and perform time series forecasting based on learned prompts from the joint space. We first design a tokenization module tailored for cross-modality alignment, which explicitly concatenates patches of decomposed time series components to create embeddings that effectively encode the temporal dynamics. Next, we leverage the pre-trained word token embeddings to derive semantic anchors and align selected anchors with time series embeddings by maximizing the cosine similarity in the joint space. This way, S²IP-LLM can retrieve relevant semantic anchors as prompts to provide strong indicators (context) for time series that exhibit different temporal dynamics. With thorough empirical studies on multiple benchmark datasets, we demonstrate that the proposed S²IP-LLM can achieve superior forecasting performance over state-of-the-art baselines. Furthermore, our ablation studies and visualizations verify the necessity of prompt learning informed by semantic space.

## 1. Introduction

Over the past few years, pre-trained large language models (LLMs) such as GPT-4 (Achiam & et al., 2023) and LLaMA (Touvron et al., 2023b;c) not only achieved great success across a diverse range of natural language processing (NLP) tasks, i.e., generate coherent and contextually relevant text, answer questions, and translate sentences between multiple languages, but also exhibited tremendous potential in tackling applications of more complex or structured domains, such as code generation, healthcare, finance, and autonomous systems, etc (Singhal et al., 2022; Cui et al., 2024; Li et al., 2023). As time series analysis is becoming increasingly important for strategic planning and operational efficiency in various real-world applications, e.g., energy load management, traffic forecasting, weather forecasting, health risk analysis, etc (Friedman, 1962; Courty & Li, 1999; Böse et al., 2017; Gao et al., 2020; Li et al., 2022; Liu et al., 2023a; Dimri et al., 2020), a natural question to ask is whether we should train a general purpose foundation model from scratch, or fine-tune pre-trained LLMs to perform time series forecasting?

Recently, significant efforts have been made to build foundation models for general-purpose time series analysis (Wu et al., 2023; Garza & Mergenthaler-Canseco, 2023; Rasul et al., 2023). TimesNet (Wu et al., 2023) uses TimesBlock as a task-general backbone to capture multi-periodicity and extract complex intraperiod- and interperiod-variations via transformed 2D tensors. TimeGPT-1 describes a general pre-trained model for time series forecasting (Garza & Mergenthaler-Canseco, 2023). These approaches, however, are hindered by two main challenges. First, time series data can be acquired in various formats, such as univariate or multivariate, often in large volumes, and from different domains, like healthcare, finance, traffic, environmental sciences, etc. This escalates the complexity of model training and poses challenges in handling different scenarios. Second, time series data, in practice, often exhibit non-stationary characteristics, resulting in the underlying statistical properties, such as means, variances, and auto-correlations shifting during collection. This could also result in concept drift, where the statistical properties of target variables change over time. These realities present significant challenges for large models to be adapted and retrained effectively.

> Figure 1 (see PDF p. 2). The demonstration of semantic space informed prompting in S²IP-LLM. The input time series is decomposed and mapped to obtain time series (TS) embedding. Next, the TS embedding is aligned with semantic anchors derived from the pre-trained word token embedding. Finally, top-K similar semantic anchors are retrieved and used as prefix-prompts with TS embedding.

On the other hand, LLMs trained on extensive and diverse text corpora can serve as a foundational knowledge base that can be applied to a variety of downstream tasks with minimal task-specific prompt learning or fine-tuning. Inspired by this, there has been a growing interest in leveraging existing LLMs to facilitate time series analysis. For instance, Tian Zhou & Jin (2023) utilizes a frozen pre-trained language model to attain state-of-the-art or equivalent performance. Jin et al. (2024) develop time-LLM to reprogram the input time series via text prototype representations by incorporating the embeddings of the dataset's text descriptions as context information. In real-world applications, however, dataset description information may not always be available or informative. In addition, the patching operation (i.e., tokenization), which splits a long time series sequence into overlapping segments over instance normalized time series input, may have limited expressibility as it could fail to capture the subtle variations of different components in time series.

In this paper, we argue that the semantic space in the form of word token embeddings (based on pre-trained LLMs) can already offer a more distinctive and informative representation space (Ethayarajh, 2019) to help align time series embeddings. Based on this, we develop Semantic Space Informed Prompt with LLM (S²IP-LLM) for time series forecasting. Specifically, as shown in Figure 1, we first design a tokenization module tailored to semantic space alignment, which explicitly concatenates patches of decomposed time series components (i.e., trend, seasonality, and residual) to create an embedding that effectively encodes the temporal dynamics more expressively. Next, we map the pre-trained word embeddings to obtain semantic anchors and align selected anchors with time series embeddings by maximizing the cosine similarity in the joint space. In this way, S²IP-LLM can retrieve relevant semantic anchors as prefix-prompts to provide strong indicators (context) for time series embeddings that exhibit different temporal dynamics. Our experiments over several standard benchmark datasets demonstrate that S²IP-LLM can achieve superior forecasting performance over state-of-the-art baselines. Moreover, our ablation studies and visualizations also verify the necessity of prompt learning in the joint space.

To summarize, our contributions include:

- We design a specialized tokenization module that concatenates patches of decomposed time series components to provide more expressive local contexts and facilitate semantic space informed prompting.
- We leverage semantic anchors derived from pre-trained word token embeddings (semantic space) to align time series embeddings and learn a distinctive and informative joint space. Moreover, aligned semantic anchors are used as prompt indicators (contexts) to enhance the representation of time series.
- Our experiments and analysis on multiple benchmark datasets demonstrate the superiority of S²IP-LLM over state of the art and the necessity of prompt learning informed by semantic space.

## 2. Related Work

### 2.1. Time Series Forecasting

In recent years, a variety of statistical and machine learning methods have been developed for time series analysis, e.g., ARIMA (Anderson & Kendall, 1976), Prophet (Taylor & Letham, 2018), etc. More recently, different types of deep neural networks have been applied for time series analysis. For instance, recurrent neural network (RNN) based models have been developed to capture auto-regressive temporal dynamics (Qin et al., 2017; Li et al., 2017; Lai et al., 2018; Gu et al., 2021). Graph neural networks (GNN) based methods are leveraged to capture variable dependencies among different time series (Cao et al., 2020; Wu et al., 2020; Shang et al., 2021; Pan et al., 2024). Transformer based models leverage the self-attention mechanisms tailored for time series to better capture the temporal dynamics, variable dependencies, or both (Woo et al., 2022; Zhou et al., 2021; Wu et al., 2021; Zhou et al., 2022; Liu et al., 2023b). More recently, MLP-based models (Challu et al., 2023; Zeng et al., 2023) and convolution-based models (Wu et al., 2023) have achieved state-of-the-art performance on par with Transformers, but with much simpler designs. Nevertheless, while these deep forecasters perform well on specific datasets, they lack the flexibility and generalizability to adapt to real-world time series data from different domains.

### 2.2. Pre-trained Large Model for Time Series Analysis

Recent advancements in natural language processing (NLP) and computer vision (CV) demonstrate that pre-trained models can effectively adapt to a range of downstream tasks through fine-tuning (Bao et al., 2021; He et al., 2022; Brown et al., 2020; Devlin et al., 2018). Inspired by this, several different pre-trained models have been developed for time series based on either supervised (Fawaz et al., 2018) or self-supervised learning (Zhang et al., 2022b; Deldari et al., 2022). During the training stage, models can learn robust representations from a variety of input time series data. Then, these models can be fine-tuned for downstream tasks of similar domains to further enhance their performance (Tang et al., 2022). With the emergence and success of Large Language Models (LLMs), including T5 (Raffel et al., 2020), GPT-based models (Radford et al., 2018; 2019; Brown et al., 2020; Ouyang et al., 2022), and LLaMA (Touvron et al., 2023a), which have showcase their robust pattern recognition and reasoning abilities over complex sequences of tokens, there is a trend to explore how to effectively transfer knowledge from these powerful pre-trained LLM models to time series domain (Jiang et al., 2024). One line of research focuses on leveraging the pre-trained LLMs as zero-shot learners. For instance, Xue & Salim (2022) and Nate Gruver & Wilson (2023) directly convert time series data to corresponding text sequence inputs and achieve encouraging results for time series forecasting. Another line of research (Tian Zhou & Jin, 2023; Chang et al., 2023) involves tokenizing the input time series data into overlapping patches and strategically leveraging or fine-tuning LLMs for time series analysis. Following this paradigm, TEST (Sun et al., 2023) and Time-LLM (Jin et al., 2024) reprogram time series data with text prototype embedding and incorporate textual prompts for time series analysis. TEMPO (Cao et al., 2023) incorporates the decomposition of time series and retrieval-based prompt design for non-stationary time series data. Different from those methods, we explicitly leverage semantic anchors derived from pre-trained word token embeddings (semantic space) to align time series embeddings and develop a simple yet effective prompt mechanism to inform LLM for forecasting tasks.

## 3. Methodology

Overview: S²IP-LLM consists of three key components as shown in Figure 2. Given the input time series, we first tokenize it and obtain the time series (TS) embedding based on time series decomposition and patching. Next, we will align the TS embedding with semantic anchors derived from the pre-trained word token embedding. Finally, top-K similar semantic anchors will be retrieved to serve as prefix-prompts for the TS embedding and the concatenated vector will be leveraged as the query for pre-trained LLMs.

In this paper, GPT-2 is used as the backbone. During the training stage, we not only learn the mapping functions of input and output but also fine-tune the positional embedding and layer norm block of GPT-2.

### 3.1. Problem Statement

We first formalize the time series forecasting problem. Let $X \in \mathbb{R}^{N \times T}$ denote the time series data containing $N$ variables and $T$ time steps, where $X_{:,t} \in \mathbb{R}^{N \times 1}$ denotes the $t$-th time step across all variables and $X_{i,:} \in \mathbb{R}^{1 \times T}$ denotes the $i$-th variable. Given a historical $\tau$-step window of time series, we aim to learn a forecasting module $F(\cdot)$ that will predict the next $\tau'$ time steps based on the input window. Mathematically, at a starting time step $t$, the corresponding forecast is given by $\hat{Y} = \hat{X}_{:,t:t+\tau'-1} = F(X_{:,t-\tau:t-1})$.

### 3.2. Time Series Tokenization

In real-world applications, non-stationary data is prevalent. To tackle this problem, we first apply the reversible instance normalization (Kim et al., 2021) on time series input such that the data has zero mean and unit standard deviation to mitigate the distribution shift in time series. Specifically, given the $i$-th time series input at time step $t$, i.e., $X_{i,t}$, the transformed value $X'_{i,t}$ can be given by:

$$
X'_{i,t} = \gamma_T \left( X_{i,t} - \frac{\mathbb{E}_t[X_{i,t}]}{\sqrt{\mathrm{Var}[X_{i,t}] + \epsilon_T}} \right) + \beta_T
\tag{1}
$$

where $\mathbb{E}_t[X_{i,t}]$ and $\mathrm{Var}[X_{i,t}]$ are the instance-specific mean and variance, respectively. $\gamma_T$ and $\beta_T$ are trainable parameters. Next, we adopt an additive seasonal-trend decomposition method to decompose normalized time series into long-term trend, seasonal, and residual components. The additive seasonal-trend decomposition is given by $X'_{i,t} = X_{i,t}^{tre'} + X_{i,t}^{sea'} + X_{i,t}^{res'}$, where `tre`, `sea`, and `res` denote the long-term trend, seasonal, and residual component, respectively. There are several options for additive seasonal-trend decomposition. One option is the classical additive seasonal-trend decomposition that first obtains long-term trend components using moving averages. Then, the seasonal component is estimated by averaging the detrended time series with pre-defined season parameters. Finally, the residual component is obtained by subtracting the estimated trend and seasonal components from the normalized time series. Another option is the Seasonal-Trend decomposition using Loess (STL) (Cleveland et al., 1990). The choice of decomposition method will based on validation results.

> Figure 2 (see PDF p. 4). The model architecture of S²IP-LLM. The input time series is normalized, decomposed, patched individually, and concatenated to represent the context of time series (TS). Semantic space informed prompting performs alignment between the contextual TS embeddings and the semantic anchors extracted from pre-trained word embeddings, and retrieves the most similar $K$ ones as prefix-prompts. The decomposed TS representations from pre-trained LLM are linearly projected and combined as the TS forecast.

Next, we follow Nie et al. (2023) to encode temporal information and local contexts of input time series by aggregating consecutive time steps into overlapped patched tokens. Take the trend component as an example, the normalized component series, $X_{i,t-\tau:t-1}^{tre'} \in \mathbb{R}^{1 \times \tau}$ is converted to patched token representation $P_{i,t-\tau:t-1}^{tre} \in \mathbb{R}^{N_P \times L_P}$, in which $L_P$ is the patch length, $N_P = \left\lfloor \frac{\tau - L_P}{S} \right\rfloor + 2$ is the number of patches and $S$ is the horizontal sliding stride. We apply patching to each variable component over the temporal dimension and then concatenate the tokens of these components into a single meta-token,

$$
P_{i,t-\tau:t-1} = [P_{i,t-\tau:t-1}^{tre}, P_{i,t-\tau:t-1}^{sea}, P_{i,t-\tau:t-1}^{res}] \in \mathbb{R}^{N_P \times 3L_P}.
$$

We then feed the meta-token into a projection layer $g(\cdot)$ to get the time series embedding $P_{i,t-\tau:t-1} = g(P_{i,t-\tau:t-1}) \in \mathbb{R}^{N_P \times D}$, where $D$ is the embedding size for the pre-trained LLMs.

### 3.3. Semantic Space Informed Prompting

Prompting has emerged as an effective technique in various applications, enabling LLMs to utilize task-specific information to achieve enhanced reasoning capabilities (Yin et al., 2023). Existing works primarily focus on employing template-based and fixed prompts for pre-trained LLMs in time series analysis (Xue & Salim, 2022; Jin et al., 2024). While these methods are intuitive, straightforward, and yield satisfactory results, their rigid prompt contexts are in line with linguistic semantics. However, time series representation inherently lacks human semantics and is more closely tied to sequence patterns in the form of temporal dynamics. Conversely, Lester et al. (2021) demonstrate the effectiveness of soft prompts in enabling LLMs to comprehend inputs more effectively. In the realm of time series analysis with LLMs, recent works (Sun et al., 2023; Cao et al., 2023) start to consider soft prompts as task-specific, randomly initialized, trainable vectors that learn from the supervised loss between LLM's output and the ground truth. However, the semantic space of LLMs, established through the pre-training, is still underexplored and may help yield more distinctive and informative representations for time series data.

Based on this intuition, we introduce a prompting mechanism informed by the pre-trained semantic space. Specifically, the pre-trained semantic word token embeddings, represented as $E \in \mathbb{R}^{V \times D}$ where $V$ is the vocabulary size, are inevitably large and dense. For example, the vocabulary size of GPT-2 (Radford et al., 2019) reaches 50,257 and may raise computational deficiency. Instead of directly using the semantic word token embedding, we derive a small set of semantic anchors $E'$ in the hidden space using a generic mapping function $f(\cdot)$ on $E$, which is denoted as $E' = f(E) \in \mathbb{R}^{V' \times D}$, where $V'$ is the reduced number of semantic anchors and $V' \ll V$. To properly retrieve relevant semantic anchors to enhance the time series embedding $P_{i,t-\tau:t-1}$, we align the semantic anchors and time series embedding based on a score-matching function $\gamma(\cdot)$. In this paper, we implement the score-matching function based on cosine similarity:

$$
\gamma(P_{i,t-\tau:t-1}, e'_m) =
\frac{P_{i,t-\tau:t-1} \cdot e'_m}
{\|P_{i,t-\tau:t-1}\| \, \|e'_m\|}
\tag{2}
$$

where $e'_m \in E'$. We select top-$K$ relevant semantic anchors based on the similarity scores and utilize them as prefix-prompt to enhance the input time series embedding, i.e.,

$$
Z_{i,t-\tau:t-1} = [e'_1; \cdots; e'_K; P_{i,t-\tau:t-1}]
= [e'_{\text{top-}K}; P_{i,t-\tau:t-1}]
\tag{3}
$$

which will serve as the input for the pre-trained LLMs.

### 3.4. Optimization Objective

We can obtain the output embedding $Z_{out}$ after the forward path of the prompt enhanced time series embedding through LLMs. We will flatten it and use a linear mapping to project the representation to the forecasting horizon $Y_{out}$. The overall forecasting should also be the additive combination of the individual component predictions due to the decomposition step. We further split and express $Y_{out}$ into a concatenation form $Y_{out} = [Y_{out}^{tre}, Y_{out}^{sea}, Y_{out}^{res}]$ and obtain the forecasting results as $\hat{Y} = Y_{out}^{tre} + Y_{out}^{sea} + Y_{out}^{res}$. At every training iteration, the overall training objective is:

$$
\min \mathcal{L}\left(\hat{Y}, X_{:,t:t+\tau'-1}\right)
- \lambda \sum \gamma\!\left(P_{i,t-\tau:t-1}, e'_{\text{top-}K}\right)
\tag{4}
$$

where the first term is the forecasting loss in the form of mean squared error (MSE), and the second term is a score-matching function to align selected semantic anchors with the time series embedding obtained via decomposition and patching. In this way, we could obtain a more informative space to facilitate the underlying forecasting task. $\lambda \ge 0$ is a hyper-parameter to trade-off the alignment.

### 3.5. Backbone and Fine-tuning Strategy

In this paper, we employ GPT-2 (Radford et al., 2019) as our pre-trained large language model (LLM) backbone. We choose to keep a significant portion of the parameters frozen, especially those parameters related to the multi-headed attention and the feed-forward networks within the Transformer blocks. This strategy can not only reduce the computational burden but also align with existing literature (Lu et al., 2022; Houlsby et al., 2019; Tian Zhou & Jin, 2023). They suggest that maintaining most of the parameters in their non-trainable state can achieve better outcomes compared to completely retraining LLMs. For GPT-2, we only fine-tune the positional embedding layer and the layer-normalization layers.

## 4. Experiments

In our experiments, we compare the proposed S²IP-LLM against a variety of baselines on 11 public datasets. We validate the effectiveness of S²IP-LLM over different time series tasks, including long-term forecasting (Section 4.1), short-term forecasting (Section 4.2), and few-shot forecasting (Section 4.3). We also provide the ablation studies and parameter sensitivity analysis in Section 4.4. Finally, we visualize the prompt enhanced time series embeddings to qualitatively assess the effectiveness of S²IP-LLM. We follow the experimental configurations (Wu et al., 2023) for all baselines using the unified pipeline.[^tslib]

**Baselines.** The baselines include a set of Transformer-based methods, i.e., iTransformer (Liu et al., 2023b), PatchTST (Nie et al., 2023), FEDformer (Zhou et al., 2022), Autoformer (Wu et al., 2021), Non-Stationary Transformer (Liu et al., 2022), ETSformer (Woo et al., 2022) and Informer (Zhou et al., 2021). We also select a set of non-transformer based techniques, i.e., DLinear (Zeng et al., 2023), TimesNet (Wu et al., 2023), and LightTS (Zhang et al., 2022a) for comparison. Finally, two approaches based on LLMs, i.e., OFA (Tian Zhou & Jin, 2023) and Time-LLM (Jin et al., 2024).[^timellm]

### 4.1. Long-term Forecasting

**Setup.** For long-term forecasting, we evaluate the effectiveness of S²IP-LLM on Weather, Electricity, Traffic, and four ETT datasets (i.e., ETTh1, ETTh2, ETTm1, and ETTm2), which have been widely adopted as benchmarking datasets for long-term forecasting tasks. Details of these datasets are shown in Appendix A.3, Table 5. The input time series length is 512, and we evaluate the performance on four different horizons `{96, 192, 336, 720}`. The evaluation metrics include the mean square error (MSE) and the mean absolute error (MAE).

**Results.** We compare the forecasting results of S²IP-LLM to 6 selected baselines in Table 1. Due to the space limitation, the comparisons with the other 6 baselines are provided in Appendix B and Table 6. We can observe that LLMs based forecasting methods, i.e., Time-LLM and OFA, generally achieve better performance than other baseline methods. This should be attributed to the prevalent expressibility of LLMs and their associated prompt-tuning and fine-tuning strategies, respectively. Moreover, most of the time, S²IP-LLM outperforms Time-LLM and OFA over 7 different datasets. This is because (1) the unique way S²IP-LLM tokenized the input time series data can yield better time series representations, and (2) the semantic space informed prompting can help further enhance the time series representation which will be further demonstrated in Section 4.5.

**Table 1.** Long-term forecasting results for `{96, 192, 336, 720}` horizons. Lower values indicate better performance. For Time-LLM variants, `L` denotes the LLaMA backbone (Touvron et al., 2023a), and `G` refers to the GPT-2 backbone (Radford et al., 2019). More results are in Appendix B, Table 6. Values are reported as `MSE / MAE`.

<table>
  <thead>
    <tr>
      <th colspan="2">Methods</th>
      <th colspan="2">S²IP-LLM</th>
      <th colspan="2">Time-LLM(L)</th>
      <th colspan="2">Time-LLM(G)</th>
      <th colspan="2">OFA</th>
      <th colspan="2">iTransformer</th>
      <th colspan="2">DLinear</th>
      <th colspan="2">PatchTST</th>
    </tr>
    <tr>
      <th>Datasets</th>
      <th>Horizon</th>
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
      <td rowspan="5">Weather</td>
      <td>96</td>
      <td>0.145</td>
      <td>0.195</td>
      <td>0.148</td>
      <td>0.197</td>
      <td>0.158</td>
      <td>0.210</td>
      <td>0.162</td>
      <td>0.212</td>
      <td>0.253</td>
      <td>0.304</td>
      <td>0.176</td>
      <td>0.237</td>
      <td>0.149</td>
      <td>0.198</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.190</td>
      <td>0.235</td>
      <td>0.194</td>
      <td>0.246</td>
      <td>0.197</td>
      <td>0.245</td>
      <td>0.204</td>
      <td>0.248</td>
      <td>0.280</td>
      <td>0.319</td>
      <td>0.220</td>
      <td>0.282</td>
      <td>0.194</td>
      <td>0.241</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.243</td>
      <td>0.280</td>
      <td>0.248</td>
      <td>0.285</td>
      <td>0.248</td>
      <td>0.285</td>
      <td>0.254</td>
      <td>0.286</td>
      <td>0.321</td>
      <td>0.344</td>
      <td>0.265</td>
      <td>0.319</td>
      <td>0.245</td>
      <td>0.282</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.312</td>
      <td>0.326</td>
      <td>0.317</td>
      <td>0.332</td>
      <td>0.319</td>
      <td>0.334</td>
      <td>0.326</td>
      <td>0.337</td>
      <td>0.364</td>
      <td>0.374</td>
      <td>0.333</td>
      <td>0.362</td>
      <td>0.314</td>
      <td>0.334</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.222</td>
      <td>0.259</td>
      <td>0.226</td>
      <td>0.265</td>
      <td>0.230</td>
      <td>0.268</td>
      <td>0.237</td>
      <td>0.270</td>
      <td>0.304</td>
      <td>0.335</td>
      <td>0.248</td>
      <td>0.300</td>
      <td>0.225</td>
      <td>0.264</td>
    </tr>
    <tr>
      <td rowspan="5">Electricity</td>
      <td>96</td>
      <td>0.135</td>
      <td>0.230</td>
      <td>0.140</td>
      <td>0.246</td>
      <td>0.137</td>
      <td>0.237</td>
      <td>0.139</td>
      <td>0.238</td>
      <td>0.147</td>
      <td>0.248</td>
      <td>0.140</td>
      <td>0.237</td>
      <td>0.129</td>
      <td>0.222</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.149</td>
      <td>0.247</td>
      <td>0.155</td>
      <td>0.253</td>
      <td>0.150</td>
      <td>0.249</td>
      <td>0.153</td>
      <td>0.251</td>
      <td>0.165</td>
      <td>0.267</td>
      <td>0.153</td>
      <td>0.249</td>
      <td>0.157</td>
      <td>0.240</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.167</td>
      <td>0.266</td>
      <td>0.175</td>
      <td>0.279</td>
      <td>0.168</td>
      <td>0.266</td>
      <td>0.169</td>
      <td>0.266</td>
      <td>0.178</td>
      <td>0.279</td>
      <td>0.169</td>
      <td>0.267</td>
      <td>0.163</td>
      <td>0.259</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.200</td>
      <td>0.287</td>
      <td>0.204</td>
      <td>0.305</td>
      <td>0.203</td>
      <td>0.293</td>
      <td>0.206</td>
      <td>0.297</td>
      <td>0.322</td>
      <td>0.398</td>
      <td>0.203</td>
      <td>0.301</td>
      <td>0.197</td>
      <td>0.290</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.161</td>
      <td>0.257</td>
      <td>0.168</td>
      <td>0.270</td>
      <td>0.164</td>
      <td>0.261</td>
      <td>0.167</td>
      <td>0.263</td>
      <td>0.203</td>
      <td>0.298</td>
      <td>0.166</td>
      <td>0.263</td>
      <td>0.161</td>
      <td>0.252</td>
    </tr>
    <tr>
      <td rowspan="5">Traffic</td>
      <td>96</td>
      <td>0.379</td>
      <td>0.274</td>
      <td>0.383</td>
      <td>0.280</td>
      <td>0.380</td>
      <td>0.277</td>
      <td>0.388</td>
      <td>0.282</td>
      <td>0.367</td>
      <td>0.288</td>
      <td>0.410</td>
      <td>0.282</td>
      <td>0.360</td>
      <td>0.249</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.397</td>
      <td>0.282</td>
      <td>0.399</td>
      <td>0.294</td>
      <td>0.399</td>
      <td>0.288</td>
      <td>0.407</td>
      <td>0.290</td>
      <td>0.378</td>
      <td>0.293</td>
      <td>0.423</td>
      <td>0.287</td>
      <td>0.379</td>
      <td>0.256</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.407</td>
      <td>0.289</td>
      <td>0.411</td>
      <td>0.306</td>
      <td>0.408</td>
      <td>0.290</td>
      <td>0.412</td>
      <td>0.294</td>
      <td>0.389</td>
      <td>0.294</td>
      <td>0.436</td>
      <td>0.296</td>
      <td>0.392</td>
      <td>0.264</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.440</td>
      <td>0.301</td>
      <td>0.448</td>
      <td>0.319</td>
      <td>0.445</td>
      <td>0.308</td>
      <td>0.450</td>
      <td>0.312</td>
      <td>0.401</td>
      <td>0.304</td>
      <td>0.466</td>
      <td>0.315</td>
      <td>0.432</td>
      <td>0.286</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.405</td>
      <td>0.286</td>
      <td>0.440</td>
      <td>0.301</td>
      <td>0.408</td>
      <td>0.290</td>
      <td>0.414</td>
      <td>0.294</td>
      <td>0.389</td>
      <td>0.295</td>
      <td>0.433</td>
      <td>0.295</td>
      <td>0.390</td>
      <td>0.263</td>
    </tr>
    <tr>
      <td rowspan="5">ETTh1</td>
      <td>96</td>
      <td>0.366</td>
      <td>0.396</td>
      <td>0.380</td>
      <td>0.406</td>
      <td>0.383</td>
      <td>0.410</td>
      <td>0.379</td>
      <td>0.402</td>
      <td>0.395</td>
      <td>0.420</td>
      <td>0.367</td>
      <td>0.396</td>
      <td>0.379</td>
      <td>0.407</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.401</td>
      <td>0.420</td>
      <td>0.426</td>
      <td>0.438</td>
      <td>0.419</td>
      <td>0.435</td>
      <td>0.415</td>
      <td>0.424</td>
      <td>0.427</td>
      <td>0.441</td>
      <td>0.401</td>
      <td>0.419</td>
      <td>0.428</td>
      <td>0.442</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.412</td>
      <td>0.431</td>
      <td>0.437</td>
      <td>0.451</td>
      <td>0.426</td>
      <td>0.440</td>
      <td>0.435</td>
      <td>0.440</td>
      <td>0.445</td>
      <td>0.457</td>
      <td>0.434</td>
      <td>0.449</td>
      <td>0.465</td>
      <td>0.465</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.440</td>
      <td>0.458</td>
      <td>0.515</td>
      <td>0.509</td>
      <td>0.428</td>
      <td>0.456</td>
      <td>0.441</td>
      <td>0.459</td>
      <td>0.537</td>
      <td>0.530</td>
      <td>0.472</td>
      <td>0.493</td>
      <td>0.504</td>
      <td>0.500</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.406</td>
      <td>0.427</td>
      <td>0.439</td>
      <td>0.451</td>
      <td>0.414</td>
      <td>0.435</td>
      <td>0.418</td>
      <td>0.431</td>
      <td>0.451</td>
      <td>0.462</td>
      <td>0.418</td>
      <td>0.439</td>
      <td>0.444</td>
      <td>0.453</td>
    </tr>
    <tr>
      <td rowspan="5">ETTh2</td>
      <td>96</td>
      <td>0.278</td>
      <td>0.340</td>
      <td>0.306</td>
      <td>0.362</td>
      <td>0.297</td>
      <td>0.357</td>
      <td>0.289</td>
      <td>0.347</td>
      <td>0.304</td>
      <td>0.360</td>
      <td>0.301</td>
      <td>0.367</td>
      <td>0.296</td>
      <td>0.353</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.346</td>
      <td>0.385</td>
      <td>0.346</td>
      <td>0.385</td>
      <td>0.349</td>
      <td>0.390</td>
      <td>0.358</td>
      <td>0.392</td>
      <td>0.377</td>
      <td>0.403</td>
      <td>0.394</td>
      <td>0.427</td>
      <td>0.382</td>
      <td>0.404</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.367</td>
      <td>0.406</td>
      <td>0.393</td>
      <td>0.422</td>
      <td>0.373</td>
      <td>0.408</td>
      <td>0.383</td>
      <td>0.414</td>
      <td>0.405</td>
      <td>0.429</td>
      <td>0.506</td>
      <td>0.495</td>
      <td>0.402</td>
      <td>0.425</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.400</td>
      <td>0.436</td>
      <td>0.397</td>
      <td>0.433</td>
      <td>0.400</td>
      <td>0.436</td>
      <td>0.438</td>
      <td>0.456</td>
      <td>0.443</td>
      <td>0.464</td>
      <td>0.805</td>
      <td>0.635</td>
      <td>0.444</td>
      <td>0.465</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.347</td>
      <td>0.391</td>
      <td>0.360</td>
      <td>0.400</td>
      <td>0.355</td>
      <td>0.398</td>
      <td>0.367</td>
      <td>0.402</td>
      <td>0.382</td>
      <td>0.414</td>
      <td>0.502</td>
      <td>0.481</td>
      <td>0.381</td>
      <td>0.411</td>
    </tr>
    <tr>
      <td rowspan="5">ETTm1</td>
      <td>96</td>
      <td>0.288</td>
      <td>0.346</td>
      <td>0.311</td>
      <td>0.365</td>
      <td>0.291</td>
      <td>0.346</td>
      <td>0.296</td>
      <td>0.353</td>
      <td>0.312</td>
      <td>0.366</td>
      <td>0.304</td>
      <td>0.348</td>
      <td>0.303</td>
      <td>0.351</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.323</td>
      <td>0.365</td>
      <td>0.364</td>
      <td>0.395</td>
      <td>0.336</td>
      <td>0.373</td>
      <td>0.335</td>
      <td>0.373</td>
      <td>0.347</td>
      <td>0.385</td>
      <td>0.336</td>
      <td>0.367</td>
      <td>0.341</td>
      <td>0.376</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.359</td>
      <td>0.390</td>
      <td>0.369</td>
      <td>0.398</td>
      <td>0.362</td>
      <td>0.390</td>
      <td>0.369</td>
      <td>0.394</td>
      <td>0.379</td>
      <td>0.404</td>
      <td>0.368</td>
      <td>0.387</td>
      <td>0.377</td>
      <td>0.401</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.403</td>
      <td>0.418</td>
      <td>0.416</td>
      <td>0.425</td>
      <td>0.410</td>
      <td>0.421</td>
      <td>0.418</td>
      <td>0.424</td>
      <td>0.441</td>
      <td>0.442</td>
      <td>0.421</td>
      <td>0.418</td>
      <td>0.431</td>
      <td>0.436</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.343</td>
      <td>0.379</td>
      <td>0.365</td>
      <td>0.395</td>
      <td>0.349</td>
      <td>0.382</td>
      <td>0.355</td>
      <td>0.386</td>
      <td>0.370</td>
      <td>0.399</td>
      <td>0.357</td>
      <td>0.389</td>
      <td>0.363</td>
      <td>0.391</td>
    </tr>
    <tr>
      <td rowspan="5">ETTm2</td>
      <td>96</td>
      <td>0.165</td>
      <td>0.257</td>
      <td>0.170</td>
      <td>0.262</td>
      <td>0.184</td>
      <td>0.275</td>
      <td>0.170</td>
      <td>0.264</td>
      <td>0.179</td>
      <td>0.271</td>
      <td>0.168</td>
      <td>0.263</td>
      <td>0.173</td>
      <td>0.262</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.222</td>
      <td>0.299</td>
      <td>0.229</td>
      <td>0.303</td>
      <td>0.238</td>
      <td>0.310</td>
      <td>0.231</td>
      <td>0.306</td>
      <td>0.242</td>
      <td>0.313</td>
      <td>0.229</td>
      <td>0.310</td>
      <td>0.231</td>
      <td>0.300</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.277</td>
      <td>0.330</td>
      <td>0.281</td>
      <td>0.335</td>
      <td>0.286</td>
      <td>0.340</td>
      <td>0.280</td>
      <td>0.339</td>
      <td>0.288</td>
      <td>0.344</td>
      <td>0.289</td>
      <td>0.352</td>
      <td>0.292</td>
      <td>0.345</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.363</td>
      <td>0.390</td>
      <td>0.379</td>
      <td>0.403</td>
      <td>0.379</td>
      <td>0.403</td>
      <td>0.373</td>
      <td>0.402</td>
      <td>0.378</td>
      <td>0.397</td>
      <td>0.416</td>
      <td>0.437</td>
      <td>0.371</td>
      <td>0.394</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.257</td>
      <td>0.319</td>
      <td>0.264</td>
      <td>0.325</td>
      <td>0.271</td>
      <td>0.332</td>
      <td>0.265</td>
      <td>0.328</td>
      <td>0.272</td>
      <td>0.331</td>
      <td>0.275</td>
      <td>0.340</td>
      <td>0.267</td>
      <td>0.325</td>
    </tr>
  </tbody>
</table>
### 4.2. Short-term Forecasting

**Setup.** We also evaluate the effectiveness of S²IP-LLM with the short-term forecasting setting based on the M4 datasets (Makridakis et al., 2018). It contains a collection of marketing data that are sampled at different frequencies. Details of these datasets can be found in Appendix A.3. The prediction horizons are significantly shorter than the long-term forecasting setting and are set between 6 and 48. The input lengths are twice the prediction horizons, similar to the experiment setting in (Jin et al., 2024; Tian Zhou & Jin, 2023). The evaluation metrics for short-term forecasting are symmetric mean absolute percentage error (SMAPE), mean absolute scaled error (MASE), and overall weighted average (OWA). The details of these evaluation metrics are provided in Appendix A.4.

**Results.** Table 2 summarizes the short-term forecasting results and the full experiment results are shown in Appendix C, Table 7. We observe that S²IP-LLM outperforms all other baselines by a large margin and is slightly better than PatchTST. This could attribute to the tokenization design as well as the semantic space informed prompting within S²IP-LLM.

<table>
  <thead>
    <tr>
      <th rowspan="2">Avg.</th>
      <th rowspan="2">Metric</th>
      <th>S²IP-LLM</th>
      <th>Time-LLM(G)</th>
      <th>OFA</th>
      <th>iTransformer</th>
      <th>DLinear</th>
      <th>PatchTST</th>
      <th>TimesNet</th>
      <th>FEDformer</th>
      <th>Autoformer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Avg.</td>
      <td>SMAPE</td>
      <td>12.021</td>
      <td>12.494</td>
      <td>12.690</td>
      <td>12.142</td>
      <td>13.639</td>
      <td>12.059</td>
      <td>12.880</td>
      <td>13.160</td>
      <td>12.909</td>
    </tr>
    <tr>
      <td>MASE</td>
      <td>1.612</td>
      <td>1.731</td>
      <td>1.808</td>
      <td>1.631</td>
      <td>2.095</td>
      <td>1.623</td>
      <td>1.836</td>
      <td>1.775</td>
      <td>1.771</td>
    </tr>
    <tr>
      <td>OWA</td>
      <td>0.857</td>
      <td>0.913</td>
      <td>0.940</td>
      <td>0.874</td>
      <td>1.051</td>
      <td>0.869</td>
      <td>0.955</td>
      <td>0.949</td>
      <td>0.939</td>
    </tr>
  </tbody>
</table>
> Table 2 (see PDF p. 7). Short-term time series forecasting results on M4 datasets. The forecasting horizons are in `[6, 48]` and the three rows provided are weighted averaged from all datasets under different sampling intervals. A lower value indicates better performance. Detailed short-term forecasting results are in Appendix C, Table 7.

### 4.3. Few-shot Forecasting

**Setup.** We follow the experimental settings in Tian Zhou & Jin (2023) to evaluate the performance in the few-shot forecasting setting, which allows us to examine whether the model can generate accurate forecasting with limited training data. We use the first 5% and 10% of the training data in these experiments.

**Results.** To ensure a fair comparison in the long-term forecasting setting, we summarize the few-shot learning experiment results under 10% and 5% training data in Table 3 and Table 4, respectively. We also report the full experiment results in Table 8 and Table 9 of Appendix D, respectively. When trained with only 10% of the data, S²IP-LLM typically ranks as either the best or the second-best compared to other baseline models across different datasets. Meanwhile, we also observe that LLMs based methods, S²IP-LLM, Time-LLM, and OFA significantly outperform other baseline methods. This is because other baseline methods are trained from scratch and they only have limited training data in this case. On the other hand, LLMs based methods can adapt/align the pre-trained knowledge with the time series embedding to enhance its representation. Even with only 5% of training data, S²IP-LLM still exhibits, if not superior, comparable performance to time-LLM and OFA.

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">S²IP-LLM</th>
      <th colspan="2">Time-LLM(G)</th>
      <th colspan="2">OFA</th>
      <th colspan="2">iTransformer</th>
      <th colspan="2">DLinear</th>
      <th colspan="2">PatchTST</th>
      <th colspan="2">TimesNet</th>
    </tr>
    <tr>
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
      <td>Weather</td>
      <td>0.233</td>
      <td>0.272</td>
      <td>0.237</td>
      <td>0.275</td>
      <td>0.238</td>
      <td>0.275</td>
      <td>0.308</td>
      <td>0.338</td>
      <td>0.241</td>
      <td>0.283</td>
      <td>0.242</td>
      <td>0.279</td>
      <td>0.279</td>
      <td>0.301</td>
    </tr>
    <tr>
      <td>Electricity</td>
      <td>0.175</td>
      <td>0.271</td>
      <td>0.177</td>
      <td>0.273</td>
      <td>0.176</td>
      <td>0.269</td>
      <td>0.196</td>
      <td>0.293</td>
      <td>0.180</td>
      <td>0.280</td>
      <td>0.180</td>
      <td>0.273</td>
      <td>0.323</td>
      <td>0.392</td>
    </tr>
    <tr>
      <td>Traffic</td>
      <td>0.427</td>
      <td>0.307</td>
      <td>0.429</td>
      <td>0.307</td>
      <td>0.440</td>
      <td>0.310</td>
      <td>0.495</td>
      <td>0.361</td>
      <td>0.447</td>
      <td>0.313</td>
      <td>0.430</td>
      <td>0.305</td>
      <td>0.951</td>
      <td>0.535</td>
    </tr>
    <tr>
      <td>ETTh1</td>
      <td>0.593</td>
      <td>0.529</td>
      <td>0.785</td>
      <td>0.553</td>
      <td>0.590</td>
      <td>0.525</td>
      <td>0.910</td>
      <td>0.860</td>
      <td>0.691</td>
      <td>0.600</td>
      <td>0.633</td>
      <td>0.542</td>
      <td>0.869</td>
      <td>0.628</td>
    </tr>
    <tr>
      <td>ETTh2</td>
      <td>0.419</td>
      <td>0.439</td>
      <td>0.424</td>
      <td>0.441</td>
      <td>0.397</td>
      <td>0.421</td>
      <td>0.489</td>
      <td>0.483</td>
      <td>0.605</td>
      <td>0.538</td>
      <td>0.415</td>
      <td>0.431</td>
      <td>0.479</td>
      <td>0.465</td>
    </tr>
    <tr>
      <td>ETTm1</td>
      <td>0.455</td>
      <td>0.435</td>
      <td>0.487</td>
      <td>0.461</td>
      <td>0.464</td>
      <td>0.441</td>
      <td>0.728</td>
      <td>0.565</td>
      <td>0.411</td>
      <td>0.429</td>
      <td>0.501</td>
      <td>0.466</td>
      <td>0.677</td>
      <td>0.537</td>
    </tr>
    <tr>
      <td>ETTm2</td>
      <td>0.284</td>
      <td>0.332</td>
      <td>0.305</td>
      <td>0.344</td>
      <td>0.293</td>
      <td>0.335</td>
      <td>0.336</td>
      <td>0.373</td>
      <td>0.316</td>
      <td>0.368</td>
      <td>0.296</td>
      <td>0.343</td>
      <td>0.320</td>
      <td>0.353</td>
    </tr>
  </tbody>
</table>
> Table 3 (see PDF p. 7). Long-term forecasting results for `{96, 192, 336, 720}` horizons. A lower value indicates a better performance. Few-shot learning on 10% training data setting. All results are averaged from four forecasting horizons `{96, 192, 336, 720}`. Detailed results are in Appendix C, Table 8.

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">S²IP-LLM</th>
      <th colspan="2">Time-LLM(G)</th>
      <th colspan="2">OFA</th>
      <th colspan="2">iTransformer</th>
      <th colspan="2">DLinear</th>
      <th colspan="2">PatchTST</th>
      <th colspan="2">TimesNet</th>
    </tr>
    <tr>
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
      <td>Weather</td>
      <td>0.260</td>
      <td>0.297</td>
      <td>0.264</td>
      <td>0.301</td>
      <td>0.263</td>
      <td>0.301</td>
      <td>0.309</td>
      <td>0.339</td>
      <td>0.263</td>
      <td>0.308</td>
      <td>0.269</td>
      <td>0.303</td>
      <td>0.298</td>
      <td>0.318</td>
    </tr>
    <tr>
      <td>Electricity</td>
      <td>0.179</td>
      <td>0.275</td>
      <td>0.181</td>
      <td>0.279</td>
      <td>0.178</td>
      <td>0.273</td>
      <td>0.201</td>
      <td>0.296</td>
      <td>0.176</td>
      <td>0.275</td>
      <td>0.181</td>
      <td>0.277</td>
      <td>0.402</td>
      <td>0.453</td>
    </tr>
    <tr>
      <td>Traffic</td>
      <td>0.420</td>
      <td>0.299</td>
      <td>0.423</td>
      <td>0.302</td>
      <td>0.434</td>
      <td>0.305</td>
      <td>0.450</td>
      <td>0.324</td>
      <td>0.450</td>
      <td>0.317</td>
      <td>0.418</td>
      <td>0.296</td>
      <td>0.867</td>
      <td>0.493</td>
    </tr>
    <tr>
      <td>ETTh1</td>
      <td>0.650</td>
      <td>0.550</td>
      <td>0.891</td>
      <td>0.627</td>
      <td>0.681</td>
      <td>0.560</td>
      <td>1.070</td>
      <td>0.710</td>
      <td>0.750</td>
      <td>0.611</td>
      <td>0.694</td>
      <td>0.569</td>
      <td>0.925</td>
      <td>0.647</td>
    </tr>
    <tr>
      <td>ETTh2</td>
      <td>0.380</td>
      <td>0.413</td>
      <td>0.581</td>
      <td>0.519</td>
      <td>0.400</td>
      <td>0.433</td>
      <td>0.488</td>
      <td>0.475</td>
      <td>0.694</td>
      <td>0.577</td>
      <td>0.827</td>
      <td>0.615</td>
      <td>0.439</td>
      <td>0.448</td>
    </tr>
    <tr>
      <td>ETTm1</td>
      <td>0.455</td>
      <td>0.446</td>
      <td>0.524</td>
      <td>0.479</td>
      <td>0.472</td>
      <td>0.450</td>
      <td>0.784</td>
      <td>0.596</td>
      <td>0.400</td>
      <td>0.417</td>
      <td>0.526</td>
      <td>0.476</td>
      <td>0.717</td>
      <td>0.561</td>
    </tr>
    <tr>
      <td>ETTm2</td>
      <td>0.296</td>
      <td>0.342</td>
      <td>0.325</td>
      <td>0.361</td>
      <td>0.308</td>
      <td>0.346</td>
      <td>0.356</td>
      <td>0.388</td>
      <td>0.399</td>
      <td>0.426</td>
      <td>0.314</td>
      <td>0.352</td>
      <td>0.344</td>
      <td>0.372</td>
    </tr>
  </tbody>
</table>
> Table 4 (see PDF p. 8). Long-term forecasting results for `{96, 192, 336, 720}` horizons. A lower value indicates a better performance. Few-shot learning on 5% training data setting. All results are averaged from four forecasting horizons `{96, 192, 336, 720}`. Detailed results are in Appendix C, Table 9.

### 4.4. Ablation Studies and Parameter Sensitivity

We conduct ablation studies on the ETTh2 and ETTm2 datasets to evaluate the parameter sensitivity for S²IP-LLM. Figure 3 (1) and (4) presents the experiment results when the length of the prompt varies on ETTh2 and ETTm2, respectively. Within a limited range, i.e. 2 to 8, an increase in the prompt length tends to improve the forecasting performance. However, excessive prompt length, such as lengths of 16 or 32, results in a significant decline in the forecasting accuracy. A similar pattern can be observed in the hyper-parameter analysis of the $\lambda$, which controls the strength of alignment. As shown in Figure 3 (2) and (5), when $\lambda$ varies from 0 to 0.05, slightly larger $\lambda$ is beneficial for representation learning within the joint space, showing better forecasting results. On the other hand, larger $\lambda$ tends to lead to indistinguishable time series representation and the forecasting performance will thus decrease. Finally, Figure 3 (3) and (6) show the effects of choosing different numbers of semantic anchors. Generally, an increased number of semantic anchors improves the forecasting results. We conjecture that the small number hinders the learning of highly representative semantic anchors in the joint space and thus will generate less informed prompts for time series embedding. We visualize the prompted time series embeddings with the different number of semantic anchors in Appendix E, Figure 6. We notice that a smaller quantity of semantic anchors leads to a less dispersed distribution in the joint space, indicating that the generated prompts could be less informative for time series embedding. We also perform ablation studies by incrementally adding the "alignment & prompting" and "decomposition" modules. In Appendix E Table 10, we observe the forecasting performance increases when we sequentially activate the prompting & alignment component and the decomposition component, which implies the importance of these modules in S²IP-LLM.

> Figure 3 (see PDF p. 8). Parameter sensitivity analysis in predicting 96 and 192 steps: (1) and (4) show the effect of prompt length on ETTh2 and ETTm2 datasets; (2) and (5) show the effect of alignment coefficient $\lambda$ on ETTh2 and ETTm2 datasets; (3) and (6) show the effect of semantic space size $V'$ on ETTh2 and ETTm2 datasets.

### 4.5. Qualitative Analysis

In this section, we perform a qualitative analysis of how semantic space informed prompting can facilitate time series representation. Figure 4 shows the visualization of learned semantic anchor embeddings, time series embeddings, and the prompted time series embeddings. The semantic anchor embeddings from the pre-trained language model show distinct clusters, suggesting a robust and differentiated embedding space. In contrast, the raw time series embeddings reveal a more spread-out and less clustered pattern, suggesting that before the alignment, the time series representation is comparatively less informative. After the alignment, the prompted time series embeddings show a clear clustered pattern, suggesting that by aligning with the semantic anchors, time series representation becomes more distinguishable in the joint space.

We also provide the visualizations of prompted time series embeddings under different hyperparameters (when $\lambda$ varies). Within a smaller range, the increase of $\lambda$ appears to enhance the separation of time series embeddings, indicating a more distinct and informative representation. However, as $\lambda$ becomes excessively large, we observe a significant decline in the clustering quality of the prompted time series embeddings, which suggests that beyond a threshold, a higher $\lambda$ value leads to less informative embeddings.

> Figure 4 (see PDF p. 9). The t-SNE and PCA plots of embeddings space: blue: semantic anchor embeddings; red: time series embeddings; orange: prefix-prompted time series embeddings.

> Figure 5 (see PDF p. 9). The t-SNE and PCA plots prefix-prompted time series embeddings with different $\lambda$.

## 5. Conclusion

In this paper, we present S²IP-LLM, a novel framework for time series forecasting utilizing pre-trained language models. S²IP-LLM introduces a time series tokenization module that provides expressive local contexts by the concatenation of decomposed time series patches. It creates informative joint space by aligning time series contexts with semantics anchors derived from the pre-trained word token embeddings. The selected aligned semantic anchors are retrieved as prompt indicators to enhance the time series representation and facilitate underlying forecasting tasks. Our thorough empirical studies justified the effectiveness of S²IP-LLM.

## Impact Statement

This work introduces significant advancements in time series forecasting, leveraging the power of pre-trained language models and semantic information. The broader impact of this work can be multifaceted. It may enhance decision-making in critical domains such as finance, healthcare, and environmental monitoring by providing more accurate and reliable forecasts and could lead to better resource allocation, improved patient care, and more effective responses to climate change. No ethical concerns must be considered. The social impacts are significant, as it has the potential to revolutionize our approach to complex time series data and the integration of emerging AI tools, including foundational models. It could change how we analyze and leverage time series data in various fields.

## References

<pre>
Achiam, O. J. and et al., S. A.
GPT-4 technical report.
2023.
URL https://api.semanticscholar.
org/CorpusID:257532815.

Anderson, O. D. and Kendall, M. G.
Time-series. 2nd
edn.
The Statistician, 25:308, 1976.
URL https:
//api.semanticscholar.org/CorpusID:
134001785.

Bao, H., Dong, L., Piao, S., and Wei, F.
Beit: Bert
pre-training of image transformers.
arXiv preprint
arXiv:2106.08254, 2021.

Böse, J.-H., Flunkert, V., Gasthaus, J., Januschowski, T.,
Lange, D., Salinas, D., Schelter, S., Seeger, M., and Wang,
Y. Probabilistic demand forecasting at scale. Proceedings
of the VLDB Endowment, 10(12):1694–1705, 2017.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D.,
Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
Askell, A., et al. Language models are few-shot learners.
Advances in neural information processing systems, 33:
1877–1901, 2020.

Cao, D., Wang, Y., Duan, J., Zhang, C., Zhu, X., Huang,
C., Tong, Y., Xu, B., Bai, J., Tong, J., et al. Spectral tem-
poral graph neural network for multivariate time-series
forecasting. Advances in neural information processing
systems, 33:17766–17778, 2020.

Cao, D., Jia, F., Arik, S. O., Pfister, T., Zheng, Y., Ye, W.,
and Liu, Y. Tempo: Prompt-based generative pre-trained
transformer for time series forecasting. arXiv preprint
arXiv:2310.04948, 2023.

Challu, C., Olivares, K. G., Oreshkin, B. N., Ramirez, F. G.,
Canseco, M. M., and Dubrawski, A. Nhits: Neural hi-
erarchical interpolation for time series forecasting. In
Proceedings of the AAAI Conference on Artificial Intelli-
gence, volume 37, pp. 6989–6997, 2023.

Chang, C., Peng, W.-C., and Chen, T.-F. Llm4ts: Two-stage
fine-tuning for time-series forecasting with pre-trained
llms. arXiv preprint arXiv:2308.08469, 2023.

Cleveland, R. B., Cleveland, W. S., McRae, J. E., and Ter-
penning, I. Stl: A seasonal-trend decomposition. J. Off.
Stat, 6(1):3–73, 1990.

Courty, P. and Li, H. Timing of seasonal sales. The Journal
of Business, 72(4):545–572, 1999.

Cui, C., Ma, Y., Cao, X., Ye, W., Zhou, Y., Liang, K., Chen,
J., Lu, J., Yang, Z., Liao, K.-D., et al. A survey on mul-
timodal large language models for autonomous driving.
In Proceedings of the IEEE/CVF Winter Conference on
Applications of Computer Vision, pp. 958–979, 2024.

Deldari, S., Xue, H., Saeed, A., He, J., Smith, D. V., and
Salim, F. D.
Beyond just vision: A review on self-
supervised representation learning on multimodal and
temporal data. arXiv preprint arXiv:2206.02353, 2022.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert:
Pre-training of deep bidirectional transformers for lan-
guage understanding. arXiv preprint arXiv:1810.04805,
2018.

Dimri, T., Ahmad, S., and Sharif, M. Time series analysis of
climate variables using seasonal arima approach. Journal
of Earth System Science, 129:1–16, 2020.

Ethayarajh, K. How contextual are contextualized word
representations? comparing the geometry of bert, elmo,
and gpt-2 embeddings. arXiv preprint arXiv:1909.00512,
2019.

Fawaz, H. I., Forestier, G., Weber, J., Idoumghar, L., and
Muller, P.-A. Transfer learning for time series classifica-
tion. In 2018 IEEE international conference on big data
(Big Data), pp. 1367–1376. IEEE, 2018.

Friedman, M. The interpolation of time series by related
series. Journal of the American Statistical Association,
57(300):729–757, 1962.

Gao, J., Song, X., Wen, Q., Wang, P., Sun, L., and Xu,
H. Robusttad: Robust time series anomaly detection via
decomposition and convolutional neural networks. arXiv
preprint arXiv:2002.09545, 2020.

Garza, A. and Mergenthaler-Canseco, M. Timegpt-1. arXiv
preprint arXiv:2310.03589, 2023.

Gu, A., Goel, K., and Ré, C. Efficiently modeling long
sequences with structured state spaces. arXiv preprint
arXiv:2111.00396, 2021.

He, K., Chen, X., Xie, S., Li, Y., Dollár, P., and Girshick,
R. Masked autoencoders are scalable vision learners. In
Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pp. 16000–16009, 2022.

Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B.,
De Laroussilhe, Q., Gesmundo, A., Attariyan, M., and
Gelly, S. Parameter-efficient transfer learning for nlp.
In International Conference on Machine Learning, pp.
2790–2799. PMLR, 2019.

Jiang, Y., Pan, Z., Zhang, X., Garg, S., Schneider, A.,
Nevmyvaka, Y., and Song, D. Empowering time series
analysis with large language models: A survey. arXiv
preprint arXiv:2402.03182, 2024.

Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J. Y., Shi, X.,
Chen, P.-Y., Liang, Y., Li, Y.-F., Pan, S., et al. Time-LLM:
Time series forecasting by reprogramming large language
models. In International Conference on Learning Repre-
sentations, 2024.

Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., and Choo, J.
Reversible instance normalization for accurate time-series
forecasting against distribution shift. In International
Conference on Learning Representations, 2021.

Lai, G., Chang, W.-C., Yang, Y., and Liu, H. Modeling
long-and short-term temporal patterns with deep neural
networks. In The 41st international ACM SIGIR confer-
ence on research & development in information retrieval,
pp. 95–104, 2018.

Lester, B., Al-Rfou, R., and Constant, N. The power of scale
for parameter-efficient prompt tuning. arXiv preprint
arXiv:2104.08691, 2021.

Li, N., Arnold, D. M., Down, D. G., Barty, R., Blake, J.,
Chiang, F., Courtney, T., Waito, M., Trifunov, R., and
Heddle, N. M. From demand forecasting to inventory
ordering decisions for red blood cells through integrat-
ing machine learning, statistical modeling, and inventory
optimization. Transfusion, 62(1):87–99, 2022.

Li, Y., Yu, R., Shahabi, C., and Liu, Y. Diffusion con-
volutional recurrent neural network: Data-driven traffic
forecasting. arXiv preprint arXiv:1707.01926, 2017.

Li, Y., Wang, S., Ding, H., and Chen, H. Large language
models in finance: A survey. In Proceedings of the Fourth
ACM International Conference on AI in Finance, pp. 374–
382, 2023.

Liu, H., Ma, Z., Yang, L., Zhou, T., Xia, R., Wang, Y., Wen,
Q., and Sun, L. Sadi: A self-adaptive decomposed inter-
pretable framework for electric load forecasting under ex-
treme events. In ICASSP 2023-2023 IEEE International
Conference on Acoustics, Speech and Signal Processing
(ICASSP), pp. 1–5. IEEE, 2023a.

Liu, Y., Wu, H., Wang, J., and Long, M. Non-stationary
transformers: Exploring the stationarity in time series
forecasting. Advances in Neural Information Processing
Systems, 35:9881–9893, 2022.

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L.,
and Long, M. itransformer: Inverted transformers are
effective for time series forecasting.
arXiv preprint
arXiv:2310.06625, 2023b.

Lu, K., Grover, A., Abbeel, P., and Mordatch, I. Frozen
pretrained transformers as universal computation engines.
In Proceedings of the AAAI Conference on Artificial In-
telligence, volume 36, pp. 7628–7636, 2022.

Makridakis, S., Spiliotis, E., and Assimakopoulos, V. The
m4 competition: Results, findings, conclusion and way
forward. International Journal of Forecasting, 34(4):
802–808, 2018.

Nate Gruver, Marc Finzi, S. Q. and Wilson, A. G. Large
Language Models Are Zero Shot Time Series Forecasters.
In Advances in Neural Information Processing Systems,
2023.

Nie, Y., H. Nguyen, N., Sinthong, P., and Kalagnanam, J. A
time series is worth 64 words: Long-term forecasting with
transformers. In International Conference on Learning
Representations, 2023.

Oreshkin, B. N., Carpov, D., Chapados, N., and Bengio, Y.
N-beats: Neural basis expansion analysis for interpretable
time series forecasting. arXiv preprint arXiv:1905.10437,
2019.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.,
Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A.,
et al. Training language models to follow instructions
with human feedback. Advances in Neural Information
Processing Systems, 35:27730–27744, 2022.

Pan, Z., Jiang, Y., Song, D., Garg, S., Rasul, K., Schneider,
A., and Nevmyvaka, Y. Structural knowledge informed
continual multivariate time series forecasting.
arXiv
preprint arXiv:2402.12722, 2024.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J.,
Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga,
L., et al. PyTorch: An imperative style, high-performance
deep learning library. Advances in neural information
processing systems, 32, 2019.

Qin, Y., Song, D., Cheng, H., Cheng, W., Jiang, G., and
Cottrell, G. W. A dual-stage attention-based recurrent
neural network for time series prediction. In Proceedings
of the 26th International Joint Conference on Artificial
Intelligence, pp. 2627–2633, 2017.

Radford, A., Narasimhan, K., Salimans, T., Sutskever, I.,
et al. Improving language understanding by generative
pre-training. 2018.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D.,
Sutskever, I., et al. Language models are unsupervised
multitask learners. OpenAI blog, 1(8):9, 2019.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S.,
Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring
the limits of transfer learning with a unified text-to-text
transformer. The Journal of Machine Learning Research,
21(1):5485–5551, 2020.

Rasul, K., Ashok, A., Williams, A. R., Khorasani, A.,
Adamopoulos, G., Bhagwatkar, R., Bilòs, M., Ghonia, H.,
Hassen, N. V., Schneider, A., et al. Lag-llama: Towards
foundation models for time series forecasting. arXiv
preprint arXiv:2310.08278, 2023.

Shang, C., Chen, J., and Bi, J. Discrete graph structure learn-
ing for forecasting multiple time series. arXiv preprint
arXiv:2101.06861, 2021.

Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung,
H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S.,
et al. Large language models encode clinical knowledge.
arXiv preprint arXiv:2212.13138, 2022.

Sun, C., Li, Y., Li, H., and Hong, S. Test: Text prototype
aligned embedding to activate llm's ability for time series.
arXiv preprint arXiv:2308.08241, 2023.

Tang, Y., Qu, A., Chow, A. H., Lam, W. H., Wong, S.,
and Ma, W. Domain adversarial spatial-temporal net-
work: a transferable framework for short-term traffic
forecasting across cities. In Proceedings of the 31st ACM
International Conference on Information & Knowledge
Management, pp. 1905–1915, 2022.

Taylor, S. J. and Letham, B. Forecasting at scale. The
American Statistician, 72(1):37–45, 2018.

Tian Zhou, Peisong Niu, X. W. L. S. and Jin, R. One Fits
All: Power general time series analysis by pretrained lm.
In NeurIPS, 2023.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,
M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E.,
Azhar, F., et al. Llama: Open and efficient foundation lan-
guage models. arXiv preprint arXiv:2302.13971, 2023a.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,
M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E.,
Azhar, F., et al. Llama: Open and efficient foundation lan-
guage models. arXiv preprint arXiv:2302.13971, 2023b.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi,
A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P.,
Bhosale, S., et al. Llama 2: Open foundation and fine-
tuned chat models. arXiv preprint arXiv:2307.09288,
2023c.

Woo, G., Liu, C., Sahoo, D., Kumar, A., and Hoi, S. Ets-
former: Exponential smoothing transformers for time-
series forecasting.
arXiv preprint arXiv:2202.01381,
2022.

Wu, H., Xu, J., Wang, J., and Long, M. Autoformer: Decom-
position transformers with auto-correlation for long-term
series forecasting. Advances in Neural Information Pro-
cessing Systems, 34:22419–22430, 2021.

Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., and Long,
M. TimesNet: Temporal 2d-variation modeling for gen-
eral time series analysis. In International Conference on
Learning Representations, 2023.

Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., and Zhang, C.
Connecting the dots: Multivariate time series forecasting
with graph neural networks. In Proceedings of the 26th
ACM SIGKDD international conference on knowledge
discovery & data mining, pp. 753–763, 2020.

Xue, H. and Salim, F. D. Promptcast: A new prompt-based
learning paradigm for time series forecasting. 2022.

Yin, S., Fu, C., Zhao, S., Li, K., Sun, X., Xu, T., and Chen,
E. A survey on multimodal large language models. arXiv
preprint arXiv:2306.13549, 2023.

Zeng, A., Chen, M., Zhang, L., and Xu, Q. Are transformers
effective for time series forecasting? In Proceedings of
the AAAI conference on artificial intelligence, volume 37,
pp. 11121–11128, 2023.

Zhang, T., Zhang, Y., Cao, W., Bian, J., Yi, X., Zheng, S.,
and Li, J. Less is more: Fast multivariate time series
forecasting with light sampling-oriented mlp structures.
arXiv preprint arXiv:2207.01186, 2022a.

Zhang, X., Zhao, Z., Tsiligkaridis, T., and Zitnik, M. Self-
supervised contrastive pre-training for time series via
time-frequency consistency. Advances in Neural Infor-
mation Processing Systems, 35:3988–4003, 2022b.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H.,
and Zhang, W. Informer: Beyond efficient transformer for
long sequence time-series forecasting. In Proceedings of
the AAAI conference on artificial intelligence, volume 35,
pp. 11106–11115, 2021.

Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., and Jin,
R. Fedformer: Frequency enhanced decomposed trans-
former for long-term series forecasting. In International
Conference on Machine Learning, pp. 27268–27286.
PMLR, 2022.
</pre>

---

## Appendix A. Experimental Details

### A.1. Implementation

We mainly follow the experimental configurations in (Wu et al., 2023) across all baselines within a unified evaluation pipeline, available at `https://github.com/thuml/Time-Series-Library`, for a fair comparison. We use GPT2-small (Radford et al., 2019) with the first 6 hidden layers enabled as the default backbone model. All our experiments are repeated three times and we report the averaged results. We implemented the model on PyTorch (Paszke et al., 2019) with all experiments conducted on NVIDIA RTX A6000 GPUs and NVIDIA A100 GPUs. We configure the patch length $P$ as 16 with a stride $S$ of 8. Our experimental results were obtained by performing a grid search over combinations within the following search spaces: long-term trend length `(24, 48, 96, 144)`, seasonal trend length `(2, 4, 8, 12, 24)`, prompt length `(2, 4, 8, 16, 32)`, and semantic space size `(1000, 2000, 5000)`. Our code is available at `https://github.com/panzijie825/S2IP-LLM`.

### A.2. Baseline Introduction

We introduce the baseline models that we choose to compare in the following section:

- **Time-LLM** (Jin et al., 2024): Time-LLM reprograms time series tokens with NLP representation using multi-head attention and fine-tunes the pre-trained LLM with the prefix prompting to perform time series analysis. We reproduced and reported the experimental results for long-term forecasting using two LLM backbones: `L` represents LLaMA (Touvron et al., 2023a), and `G` represents GPT-2 (Radford et al., 2019).
- **OFA** (Tian Zhou & Jin, 2023): OFA represents time series data into patched tokens to fine-tune the pre-trained GPT2 (Radford et al., 2019) for various time series analysis tasks.
- **iTransformer** (Liu et al., 2023b): iTransformer applies the attention and feed-forward network on the inverted dimensions of the time series data to capture multivariate correlations.
- **DLinear** (Zeng et al., 2023): DLinear incorporates the decomposition with linear layer to model the time series data via modeling trend and seasonal components separately.
- **PatchTST** (Nie et al., 2023): PatchTST leverages a Transformer-based model for time series forecasting by segmenting data into patches and using a channel-independent design to efficiently reduce computational costs and boost forecasting performance.
- **TimesNet** (Wu et al., 2023): TimesNet converts 1D time series data into 2D representation and capture intra- and inter-period relations. It designs TimesBlock with an inception block to extract complex temporal patterns, leading to multiple time series tasks.
- **FEDformer** (Zhou et al., 2022): FEDformer incorporates seasonal-trend decomposition with Transformers for time series forecasting. It leverages information from the frequency domain, gaining efficiency and accuracy in time series analysis.
- **Autoformer** (Wu et al., 2021): Autoformer proposes the decomposition architecture with Auto-Correlation mechanisms to efficiently and accurately perform long-term forecasting.
- **Stationary** (Liu et al., 2022): Non-stationary Transformers proposes a framework with two interdependent modules, namely series stationarization and de-stationary attention to gain robust time series forecasting results.
- **ETSformer** (Woo et al., 2022): ETSformer integrates exponential smoothing principles by replacing traditional self-attention with exponential smoothing attention and frequency attention for time series forecasting.
- **LightTS** (Zhang et al., 2022a): LightTS is a time series classification framework that includes adaptive ensemble distillation and Pareto optimization, resulting in accurate classification with limited resources.

We note that patching-based methods, i.e. OFA (Tian Zhou & Jin, 2023), PatchTST (Nie et al., 2023), and Time-LLM (Jin et al., 2024) treat multivariate time series as independently univariate time series, which essentially provide more training data for those models. For transformer-based models which rely on multivariate times input, this could be the reason that their performances are not as good as patching-based ones.

### A.3. Details of Datasets

We experiment the long-term forecasting on the widely adopted Electricity Transformer Temperature (ETT) datasets (Zhou et al., 2021), Weather, Electricity, and Traffic from (Wu et al., 2023). We also experiment the short-term forecasting using the M4 benchmark dataset (Makridakis et al., 2018).

ETT datasets are comprised of roughly two years of data from two locations in China. The data are further divided into four distinct datasets, each with different sampling rates: ETTh1 and ETTh2 are sampled hourly, and ETTm1 and ETTm2 are sampled every 15 minutes. Every ETT dataset includes six power load features and a target variable: the oil temperature. The Electricity dataset comprises records of electricity consumption from 321 customers and is measured with a 1-hour sampling rate. The Weather dataset contains one-year records from 21 meteorological stations located in Germany. The sampling rate for the Weather dataset is 10 minutes. The Traffic dataset includes the per-hour sampled occupancy rates of the freeway system, which were recorded from 862 sensors in California. The M4 benchmark dataset has 100 thousand time series, which were collected from various domains ranging from business to economic forecasting. The time series data are partitioned into six groups with varied sampling rates from yearly to hourly.

The full data statistics are summarized in Table 5.

| Task | Dataset | Dim. | Series Length | Dataset Size | Frequency | Information |
| --- | --- | ---: | --- | --- | --- | --- |
| Long-term Forecasting | ETTm1 | 7 | {96, 192, 336, 720} | (34465, 11521, 11521) | 15 min | Temperature |
| Long-term Forecasting | ETTm2 | 7 | {96, 192, 336, 720} | (34465, 11521, 11521) | 15 min | Temperature |
| Long-term Forecasting | ETTh1 | 7 | {96, 192, 336, 720} | (8545, 2881, 2881) | 1 hour | Temperature |
| Long-term Forecasting | ETTh2 | 7 | {96, 192, 336, 720} | (8545, 2881, 2881) | 1 hour | Temperature |
| Long-term Forecasting | Electricity | 321 | {96, 192, 336, 720} | (18317, 2633, 5261) | 1 hour | Electricity |
| Long-term Forecasting | Traffic | 862 | {96, 192, 336, 720} | (12185, 1757, 3509) | 1 hour | Transportation |
| Long-term Forecasting | Weather | 21 | {96, 192, 336, 720} | (36792, 5271, 10540) | 10 min | Weather |
| Short-term Forecasting | M4-Yearly | 1 | 6 | (23000, 0, 23000) | Yearly | Demographic |
| Short-term Forecasting | M4-Quarterly | 1 | 8 | (24000, 0, 24000) | Quarterly | Finance |
| Short-term Forecasting | M4-Monthly | 1 | 18 | (48000, 0, 48000) | Monthly | Industry |
| Short-term Forecasting | M4-Weekly | 1 | 13 | (359, 0, 359) | Weekly | Macro |
| Short-term Forecasting | M4-Daily | 1 | 14 | (4227, 0, 4227) | Daily | Micro |
| Short-term Forecasting | M4-Hourly | 1 | 48 | (414, 0, 414) | Hourly | Other |

> Table 5 (see PDF p. 14). Dataset statistics are from (Wu et al., 2023). The dimension indicates the number of time series variables, and the dataset size is organized in (training, validation, and testing).

### A.4. Evaluation Metrics

For evaluation metrics, we use the mean square error (MSE) and mean absolute error (MAE) for long-term forecasting. For short-term forecasting on the M4 benchmark, we use the symmetric mean absolute percentage error (SMAPE), mean absolute scaled error (MASE), and overall weighted average (OWA) (Oreshkin et al., 2019), which is a specific metric for the M4 competition. We present the calculations of these metrics as follows:

$$
\mathrm{MSE} = \frac{1}{H}\sum_{h=1}^{H}(Y_h - \hat{Y}_h)^2
$$

$$
\mathrm{MAE} = \frac{1}{H}\sum_{h=1}^{H}|Y_h - \hat{Y}_h|
$$

$$
\mathrm{SMAPE} = \frac{200}{H}\sum_{h=1}^{H}\frac{|Y_h - \hat{Y}_h|}{|Y_h| + |\hat{Y}_h|}
$$

$$
\mathrm{MAPE} = \frac{100}{H}\sum_{h=1}^{H}\frac{|Y_h - \hat{Y}_h|}{|Y_h|}
$$

$$
\mathrm{MASE} = \frac{1}{H}\sum_{h=1}^{H}
\frac{|Y_h - \hat{Y}_h|}{\frac{1}{H-s}\sum_{j=s+1}^{H}|Y_j - Y_{j-s}|}
$$

$$
\mathrm{OWA} = \frac{1}{2}\left[
\frac{\mathrm{SMAPE}}{\mathrm{SMAPE}_{\mathrm{Naive2}}}
+
\frac{\mathrm{MASE}}{\mathrm{MASE}_{\mathrm{Naive2}}}
\right]
$$

where $s$ is the time series data periodicity. $H$ denotes the prediction intervals. $Y_h$ and $\hat{Y}_h$ are the $h$-th ground truth and prediction where $h \in \{1, \ldots, H\}$. For the evaluation metrics in long-term forecasting, we clarify that the reported metrics are the normalized versions of MAE/MSE. Although we apply global standardization to the data, the information that the scaler used is from training data solely.

## Appendix B. Long-term Forecasting Results

Table 6 shows the detailed results of all prediction lengths of five Transformer-based forecasting models. S²IP-LLM shows a strong and relatively stable performance across different datasets compared to other transformer-based models.

We note that patching-based methods, i.e. OFA (Tian Zhou & Jin, 2023), PatchTST (Nie et al., 2023) treat multivariate time series independently as univariate time series, which essentially provide more training data for univariate time series input based models. This potentially contributes to their advantages in terms of performance. Thus, it may create an unfair comparison to methods with truly multivariate inputs, i.e. the other transformer-based models.

**Table 6.** Transformer-based Models long-term forecasting results for `{96, 192, 336, 720}` horizons. A lower value indicates a better performance. Values are reported as `MSE / MAE`.

<table>
  <thead>
    <tr>
      <th colspan="2">Methods</th>
      <th colspan="2">S²IP-LLM</th>
      <th colspan="2">TimesNet</th>
      <th colspan="2">FEDformer</th>
      <th colspan="2">Autoformer</th>
      <th colspan="2">Stationary</th>
      <th colspan="2">ETSformer</th>
      <th colspan="2">LightTS</th>
    </tr>
    <tr>
      <th>Datasets</th>
      <th>Horizon</th>
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
      <td rowspan="5">Weather</td>
      <td>96</td>
      <td>0.145</td>
      <td>0.195</td>
      <td>0.172</td>
      <td>0.220</td>
      <td>0.217</td>
      <td>0.296</td>
      <td>0.266</td>
      <td>0.336</td>
      <td>0.173</td>
      <td>0.223</td>
      <td>0.197</td>
      <td>0.281</td>
      <td>0.182</td>
      <td>0.242</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.190</td>
      <td>0.235</td>
      <td>0.219</td>
      <td>0.261</td>
      <td>0.276</td>
      <td>0.336</td>
      <td>0.307</td>
      <td>0.367</td>
      <td>0.245</td>
      <td>0.285</td>
      <td>0.237</td>
      <td>0.312</td>
      <td>0.227</td>
      <td>0.287</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.243</td>
      <td>0.280</td>
      <td>0.280</td>
      <td>0.306</td>
      <td>0.339</td>
      <td>0.380</td>
      <td>0.359</td>
      <td>0.395</td>
      <td>0.321</td>
      <td>0.338</td>
      <td>0.298</td>
      <td>0.353</td>
      <td>0.282</td>
      <td>0.334</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.312</td>
      <td>0.326</td>
      <td>0.365</td>
      <td>0.359</td>
      <td>0.403</td>
      <td>0.428</td>
      <td>0.419</td>
      <td>0.428</td>
      <td>0.414</td>
      <td>0.410</td>
      <td>0.352</td>
      <td>0.288</td>
      <td>0.352</td>
      <td>0.386</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.222</td>
      <td>0.259</td>
      <td>0.259</td>
      <td>0.287</td>
      <td>0.309</td>
      <td>0.360</td>
      <td>0.338</td>
      <td>0.382</td>
      <td>0.288</td>
      <td>0.314</td>
      <td>0.271</td>
      <td>0.334</td>
      <td>0.261</td>
      <td>0.312</td>
    </tr>
    <tr>
      <td rowspan="5">Electricity</td>
      <td>96</td>
      <td>0.135</td>
      <td>0.230</td>
      <td>0.168</td>
      <td>0.272</td>
      <td>0.193</td>
      <td>0.308</td>
      <td>0.201</td>
      <td>0.317</td>
      <td>0.169</td>
      <td>0.273</td>
      <td>0.187</td>
      <td>0.304</td>
      <td>0.207</td>
      <td>0.307</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.149</td>
      <td>0.247</td>
      <td>0.184</td>
      <td>0.289</td>
      <td>0.201</td>
      <td>0.315</td>
      <td>0.222</td>
      <td>0.334</td>
      <td>0.182</td>
      <td>0.286</td>
      <td>0.199</td>
      <td>0.315</td>
      <td>0.213</td>
      <td>0.316</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.167</td>
      <td>0.266</td>
      <td>0.198</td>
      <td>0.300</td>
      <td>0.214</td>
      <td>0.329</td>
      <td>0.231</td>
      <td>0.338</td>
      <td>0.200</td>
      <td>0.304</td>
      <td>0.212</td>
      <td>0.329</td>
      <td>0.230</td>
      <td>0.333</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.200</td>
      <td>0.287</td>
      <td>0.220</td>
      <td>0.320</td>
      <td>0.246</td>
      <td>0.355</td>
      <td>0.254</td>
      <td>0.361</td>
      <td>0.222</td>
      <td>0.321</td>
      <td>0.233</td>
      <td>0.345</td>
      <td>0.265</td>
      <td>0.360</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.161</td>
      <td>0.257</td>
      <td>0.192</td>
      <td>0.295</td>
      <td>0.214</td>
      <td>0.327</td>
      <td>0.227</td>
      <td>0.338</td>
      <td>0.193</td>
      <td>0.296</td>
      <td>0.208</td>
      <td>0.323</td>
      <td>0.229</td>
      <td>0.329</td>
    </tr>
    <tr>
      <td rowspan="5">Traffic</td>
      <td>96</td>
      <td>0.379</td>
      <td>0.274</td>
      <td>0.593</td>
      <td>0.321</td>
      <td>0.587</td>
      <td>0.366</td>
      <td>0.613</td>
      <td>0.388</td>
      <td>0.612</td>
      <td>0.338</td>
      <td>0.607</td>
      <td>0.392</td>
      <td>0.615</td>
      <td>0.391</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.397</td>
      <td>0.282</td>
      <td>0.617</td>
      <td>0.336</td>
      <td>0.604</td>
      <td>0.373</td>
      <td>0.616</td>
      <td>0.382</td>
      <td>0.613</td>
      <td>0.340</td>
      <td>0.621</td>
      <td>0.399</td>
      <td>0.601</td>
      <td>0.382</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.407</td>
      <td>0.289</td>
      <td>0.629</td>
      <td>0.336</td>
      <td>0.621</td>
      <td>0.383</td>
      <td>0.622</td>
      <td>0.337</td>
      <td>0.618</td>
      <td>0.328</td>
      <td>0.622</td>
      <td>0.396</td>
      <td>0.613</td>
      <td>0.386</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.440</td>
      <td>0.301</td>
      <td>0.640</td>
      <td>0.350</td>
      <td>0.626</td>
      <td>0.382</td>
      <td>0.660</td>
      <td>0.408</td>
      <td>0.653</td>
      <td>0.355</td>
      <td>0.632</td>
      <td>0.396</td>
      <td>0.658</td>
      <td>0.407</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.405</td>
      <td>0.286</td>
      <td>0.620</td>
      <td>0.336</td>
      <td>0.610</td>
      <td>0.376</td>
      <td>0.628</td>
      <td>0.379</td>
      <td>0.624</td>
      <td>0.340</td>
      <td>0.621</td>
      <td>0.396</td>
      <td>0.622</td>
      <td>0.392</td>
    </tr>
    <tr>
      <td rowspan="5">ETTh1</td>
      <td>96</td>
      <td>0.366</td>
      <td>0.396</td>
      <td>0.468</td>
      <td>0.475</td>
      <td>0.376</td>
      <td>0.419</td>
      <td>0.530</td>
      <td>0.517</td>
      <td>0.513</td>
      <td>0.491</td>
      <td>0.644</td>
      <td>0.589</td>
      <td>0.440</td>
      <td>0.450</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.401</td>
      <td>0.420</td>
      <td>0.484</td>
      <td>0.485</td>
      <td>0.420</td>
      <td>0.448</td>
      <td>0.537</td>
      <td>0.521</td>
      <td>0.534</td>
      <td>0.504</td>
      <td>0.736</td>
      <td>0.648</td>
      <td>0.498</td>
      <td>0.479</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.412</td>
      <td>0.431</td>
      <td>0.536</td>
      <td>0.516</td>
      <td>0.459</td>
      <td>0.465</td>
      <td>0.596</td>
      <td>0.583</td>
      <td>0.588</td>
      <td>0.535</td>
      <td>0.827</td>
      <td>0.707</td>
      <td>0.550</td>
      <td>0.510</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.440</td>
      <td>0.458</td>
      <td>0.593</td>
      <td>0.537</td>
      <td>0.506</td>
      <td>0.507</td>
      <td>0.713</td>
      <td>0.639</td>
      <td>0.643</td>
      <td>0.616</td>
      <td>0.946</td>
      <td>0.766</td>
      <td>0.615</td>
      <td>0.571</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.406</td>
      <td>0.427</td>
      <td>0.520</td>
      <td>0.505</td>
      <td>0.440</td>
      <td>0.460</td>
      <td>0.594</td>
      <td>0.565</td>
      <td>0.570</td>
      <td>0.537</td>
      <td>0.788</td>
      <td>0.677</td>
      <td>0.526</td>
      <td>0.502</td>
    </tr>
    <tr>
      <td rowspan="5">ETTh2</td>
      <td>96</td>
      <td>0.278</td>
      <td>0.340</td>
      <td>0.376</td>
      <td>0.415</td>
      <td>0.358</td>
      <td>0.397</td>
      <td>0.454</td>
      <td>0.490</td>
      <td>0.476</td>
      <td>0.458</td>
      <td>0.340</td>
      <td>0.391</td>
      <td>0.408</td>
      <td>0.445</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.346</td>
      <td>0.385</td>
      <td>0.409</td>
      <td>0.440</td>
      <td>0.429</td>
      <td>0.439</td>
      <td>0.486</td>
      <td>0.517</td>
      <td>0.512</td>
      <td>0.493</td>
      <td>0.430</td>
      <td>0.439</td>
      <td>0.561</td>
      <td>0.526</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.367</td>
      <td>0.406</td>
      <td>0.425</td>
      <td>0.455</td>
      <td>0.496</td>
      <td>0.487</td>
      <td>0.493</td>
      <td>0.533</td>
      <td>0.552</td>
      <td>0.551</td>
      <td>0.485</td>
      <td>0.479</td>
      <td>0.673</td>
      <td>0.580</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.400</td>
      <td>0.436</td>
      <td>0.488</td>
      <td>0.494</td>
      <td>0.463</td>
      <td>0.474</td>
      <td>0.515</td>
      <td>0.543</td>
      <td>0.562</td>
      <td>0.560</td>
      <td>0.500</td>
      <td>0.497</td>
      <td>1.006</td>
      <td>0.721</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.347</td>
      <td>0.391</td>
      <td>0.425</td>
      <td>0.451</td>
      <td>0.437</td>
      <td>0.449</td>
      <td>0.487</td>
      <td>0.520</td>
      <td>0.526</td>
      <td>0.516</td>
      <td>0.439</td>
      <td>0.452</td>
      <td>0.662</td>
      <td>0.568</td>
    </tr>
    <tr>
      <td rowspan="5">ETTm1</td>
      <td>96</td>
      <td>0.288</td>
      <td>0.346</td>
      <td>0.329</td>
      <td>0.377</td>
      <td>0.379</td>
      <td>0.419</td>
      <td>0.568</td>
      <td>0.516</td>
      <td>0.386</td>
      <td>0.398</td>
      <td>0.375</td>
      <td>0.398</td>
      <td>0.383</td>
      <td>0.409</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.323</td>
      <td>0.365</td>
      <td>0.371</td>
      <td>0.401</td>
      <td>0.426</td>
      <td>0.441</td>
      <td>0.573</td>
      <td>0.528</td>
      <td>0.459</td>
      <td>0.444</td>
      <td>0.408</td>
      <td>0.410</td>
      <td>0.421</td>
      <td>0.431</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.359</td>
      <td>0.390</td>
      <td>0.417</td>
      <td>0.428</td>
      <td>0.445</td>
      <td>0.459</td>
      <td>0.587</td>
      <td>0.534</td>
      <td>0.495</td>
      <td>0.464</td>
      <td>0.435</td>
      <td>0.428</td>
      <td>0.454</td>
      <td>0.456</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.403</td>
      <td>0.418</td>
      <td>0.483</td>
      <td>0.464</td>
      <td>0.543</td>
      <td>0.490</td>
      <td>0.589</td>
      <td>0.536</td>
      <td>0.585</td>
      <td>0.516</td>
      <td>0.499</td>
      <td>0.462</td>
      <td>0.549</td>
      <td>0.520</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.343</td>
      <td>0.379</td>
      <td>0.400</td>
      <td>0.417</td>
      <td>0.448</td>
      <td>0.452</td>
      <td>0.579</td>
      <td>0.529</td>
      <td>0.481</td>
      <td>0.456</td>
      <td>0.429</td>
      <td>0.425</td>
      <td>0.452</td>
      <td>0.454</td>
    </tr>
    <tr>
      <td rowspan="5">ETTm2</td>
      <td>96</td>
      <td>0.165</td>
      <td>0.257</td>
      <td>0.201</td>
      <td>0.286</td>
      <td>0.203</td>
      <td>0.287</td>
      <td>0.287</td>
      <td>0.359</td>
      <td>0.192</td>
      <td>0.274</td>
      <td>0.189</td>
      <td>0.280</td>
      <td>0.239</td>
      <td>0.335</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.222</td>
      <td>0.299</td>
      <td>0.260</td>
      <td>0.329</td>
      <td>0.269</td>
      <td>0.328</td>
      <td>0.325</td>
      <td>0.388</td>
      <td>0.280</td>
      <td>0.339</td>
      <td>0.253</td>
      <td>0.319</td>
      <td>0.346</td>
      <td>0.412</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.277</td>
      <td>0.330</td>
      <td>0.331</td>
      <td>0.376</td>
      <td>0.325</td>
      <td>0.366</td>
      <td>0.498</td>
      <td>0.491</td>
      <td>0.334</td>
      <td>0.361</td>
      <td>0.314</td>
      <td>0.357</td>
      <td>0.506</td>
      <td>0.506</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.363</td>
      <td>0.390</td>
      <td>0.428</td>
      <td>0.430</td>
      <td>0.421</td>
      <td>0.415</td>
      <td>0.548</td>
      <td>0.517</td>
      <td>0.417</td>
      <td>0.413</td>
      <td>0.414</td>
      <td>0.413</td>
      <td>0.702</td>
      <td>0.606</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.257</td>
      <td>0.319</td>
      <td>0.305</td>
      <td>0.355</td>
      <td>0.305</td>
      <td>0.349</td>
      <td>0.414</td>
      <td>0.439</td>
      <td>0.306</td>
      <td>0.347</td>
      <td>0.293</td>
      <td>0.342</td>
      <td>0.448</td>
      <td>0.465</td>
    </tr>
  </tbody>
</table>
## Appendix C. Full Short-term Forecasting Results

Table 7 shows the full short-term forecasting experiment results on M4 datasets. S²IP-LLM consistently outperforms the majority of baseline models in most cases. It surpasses the performance of OFA significantly and achieves slightly better forecasting performance than PatchTST, which can be attributed to proposed semantic space informed prompting.

**Table 7.** Detailed short-term time series forecasting results on M4 datasets. The forecasting horizons are in `[6, 48]` and the last three rows are weighted averaged from all datasets under different sampling intervals. A lower value indicates better performance.

<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th rowspan="2">Metric</th>
      <th>S²IP-LLM</th>
      <th>Time-LLM(G)</th>
      <th>OFA</th>
      <th>iTrans.</th>
      <th>DLinear</th>
      <th>PatchTST</th>
      <th>N-HiTS</th>
      <th>N-BEATS</th>
      <th>TimesNet</th>
      <th>FEDformer</th>
      <th>Autoformer</th>
      <th>Stationary</th>
      <th>ETSformer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Year</td>
      <td>SMAPE</td>
      <td>13.413</td>
      <td>13.75</td>
      <td>15.11</td>
      <td>13.652</td>
      <td>16.965</td>
      <td>13.477</td>
      <td>13.422</td>
      <td>13.487</td>
      <td>15.378</td>
      <td>14.021</td>
      <td>13.974</td>
      <td>14.727</td>
      <td>18.009</td>
    </tr>
    <tr>
      <td>MASE</td>
      <td>3.024</td>
      <td>3.055</td>
      <td>3.565</td>
      <td>3.095</td>
      <td>4.283</td>
      <td>3.019</td>
      <td>3.056</td>
      <td>3.036</td>
      <td>3.554</td>
      <td>3.036</td>
      <td>3.134</td>
      <td>3.078</td>
      <td>4.487</td>
    </tr>
    <tr>
      <td>OWA</td>
      <td>0.792</td>
      <td>0.805</td>
      <td>0.911</td>
      <td>0.807</td>
      <td>1.058</td>
      <td>0.792</td>
      <td>0.795</td>
      <td>0.795</td>
      <td>0.918</td>
      <td>0.811</td>
      <td>0.822</td>
      <td>0.807</td>
      <td>1.115</td>
    </tr>
    <tr>
      <td rowspan="3">Quart</td>
      <td>SMAPE</td>
      <td>10.352</td>
      <td>10.671</td>
      <td>10.597</td>
      <td>10.353</td>
      <td>12.145</td>
      <td>10.38</td>
      <td>10.185</td>
      <td>10.564</td>
      <td>10.465</td>
      <td>11.100</td>
      <td>11.338</td>
      <td>10.958</td>
      <td>13.376</td>
    </tr>
    <tr>
      <td>MASE</td>
      <td>1.228</td>
      <td>1.276</td>
      <td>1.253</td>
      <td>1.209</td>
      <td>1.520</td>
      <td>1.233</td>
      <td>1.18</td>
      <td>1.252</td>
      <td>1.227</td>
      <td>1.35</td>
      <td>1.365</td>
      <td>1.325</td>
      <td>1.906</td>
    </tr>
    <tr>
      <td>OWA</td>
      <td>0.922</td>
      <td>0.95</td>
      <td>0.938</td>
      <td>0.911</td>
      <td>1.106</td>
      <td>0.921</td>
      <td>0.893</td>
      <td>0.936</td>
      <td>0.923</td>
      <td>0.996</td>
      <td>1.012</td>
      <td>0.981</td>
      <td>1.302</td>
    </tr>
    <tr>
      <td rowspan="3">Month</td>
      <td>SMAPE</td>
      <td>12.995</td>
      <td>13.416</td>
      <td>13.258</td>
      <td>13.079</td>
      <td>13.514</td>
      <td>12.959</td>
      <td>13.059</td>
      <td>13.089</td>
      <td>13.513</td>
      <td>14.403</td>
      <td>13.958</td>
      <td>13.917</td>
      <td>14.588</td>
    </tr>
    <tr>
      <td>MASE</td>
      <td>0.97</td>
      <td>1.045</td>
      <td>1.003</td>
      <td>0.974</td>
      <td>1.037</td>
      <td>0.970</td>
      <td>1.013</td>
      <td>0.996</td>
      <td>1.039</td>
      <td>1.147</td>
      <td>1.103</td>
      <td>1.097</td>
      <td>1.368</td>
    </tr>
    <tr>
      <td>OWA</td>
      <td>0.91</td>
      <td>0.957</td>
      <td>0.931</td>
      <td>0.911</td>
      <td>0.956</td>
      <td>0.905</td>
      <td>0.929</td>
      <td>0.922</td>
      <td>0.957</td>
      <td>1.038</td>
      <td>1.002</td>
      <td>0.998</td>
      <td>1.149</td>
    </tr>
    <tr>
      <td rowspan="3">Others</td>
      <td>SMAPE</td>
      <td>4.805</td>
      <td>4.973</td>
      <td>6.124</td>
      <td>4.78</td>
      <td>6.709</td>
      <td>4.952</td>
      <td>4.711</td>
      <td>6.599</td>
      <td>6.913</td>
      <td>7.148</td>
      <td>5.485</td>
      <td>6.302</td>
      <td>7.267</td>
    </tr>
    <tr>
      <td>MASE</td>
      <td>3.247</td>
      <td>3.412</td>
      <td>4.116</td>
      <td>3.231</td>
      <td>4.953</td>
      <td>3.347</td>
      <td>3.054</td>
      <td>4.430</td>
      <td>4.507</td>
      <td>4.064</td>
      <td>3.865</td>
      <td>4.064</td>
      <td>5.240</td>
    </tr>
    <tr>
      <td>OWA</td>
      <td>1.017</td>
      <td>1.053</td>
      <td>1.259</td>
      <td>1.012</td>
      <td>1.487</td>
      <td>1.049</td>
      <td>0.977</td>
      <td>1.393</td>
      <td>1.438</td>
      <td>1.304</td>
      <td>1.187</td>
      <td>1.304</td>
      <td>1.591</td>
    </tr>
    <tr>
      <td rowspan="3">Avg</td>
      <td>SMAPE</td>
      <td>12.021</td>
      <td>12.494</td>
      <td>12.690</td>
      <td>12.142</td>
      <td>13.639</td>
      <td>12.059</td>
      <td>12.035</td>
      <td>12.250</td>
      <td>12.880</td>
      <td>13.160</td>
      <td>12.909</td>
      <td>12.780</td>
      <td>14.718</td>
    </tr>
    <tr>
      <td>MASE</td>
      <td>1.612</td>
      <td>1.731</td>
      <td>1.808</td>
      <td>1.631</td>
      <td>2.095</td>
      <td>1.623</td>
      <td>1.625</td>
      <td>1.698</td>
      <td>1.836</td>
      <td>1.775</td>
      <td>1.771</td>
      <td>1.756</td>
      <td>2.408</td>
    </tr>
    <tr>
      <td>OWA</td>
      <td>0.857</td>
      <td>0.913</td>
      <td>0.94</td>
      <td>0.874</td>
      <td>1.051</td>
      <td>0.869</td>
      <td>0.869</td>
      <td>0.896</td>
      <td>0.955</td>
      <td>0.949</td>
      <td>0.939</td>
      <td>0.930</td>
      <td>1.172</td>
    </tr>
  </tbody>
</table>
## Appendix D. Full Few-shot Forecasting Results

Table 8 and Table 9 show the full few-short forecasting experiment results with 10% and 5% of the training data respectively.

**Table 8.** Detailed few-shot learning results on 10% training data. Values are reported as `MSE / MAE`.

<table>
  <thead>
    <tr>
      <th colspan="2">Methods</th>
      <th colspan="2">S²IP-LLM</th>
      <th colspan="2">Time-LLM(G)</th>
      <th colspan="2">OFA</th>
      <th colspan="2">iTransformer</th>
      <th colspan="2">DLinear</th>
      <th colspan="2">PatchTST</th>
      <th colspan="2">TimesNet</th>
      <th colspan="2">FEDformer</th>
      <th colspan="2">Autoformer</th>
      <th colspan="2">Stationary</th>
      <th colspan="2">ETSformer</th>
      <th colspan="2">LightTS</th>
    </tr>
    <tr>
      <th>Datasets</th>
      <th>Horizon</th>
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
      <td rowspan="5">Weather</td>
      <td>96</td>
      <td>0.159</td>
      <td>0.210</td>
      <td>0.160</td>
      <td>0.213</td>
      <td>0.163</td>
      <td>0.215</td>
      <td>0.253</td>
      <td>0.307</td>
      <td>0.171</td>
      <td>0.224</td>
      <td>0.165</td>
      <td>0.215</td>
      <td>0.184</td>
      <td>0.230</td>
      <td>0.188</td>
      <td>0.253</td>
      <td>0.221</td>
      <td>0.297</td>
      <td>0.192</td>
      <td>0.234</td>
      <td>0.199</td>
      <td>0.272</td>
      <td>0.217</td>
      <td>0.269</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.200</td>
      <td>0.251</td>
      <td>0.204</td>
      <td>0.254</td>
      <td>0.210</td>
      <td>0.254</td>
      <td>0.292</td>
      <td>0.328</td>
      <td>0.215</td>
      <td>0.263</td>
      <td>0.210</td>
      <td>0.257</td>
      <td>0.245</td>
      <td>0.283</td>
      <td>0.250</td>
      <td>0.304</td>
      <td>0.270</td>
      <td>0.322</td>
      <td>0.269</td>
      <td>0.295</td>
      <td>0.279</td>
      <td>0.332</td>
      <td>0.259</td>
      <td>0.304</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.257</td>
      <td>0.293</td>
      <td>0.255</td>
      <td>0.291</td>
      <td>0.256</td>
      <td>0.292</td>
      <td>0.322</td>
      <td>0.346</td>
      <td>0.258</td>
      <td>0.299</td>
      <td>0.259</td>
      <td>0.297</td>
      <td>0.305</td>
      <td>0.321</td>
      <td>0.312</td>
      <td>0.346</td>
      <td>0.320</td>
      <td>0.351</td>
      <td>0.370</td>
      <td>0.357</td>
      <td>0.356</td>
      <td>0.386</td>
      <td>0.303</td>
      <td>0.334</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.317</td>
      <td>0.335</td>
      <td>0.329</td>
      <td>0.345</td>
      <td>0.321</td>
      <td>0.339</td>
      <td>0.365</td>
      <td>0.374</td>
      <td>0.320</td>
      <td>0.346</td>
      <td>0.332</td>
      <td>0.346</td>
      <td>0.381</td>
      <td>0.371</td>
      <td>0.387</td>
      <td>0.393</td>
      <td>0.390</td>
      <td>0.396</td>
      <td>0.441</td>
      <td>0.405</td>
      <td>0.437</td>
      <td>0.448</td>
      <td>0.377</td>
      <td>0.382</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.233</td>
      <td>0.272</td>
      <td>0.237</td>
      <td>0.275</td>
      <td>0.238</td>
      <td>0.275</td>
      <td>0.308</td>
      <td>0.338</td>
      <td>0.241</td>
      <td>0.283</td>
      <td>0.242</td>
      <td>0.279</td>
      <td>0.279</td>
      <td>0.301</td>
      <td>0.284</td>
      <td>0.324</td>
      <td>0.300</td>
      <td>0.342</td>
      <td>0.318</td>
      <td>0.323</td>
      <td>0.318</td>
      <td>0.360</td>
      <td>0.289</td>
      <td>0.322</td>
    </tr>
    <tr>
      <td rowspan="5">Electricity</td>
      <td>96</td>
      <td>0.143</td>
      <td>0.243</td>
      <td>0.137</td>
      <td>0.240</td>
      <td>0.139</td>
      <td>0.237</td>
      <td>0.154</td>
      <td>0.257</td>
      <td>0.150</td>
      <td>0.253</td>
      <td>0.140</td>
      <td>0.238</td>
      <td>0.299</td>
      <td>0.373</td>
      <td>0.231</td>
      <td>0.323</td>
      <td>0.261</td>
      <td>0.348</td>
      <td>0.420</td>
      <td>0.466</td>
      <td>0.599</td>
      <td>0.587</td>
      <td>0.350</td>
      <td>0.425</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.159</td>
      <td>0.258</td>
      <td>0.159</td>
      <td>0.258</td>
      <td>0.156</td>
      <td>0.252</td>
      <td>0.171</td>
      <td>0.272</td>
      <td>0.164</td>
      <td>0.264</td>
      <td>0.160</td>
      <td>0.255</td>
      <td>0.305</td>
      <td>0.379</td>
      <td>0.261</td>
      <td>0.356</td>
      <td>0.338</td>
      <td>0.406</td>
      <td>0.411</td>
      <td>0.459</td>
      <td>0.620</td>
      <td>0.598</td>
      <td>0.376</td>
      <td>0.448</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.170</td>
      <td>0.269</td>
      <td>0.181</td>
      <td>0.278</td>
      <td>0.175</td>
      <td>0.270</td>
      <td>0.196</td>
      <td>0.295</td>
      <td>0.181</td>
      <td>0.282</td>
      <td>0.180</td>
      <td>0.276</td>
      <td>0.319</td>
      <td>0.391</td>
      <td>0.360</td>
      <td>0.445</td>
      <td>0.410</td>
      <td>0.474</td>
      <td>0.434</td>
      <td>0.473</td>
      <td>0.662</td>
      <td>0.619</td>
      <td>0.428</td>
      <td>0.485</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.230</td>
      <td>0.315</td>
      <td>0.232</td>
      <td>0.317</td>
      <td>0.233</td>
      <td>0.317</td>
      <td>0.263</td>
      <td>0.348</td>
      <td>0.223</td>
      <td>0.321</td>
      <td>0.241</td>
      <td>0.323</td>
      <td>0.369</td>
      <td>0.426</td>
      <td>0.530</td>
      <td>0.585</td>
      <td>0.715</td>
      <td>0.685</td>
      <td>0.510</td>
      <td>0.521</td>
      <td>0.757</td>
      <td>0.664</td>
      <td>0.611</td>
      <td>0.597</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.175</td>
      <td>0.271</td>
      <td>0.177</td>
      <td>0.273</td>
      <td>0.176</td>
      <td>0.269</td>
      <td>0.196</td>
      <td>0.293</td>
      <td>0.180</td>
      <td>0.280</td>
      <td>0.180</td>
      <td>0.273</td>
      <td>0.323</td>
      <td>0.392</td>
      <td>0.346</td>
      <td>0.427</td>
      <td>0.431</td>
      <td>0.478</td>
      <td>0.444</td>
      <td>0.480</td>
      <td>0.660</td>
      <td>0.617</td>
      <td>0.441</td>
      <td>0.489</td>
    </tr>
    <tr>
      <td rowspan="5">Traffic</td>
      <td>96</td>
      <td>0.403</td>
      <td>0.293</td>
      <td>0.406</td>
      <td>0.295</td>
      <td>0.414</td>
      <td>0.297</td>
      <td>0.448</td>
      <td>0.329</td>
      <td>0.419</td>
      <td>0.298</td>
      <td>0.403</td>
      <td>0.289</td>
      <td>0.719</td>
      <td>0.416</td>
      <td>0.639</td>
      <td>0.400</td>
      <td>0.672</td>
      <td>0.405</td>
      <td>1.412</td>
      <td>0.802</td>
      <td>1.643</td>
      <td>0.855</td>
      <td>1.157</td>
      <td>0.636</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.412</td>
      <td>0.295</td>
      <td>0.416</td>
      <td>0.300</td>
      <td>0.426</td>
      <td>0.301</td>
      <td>0.487</td>
      <td>0.360</td>
      <td>0.434</td>
      <td>0.305</td>
      <td>0.415</td>
      <td>0.296</td>
      <td>0.748</td>
      <td>0.428</td>
      <td>0.637</td>
      <td>0.416</td>
      <td>0.727</td>
      <td>0.424</td>
      <td>1.419</td>
      <td>0.806</td>
      <td>1.641</td>
      <td>0.854</td>
      <td>1.207</td>
      <td>0.661</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.427</td>
      <td>0.316</td>
      <td>0.430</td>
      <td>0.309</td>
      <td>0.434</td>
      <td>0.303</td>
      <td>0.514</td>
      <td>0.372</td>
      <td>0.449</td>
      <td>0.313</td>
      <td>0.426</td>
      <td>0.304</td>
      <td>0.853</td>
      <td>0.471</td>
      <td>0.655</td>
      <td>0.427</td>
      <td>0.749</td>
      <td>0.454</td>
      <td>1.443</td>
      <td>0.815</td>
      <td>1.711</td>
      <td>0.878</td>
      <td>1.334</td>
      <td>0.713</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.469</td>
      <td>0.325</td>
      <td>0.467</td>
      <td>0.324</td>
      <td>0.487</td>
      <td>0.337</td>
      <td>0.532</td>
      <td>0.383</td>
      <td>0.484</td>
      <td>0.336</td>
      <td>0.474</td>
      <td>0.331</td>
      <td>1.485</td>
      <td>0.825</td>
      <td>0.722</td>
      <td>0.456</td>
      <td>0.847</td>
      <td>0.499</td>
      <td>1.539</td>
      <td>0.837</td>
      <td>2.660</td>
      <td>1.157</td>
      <td>1.292</td>
      <td>0.726</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.427</td>
      <td>0.307</td>
      <td>0.429</td>
      <td>0.307</td>
      <td>0.440</td>
      <td>0.310</td>
      <td>0.495</td>
      <td>0.361</td>
      <td>0.447</td>
      <td>0.313</td>
      <td>0.430</td>
      <td>0.305</td>
      <td>0.951</td>
      <td>0.535</td>
      <td>0.663</td>
      <td>0.425</td>
      <td>0.749</td>
      <td>0.446</td>
      <td>1.453</td>
      <td>0.815</td>
      <td>1.914</td>
      <td>0.936</td>
      <td>1.248</td>
      <td>0.684</td>
    </tr>
    <tr>
      <td rowspan="5">ETTh1</td>
      <td>96</td>
      <td>0.481</td>
      <td>0.474</td>
      <td>0.720</td>
      <td>0.533</td>
      <td>0.458</td>
      <td>0.456</td>
      <td>0.790</td>
      <td>0.586</td>
      <td>0.492</td>
      <td>0.495</td>
      <td>0.516</td>
      <td>0.485</td>
      <td>0.861</td>
      <td>0.628</td>
      <td>0.512</td>
      <td>0.499</td>
      <td>0.613</td>
      <td>0.552</td>
      <td>0.918</td>
      <td>0.639</td>
      <td>1.112</td>
      <td>0.806</td>
      <td>1.298</td>
      <td>0.838</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.518</td>
      <td>0.491</td>
      <td>0.747</td>
      <td>0.545</td>
      <td>0.570</td>
      <td>0.516</td>
      <td>0.837</td>
      <td>0.609</td>
      <td>0.565</td>
      <td>0.538</td>
      <td>0.598</td>
      <td>0.524</td>
      <td>0.797</td>
      <td>0.593</td>
      <td>0.624</td>
      <td>0.555</td>
      <td>0.722</td>
      <td>0.598</td>
      <td>0.915</td>
      <td>0.629</td>
      <td>1.155</td>
      <td>0.823</td>
      <td>1.322</td>
      <td>0.854</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.664</td>
      <td>0.570</td>
      <td>0.793</td>
      <td>0.551</td>
      <td>0.608</td>
      <td>0.535</td>
      <td>0.780</td>
      <td>0.575</td>
      <td>0.721</td>
      <td>0.622</td>
      <td>0.657</td>
      <td>0.550</td>
      <td>0.941</td>
      <td>0.648</td>
      <td>0.691</td>
      <td>0.574</td>
      <td>0.750</td>
      <td>0.619</td>
      <td>0.939</td>
      <td>0.644</td>
      <td>1.179</td>
      <td>0.832</td>
      <td>1.347</td>
      <td>0.870</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.711</td>
      <td>0.584</td>
      <td>0.880</td>
      <td>0.584</td>
      <td>0.725</td>
      <td>0.591</td>
      <td>1.234</td>
      <td>0.811</td>
      <td>0.986</td>
      <td>0.743</td>
      <td>0.762</td>
      <td>0.610</td>
      <td>0.877</td>
      <td>0.641</td>
      <td>0.728</td>
      <td>0.614</td>
      <td>0.721</td>
      <td>0.616</td>
      <td>0.887</td>
      <td>0.645</td>
      <td>1.273</td>
      <td>0.874</td>
      <td>1.534</td>
      <td>0.947</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.593</td>
      <td>0.529</td>
      <td>0.785</td>
      <td>0.553</td>
      <td>0.590</td>
      <td>0.525</td>
      <td>0.910</td>
      <td>0.860</td>
      <td>0.691</td>
      <td>0.600</td>
      <td>0.633</td>
      <td>0.542</td>
      <td>0.869</td>
      <td>0.628</td>
      <td>0.639</td>
      <td>0.561</td>
      <td>0.702</td>
      <td>0.596</td>
      <td>0.915</td>
      <td>0.639</td>
      <td>1.180</td>
      <td>0.834</td>
      <td>1.375</td>
      <td>0.877</td>
    </tr>
    <tr>
      <td rowspan="5">ETTh2</td>
      <td>96</td>
      <td>0.354</td>
      <td>0.400</td>
      <td>0.334</td>
      <td>0.381</td>
      <td>0.331</td>
      <td>0.374</td>
      <td>0.404</td>
      <td>0.435</td>
      <td>0.357</td>
      <td>0.411</td>
      <td>0.353</td>
      <td>0.389</td>
      <td>0.378</td>
      <td>0.409</td>
      <td>0.382</td>
      <td>0.416</td>
      <td>0.413</td>
      <td>0.451</td>
      <td>0.389</td>
      <td>0.411</td>
      <td>0.678</td>
      <td>0.619</td>
      <td>2.022</td>
      <td>1.006</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.401</td>
      <td>0.423</td>
      <td>0.430</td>
      <td>0.438</td>
      <td>0.402</td>
      <td>0.411</td>
      <td>0.470</td>
      <td>0.474</td>
      <td>0.569</td>
      <td>0.519</td>
      <td>0.403</td>
      <td>0.414</td>
      <td>0.490</td>
      <td>0.467</td>
      <td>0.478</td>
      <td>0.474</td>
      <td>0.474</td>
      <td>0.477</td>
      <td>0.473</td>
      <td>0.455</td>
      <td>0.785</td>
      <td>0.666</td>
      <td>2.329</td>
      <td>1.104</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.442</td>
      <td>0.450</td>
      <td>0.449</td>
      <td>0.458</td>
      <td>0.406</td>
      <td>0.433</td>
      <td>0.489</td>
      <td>0.485</td>
      <td>0.671</td>
      <td>0.572</td>
      <td>0.426</td>
      <td>0.441</td>
      <td>0.537</td>
      <td>0.494</td>
      <td>0.504</td>
      <td>0.501</td>
      <td>0.547</td>
      <td>0.543</td>
      <td>0.477</td>
      <td>0.472</td>
      <td>0.839</td>
      <td>0.694</td>
      <td>2.453</td>
      <td>1.122</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.480</td>
      <td>0.486</td>
      <td>0.485</td>
      <td>0.490</td>
      <td>0.449</td>
      <td>0.464</td>
      <td>0.593</td>
      <td>0.538</td>
      <td>0.824</td>
      <td>0.648</td>
      <td>0.477</td>
      <td>0.480</td>
      <td>0.510</td>
      <td>0.491</td>
      <td>0.499</td>
      <td>0.509</td>
      <td>0.516</td>
      <td>0.523</td>
      <td>0.507</td>
      <td>0.480</td>
      <td>1.273</td>
      <td>0.874</td>
      <td>3.816</td>
      <td>1.407</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.419</td>
      <td>0.439</td>
      <td>0.424</td>
      <td>0.441</td>
      <td>0.397</td>
      <td>0.421</td>
      <td>0.489</td>
      <td>0.483</td>
      <td>0.605</td>
      <td>0.538</td>
      <td>0.415</td>
      <td>0.431</td>
      <td>0.479</td>
      <td>0.465</td>
      <td>0.466</td>
      <td>0.475</td>
      <td>0.488</td>
      <td>0.499</td>
      <td>0.462</td>
      <td>0.455</td>
      <td>0.894</td>
      <td>0.713</td>
      <td>2.655</td>
      <td>1.160</td>
    </tr>
    <tr>
      <td rowspan="5">ETTm1</td>
      <td>96</td>
      <td>0.388</td>
      <td>0.401</td>
      <td>0.412</td>
      <td>0.422</td>
      <td>0.390</td>
      <td>0.404</td>
      <td>0.709</td>
      <td>0.556</td>
      <td>0.352</td>
      <td>0.392</td>
      <td>0.410</td>
      <td>0.419</td>
      <td>0.583</td>
      <td>0.501</td>
      <td>0.578</td>
      <td>0.518</td>
      <td>0.774</td>
      <td>0.614</td>
      <td>0.761</td>
      <td>0.568</td>
      <td>0.911</td>
      <td>0.688</td>
      <td>0.921</td>
      <td>0.682</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.422</td>
      <td>0.421</td>
      <td>0.447</td>
      <td>0.438</td>
      <td>0.429</td>
      <td>0.423</td>
      <td>0.717</td>
      <td>0.548</td>
      <td>0.382</td>
      <td>0.412</td>
      <td>0.437</td>
      <td>0.434</td>
      <td>0.630</td>
      <td>0.528</td>
      <td>0.617</td>
      <td>0.546</td>
      <td>0.754</td>
      <td>0.592</td>
      <td>0.781</td>
      <td>0.574</td>
      <td>0.955</td>
      <td>0.703</td>
      <td>0.957</td>
      <td>0.701</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.456</td>
      <td>0.430</td>
      <td>0.497</td>
      <td>0.465</td>
      <td>0.469</td>
      <td>0.439</td>
      <td>0.735</td>
      <td>0.575</td>
      <td>0.419</td>
      <td>0.434</td>
      <td>0.476</td>
      <td>0.454</td>
      <td>0.725</td>
      <td>0.568</td>
      <td>0.998</td>
      <td>0.775</td>
      <td>0.869</td>
      <td>0.677</td>
      <td>0.803</td>
      <td>0.587</td>
      <td>0.991</td>
      <td>0.719</td>
      <td>0.998</td>
      <td>0.716</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.554</td>
      <td>0.490</td>
      <td>0.594</td>
      <td>0.521</td>
      <td>0.569</td>
      <td>0.498</td>
      <td>0.752</td>
      <td>0.584</td>
      <td>0.490</td>
      <td>0.477</td>
      <td>0.681</td>
      <td>0.556</td>
      <td>0.769</td>
      <td>0.549</td>
      <td>0.693</td>
      <td>0.579</td>
      <td>0.810</td>
      <td>0.630</td>
      <td>0.844</td>
      <td>0.581</td>
      <td>1.062</td>
      <td>0.747</td>
      <td>1.007</td>
      <td>0.719</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.455</td>
      <td>0.435</td>
      <td>0.487</td>
      <td>0.461</td>
      <td>0.464</td>
      <td>0.441</td>
      <td>0.728</td>
      <td>0.565</td>
      <td>0.411</td>
      <td>0.429</td>
      <td>0.501</td>
      <td>0.466</td>
      <td>0.677</td>
      <td>0.537</td>
      <td>0.722</td>
      <td>0.605</td>
      <td>0.802</td>
      <td>0.628</td>
      <td>0.797</td>
      <td>0.578</td>
      <td>0.980</td>
      <td>0.714</td>
      <td>0.971</td>
      <td>0.705</td>
    </tr>
    <tr>
      <td rowspan="5">ETTm2</td>
      <td>96</td>
      <td>0.192</td>
      <td>0.274</td>
      <td>0.224</td>
      <td>0.296</td>
      <td>0.188</td>
      <td>0.269</td>
      <td>0.245</td>
      <td>0.322</td>
      <td>0.213</td>
      <td>0.303</td>
      <td>0.191</td>
      <td>0.274</td>
      <td>0.212</td>
      <td>0.285</td>
      <td>0.291</td>
      <td>0.399</td>
      <td>0.352</td>
      <td>0.454</td>
      <td>0.229</td>
      <td>0.308</td>
      <td>0.331</td>
      <td>0.430</td>
      <td>0.813</td>
      <td>0.688</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.246</td>
      <td>0.313</td>
      <td>0.260</td>
      <td>0.317</td>
      <td>0.251</td>
      <td>0.309</td>
      <td>0.274</td>
      <td>0.338</td>
      <td>0.278</td>
      <td>0.345</td>
      <td>0.252</td>
      <td>0.317</td>
      <td>0.270</td>
      <td>0.323</td>
      <td>0.307</td>
      <td>0.379</td>
      <td>0.694</td>
      <td>0.691</td>
      <td>0.291</td>
      <td>0.343</td>
      <td>0.400</td>
      <td>0.464</td>
      <td>1.008</td>
      <td>0.768</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.301</td>
      <td>0.340</td>
      <td>0.312</td>
      <td>0.349</td>
      <td>0.307</td>
      <td>0.346</td>
      <td>0.361</td>
      <td>0.394</td>
      <td>0.338</td>
      <td>0.385</td>
      <td>0.306</td>
      <td>0.353</td>
      <td>0.323</td>
      <td>0.353</td>
      <td>0.543</td>
      <td>0.559</td>
      <td>2.408</td>
      <td>1.407</td>
      <td>0.348</td>
      <td>0.376</td>
      <td>0.469</td>
      <td>0.498</td>
      <td>1.031</td>
      <td>0.775</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.400</td>
      <td>0.403</td>
      <td>0.424</td>
      <td>0.416</td>
      <td>0.426</td>
      <td>0.417</td>
      <td>0.467</td>
      <td>0.442</td>
      <td>0.436</td>
      <td>0.440</td>
      <td>0.433</td>
      <td>0.427</td>
      <td>0.474</td>
      <td>0.449</td>
      <td>0.712</td>
      <td>0.614</td>
      <td>1.913</td>
      <td>1.166</td>
      <td>0.461</td>
      <td>0.438</td>
      <td>0.589</td>
      <td>0.557</td>
      <td>1.096</td>
      <td>0.791</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.284</td>
      <td>0.332</td>
      <td>0.305</td>
      <td>0.344</td>
      <td>0.293</td>
      <td>0.335</td>
      <td>0.336</td>
      <td>0.373</td>
      <td>0.316</td>
      <td>0.368</td>
      <td>0.296</td>
      <td>0.343</td>
      <td>0.320</td>
      <td>0.353</td>
      <td>0.463</td>
      <td>0.488</td>
      <td>1.342</td>
      <td>0.930</td>
      <td>0.332</td>
      <td>0.366</td>
      <td>0.447</td>
      <td>0.487</td>
      <td>0.987</td>
      <td>0.756</td>
    </tr>
  </tbody>
</table>
**Table 9.** Detailed few-shot learning results on 5% training data. `-` means 5% data is not sufficient to constitute a training set. Values are reported as `MSE / MAE`.

<table>
  <thead>
    <tr>
      <th colspan="2">Methods</th>
      <th colspan="2">S²IP-LLM</th>
      <th colspan="2">Time-LLM(G)</th>
      <th colspan="2">OFA</th>
      <th colspan="2">iTransformer</th>
      <th colspan="2">DLinear</th>
      <th colspan="2">PatchTST</th>
      <th colspan="2">TimesNet</th>
      <th colspan="2">FEDformer</th>
      <th colspan="2">Autoformer</th>
      <th colspan="2">Stationary</th>
      <th colspan="2">ETSformer</th>
      <th colspan="2">LightTS</th>
    </tr>
    <tr>
      <th>Datasets</th>
      <th>Horizon</th>
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
      <td rowspan="5">Weather</td>
      <td>96</td>
      <td>0.175</td>
      <td>0.228</td>
      <td>0.176</td>
      <td>0.230</td>
      <td>0.175</td>
      <td>0.230</td>
      <td>0.264</td>
      <td>0.307</td>
      <td>0.184</td>
      <td>0.242</td>
      <td>0.171</td>
      <td>0.224</td>
      <td>0.207</td>
      <td>0.253</td>
      <td>0.229</td>
      <td>0.309</td>
      <td>0.227</td>
      <td>0.299</td>
      <td>0.215</td>
      <td>0.252</td>
      <td>0.218</td>
      <td>0.295</td>
      <td>0.230</td>
      <td>0.285</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.225</td>
      <td>0.271</td>
      <td>0.226</td>
      <td>0.275</td>
      <td>0.227</td>
      <td>0.276</td>
      <td>0.284</td>
      <td>0.326</td>
      <td>0.228</td>
      <td>0.283</td>
      <td>0.230</td>
      <td>0.277</td>
      <td>0.272</td>
      <td>0.307</td>
      <td>0.265</td>
      <td>0.317</td>
      <td>0.278</td>
      <td>0.333</td>
      <td>0.290</td>
      <td>0.307</td>
      <td>0.294</td>
      <td>0.331</td>
      <td>0.274</td>
      <td>0.323</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.282</td>
      <td>0.321</td>
      <td>0.292</td>
      <td>0.325</td>
      <td>0.286</td>
      <td>0.322</td>
      <td>0.323</td>
      <td>0.349</td>
      <td>0.279</td>
      <td>0.322</td>
      <td>0.294</td>
      <td>0.326</td>
      <td>0.313</td>
      <td>0.328</td>
      <td>0.353</td>
      <td>0.392</td>
      <td>0.351</td>
      <td>0.393</td>
      <td>0.353</td>
      <td>0.348</td>
      <td>0.359</td>
      <td>0.398</td>
      <td>0.318</td>
      <td>0.355</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.361</td>
      <td>0.371</td>
      <td>0.364</td>
      <td>0.375</td>
      <td>0.366</td>
      <td>0.379</td>
      <td>0.366</td>
      <td>0.375</td>
      <td>0.364</td>
      <td>0.388</td>
      <td>0.384</td>
      <td>0.387</td>
      <td>0.400</td>
      <td>0.385</td>
      <td>0.391</td>
      <td>0.394</td>
      <td>0.387</td>
      <td>0.389</td>
      <td>0.452</td>
      <td>0.407</td>
      <td>0.461</td>
      <td>0.461</td>
      <td>0.401</td>
      <td>0.418</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.260</td>
      <td>0.297</td>
      <td>0.264</td>
      <td>0.301</td>
      <td>0.263</td>
      <td>0.301</td>
      <td>0.309</td>
      <td>0.339</td>
      <td>0.263</td>
      <td>0.308</td>
      <td>0.269</td>
      <td>0.303</td>
      <td>0.298</td>
      <td>0.318</td>
      <td>0.309</td>
      <td>0.353</td>
      <td>0.310</td>
      <td>0.353</td>
      <td>0.327</td>
      <td>0.328</td>
      <td>0.333</td>
      <td>0.371</td>
      <td>0.305</td>
      <td>0.345</td>
    </tr>
    <tr>
      <td rowspan="5">Electricity</td>
      <td>96</td>
      <td>0.148</td>
      <td>0.248</td>
      <td>0.148</td>
      <td>0.248</td>
      <td>0.143</td>
      <td>0.241</td>
      <td>0.162</td>
      <td>0.264</td>
      <td>0.150</td>
      <td>0.251</td>
      <td>0.145</td>
      <td>0.244</td>
      <td>0.315</td>
      <td>0.389</td>
      <td>0.235</td>
      <td>0.322</td>
      <td>0.297</td>
      <td>0.367</td>
      <td>0.484</td>
      <td>0.518</td>
      <td>0.697</td>
      <td>0.638</td>
      <td>0.639</td>
      <td>0.609</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.159</td>
      <td>0.255</td>
      <td>0.160</td>
      <td>0.257</td>
      <td>0.159</td>
      <td>0.255</td>
      <td>0.180</td>
      <td>0.278</td>
      <td>0.163</td>
      <td>0.263</td>
      <td>0.163</td>
      <td>0.260</td>
      <td>0.318</td>
      <td>0.396</td>
      <td>0.247</td>
      <td>0.341</td>
      <td>0.308</td>
      <td>0.375</td>
      <td>0.501</td>
      <td>0.531</td>
      <td>0.718</td>
      <td>0.648</td>
      <td>0.772</td>
      <td>0.678</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.175</td>
      <td>0.271</td>
      <td>0.183</td>
      <td>0.282</td>
      <td>0.179</td>
      <td>0.274</td>
      <td>0.207</td>
      <td>0.305</td>
      <td>0.175</td>
      <td>0.278</td>
      <td>0.183</td>
      <td>0.281</td>
      <td>0.340</td>
      <td>0.415</td>
      <td>0.267</td>
      <td>0.356</td>
      <td>0.354</td>
      <td>0.411</td>
      <td>0.574</td>
      <td>0.578</td>
      <td>0.758</td>
      <td>0.667</td>
      <td>0.901</td>
      <td>0.745</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.235</td>
      <td>0.326</td>
      <td>0.236</td>
      <td>0.329</td>
      <td>0.233</td>
      <td>0.323</td>
      <td>0.258</td>
      <td>0.339</td>
      <td>0.219</td>
      <td>0.311</td>
      <td>0.233</td>
      <td>0.323</td>
      <td>0.635</td>
      <td>0.613</td>
      <td>0.318</td>
      <td>0.394</td>
      <td>0.426</td>
      <td>0.466</td>
      <td>0.952</td>
      <td>0.786</td>
      <td>1.028</td>
      <td>0.788</td>
      <td>1.200</td>
      <td>0.871</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.179</td>
      <td>0.275</td>
      <td>0.181</td>
      <td>0.279</td>
      <td>0.178</td>
      <td>0.273</td>
      <td>0.201</td>
      <td>0.296</td>
      <td>0.176</td>
      <td>0.275</td>
      <td>0.181</td>
      <td>0.277</td>
      <td>0.402</td>
      <td>0.453</td>
      <td>0.266</td>
      <td>0.353</td>
      <td>0.346</td>
      <td>0.404</td>
      <td>0.627</td>
      <td>0.603</td>
      <td>0.800</td>
      <td>0.685</td>
      <td>0.878</td>
      <td>0.725</td>
    </tr>
    <tr>
      <td rowspan="5">Traffic</td>
      <td>96</td>
      <td>0.410</td>
      <td>0.288</td>
      <td>0.414</td>
      <td>0.293</td>
      <td>0.419</td>
      <td>0.298</td>
      <td>0.431</td>
      <td>0.312</td>
      <td>0.427</td>
      <td>0.304</td>
      <td>0.404</td>
      <td>0.286</td>
      <td>0.854</td>
      <td>0.492</td>
      <td>0.670</td>
      <td>0.421</td>
      <td>0.795</td>
      <td>0.481</td>
      <td>1.468</td>
      <td>0.821</td>
      <td>1.643</td>
      <td>0.855</td>
      <td>1.157</td>
      <td>0.636</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.416</td>
      <td>0.298</td>
      <td>0.419</td>
      <td>0.300</td>
      <td>0.434</td>
      <td>0.305</td>
      <td>0.456</td>
      <td>0.326</td>
      <td>0.447</td>
      <td>0.315</td>
      <td>0.412</td>
      <td>0.294</td>
      <td>0.894</td>
      <td>0.517</td>
      <td>0.653</td>
      <td>0.405</td>
      <td>0.837</td>
      <td>0.503</td>
      <td>1.509</td>
      <td>0.838</td>
      <td>1.856</td>
      <td>0.928</td>
      <td>1.688</td>
      <td>0.848</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.435</td>
      <td>0.313</td>
      <td>0.438</td>
      <td>0.315</td>
      <td>0.449</td>
      <td>0.313</td>
      <td>0.465</td>
      <td>0.334</td>
      <td>0.478</td>
      <td>0.333</td>
      <td>0.439</td>
      <td>0.310</td>
      <td>0.853</td>
      <td>0.471</td>
      <td>0.707</td>
      <td>0.445</td>
      <td>0.867</td>
      <td>0.523</td>
      <td>1.602</td>
      <td>0.860</td>
      <td>2.080</td>
      <td>0.999</td>
      <td>1.826</td>
      <td>0.903</td>
    </tr>
    <tr>
      <td>720</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.420</td>
      <td>0.299</td>
      <td>0.423</td>
      <td>0.302</td>
      <td>0.434</td>
      <td>0.305</td>
      <td>0.450</td>
      <td>0.324</td>
      <td>0.450</td>
      <td>0.317</td>
      <td>0.418</td>
      <td>0.296</td>
      <td>0.867</td>
      <td>0.493</td>
      <td>0.676</td>
      <td>0.423</td>
      <td>0.833</td>
      <td>0.502</td>
      <td>1.526</td>
      <td>0.839</td>
      <td>1.859</td>
      <td>0.927</td>
      <td>1.557</td>
      <td>0.795</td>
    </tr>
    <tr>
      <td rowspan="5">ETTh1</td>
      <td>96</td>
      <td>0.500</td>
      <td>0.493</td>
      <td>0.732</td>
      <td>0.556</td>
      <td>0.543</td>
      <td>0.506</td>
      <td>0.808</td>
      <td>0.610</td>
      <td>0.547</td>
      <td>0.503</td>
      <td>0.557</td>
      <td>0.519</td>
      <td>0.892</td>
      <td>0.625</td>
      <td>0.593</td>
      <td>0.529</td>
      <td>0.681</td>
      <td>0.570</td>
      <td>0.952</td>
      <td>0.650</td>
      <td>1.169</td>
      <td>0.832</td>
      <td>1.483</td>
      <td>0.910</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.690</td>
      <td>0.539</td>
      <td>0.872</td>
      <td>0.604</td>
      <td>0.748</td>
      <td>0.580</td>
      <td>0.928</td>
      <td>0.658</td>
      <td>0.720</td>
      <td>0.604</td>
      <td>0.711</td>
      <td>0.570</td>
      <td>0.940</td>
      <td>0.665</td>
      <td>0.652</td>
      <td>0.563</td>
      <td>0.725</td>
      <td>0.602</td>
      <td>0.943</td>
      <td>0.645</td>
      <td>1.221</td>
      <td>0.853</td>
      <td>1.525</td>
      <td>0.930</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.761</td>
      <td>0.620</td>
      <td>1.071</td>
      <td>0.721</td>
      <td>0.754</td>
      <td>0.595</td>
      <td>1.475</td>
      <td>0.861</td>
      <td>0.984</td>
      <td>0.727</td>
      <td>0.816</td>
      <td>0.619</td>
      <td>0.945</td>
      <td>0.653</td>
      <td>0.731</td>
      <td>0.594</td>
      <td>0.761</td>
      <td>0.624</td>
      <td>0.935</td>
      <td>0.644</td>
      <td>1.179</td>
      <td>0.832</td>
      <td>1.347</td>
      <td>0.870</td>
    </tr>
    <tr>
      <td>720</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.650</td>
      <td>0.550</td>
      <td>0.891</td>
      <td>0.627</td>
      <td>0.681</td>
      <td>0.560</td>
      <td>1.070</td>
      <td>0.710</td>
      <td>0.750</td>
      <td>0.611</td>
      <td>0.694</td>
      <td>0.569</td>
      <td>0.925</td>
      <td>0.647</td>
      <td>0.658</td>
      <td>0.562</td>
      <td>0.722</td>
      <td>0.598</td>
      <td>0.943</td>
      <td>0.646</td>
      <td>1.189</td>
      <td>0.839</td>
      <td>1.451</td>
      <td>0.903</td>
    </tr>
    <tr>
      <td rowspan="5">ETTh2</td>
      <td>96</td>
      <td>0.363</td>
      <td>0.409</td>
      <td>0.399</td>
      <td>0.420</td>
      <td>0.376</td>
      <td>0.421</td>
      <td>0.397</td>
      <td>0.427</td>
      <td>0.442</td>
      <td>0.456</td>
      <td>0.401</td>
      <td>0.421</td>
      <td>0.409</td>
      <td>0.420</td>
      <td>0.390</td>
      <td>0.424</td>
      <td>0.428</td>
      <td>0.468</td>
      <td>0.408</td>
      <td>0.423</td>
      <td>0.678</td>
      <td>0.619</td>
      <td>2.022</td>
      <td>1.006</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.375</td>
      <td>0.411</td>
      <td>0.487</td>
      <td>0.479</td>
      <td>0.418</td>
      <td>0.441</td>
      <td>0.438</td>
      <td>0.445</td>
      <td>0.617</td>
      <td>0.542</td>
      <td>0.452</td>
      <td>0.455</td>
      <td>0.483</td>
      <td>0.464</td>
      <td>0.457</td>
      <td>0.465</td>
      <td>0.496</td>
      <td>0.504</td>
      <td>0.497</td>
      <td>0.468</td>
      <td>0.845</td>
      <td>0.697</td>
      <td>3.534</td>
      <td>1.348</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.403</td>
      <td>0.421</td>
      <td>0.858</td>
      <td>0.660</td>
      <td>0.408</td>
      <td>0.439</td>
      <td>0.631</td>
      <td>0.553</td>
      <td>1.424</td>
      <td>0.849</td>
      <td>0.464</td>
      <td>0.469</td>
      <td>0.499</td>
      <td>0.479</td>
      <td>0.477</td>
      <td>0.483</td>
      <td>0.486</td>
      <td>0.496</td>
      <td>0.507</td>
      <td>0.481</td>
      <td>0.905</td>
      <td>0.727</td>
      <td>4.063</td>
      <td>1.451</td>
    </tr>
    <tr>
      <td>720</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.380</td>
      <td>0.413</td>
      <td>0.581</td>
      <td>0.519</td>
      <td>0.400</td>
      <td>0.433</td>
      <td>0.488</td>
      <td>0.475</td>
      <td>0.694</td>
      <td>0.577</td>
      <td>0.827</td>
      <td>0.615</td>
      <td>0.439</td>
      <td>0.448</td>
      <td>0.463</td>
      <td>0.454</td>
      <td>0.441</td>
      <td>0.457</td>
      <td>0.470</td>
      <td>0.489</td>
      <td>0.809</td>
      <td>0.681</td>
      <td>3.206</td>
      <td>1.268</td>
    </tr>
    <tr>
      <td rowspan="5">ETTm1</td>
      <td>96</td>
      <td>0.357</td>
      <td>0.390</td>
      <td>0.422</td>
      <td>0.424</td>
      <td>0.386</td>
      <td>0.405</td>
      <td>0.589</td>
      <td>0.510</td>
      <td>0.332</td>
      <td>0.374</td>
      <td>0.399</td>
      <td>0.414</td>
      <td>0.606</td>
      <td>0.518</td>
      <td>0.628</td>
      <td>0.544</td>
      <td>0.726</td>
      <td>0.578</td>
      <td>0.823</td>
      <td>0.587</td>
      <td>1.031</td>
      <td>0.747</td>
      <td>1.048</td>
      <td>0.733</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.432</td>
      <td>0.434</td>
      <td>0.448</td>
      <td>0.440</td>
      <td>0.440</td>
      <td>0.438</td>
      <td>0.703</td>
      <td>0.565</td>
      <td>0.358</td>
      <td>0.390</td>
      <td>0.441</td>
      <td>0.436</td>
      <td>0.681</td>
      <td>0.539</td>
      <td>0.666</td>
      <td>0.566</td>
      <td>0.750</td>
      <td>0.591</td>
      <td>0.844</td>
      <td>0.591</td>
      <td>1.087</td>
      <td>0.766</td>
      <td>1.097</td>
      <td>0.756</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.440</td>
      <td>0.442</td>
      <td>0.519</td>
      <td>0.482</td>
      <td>0.485</td>
      <td>0.459</td>
      <td>0.898</td>
      <td>0.641</td>
      <td>0.402</td>
      <td>0.416</td>
      <td>0.499</td>
      <td>0.467</td>
      <td>0.786</td>
      <td>0.597</td>
      <td>0.807</td>
      <td>0.628</td>
      <td>0.851</td>
      <td>0.659</td>
      <td>0.870</td>
      <td>0.603</td>
      <td>1.138</td>
      <td>0.787</td>
      <td>1.147</td>
      <td>0.775</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.593</td>
      <td>0.521</td>
      <td>0.708</td>
      <td>0.573</td>
      <td>0.577</td>
      <td>0.499</td>
      <td>0.948</td>
      <td>0.671</td>
      <td>0.511</td>
      <td>0.489</td>
      <td>0.767</td>
      <td>0.587</td>
      <td>0.796</td>
      <td>0.593</td>
      <td>0.822</td>
      <td>0.633</td>
      <td>0.857</td>
      <td>0.655</td>
      <td>0.893</td>
      <td>0.611</td>
      <td>1.245</td>
      <td>0.831</td>
      <td>1.200</td>
      <td>0.799</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.455</td>
      <td>0.446</td>
      <td>0.524</td>
      <td>0.479</td>
      <td>0.472</td>
      <td>0.450</td>
      <td>0.784</td>
      <td>0.596</td>
      <td>0.400</td>
      <td>0.417</td>
      <td>0.526</td>
      <td>0.476</td>
      <td>0.717</td>
      <td>0.561</td>
      <td>0.730</td>
      <td>0.592</td>
      <td>0.796</td>
      <td>0.620</td>
      <td>0.857</td>
      <td>0.598</td>
      <td>1.125</td>
      <td>0.782</td>
      <td>1.123</td>
      <td>0.765</td>
    </tr>
    <tr>
      <td rowspan="5">ETTm2</td>
      <td>96</td>
      <td>0.197</td>
      <td>0.278</td>
      <td>0.225</td>
      <td>0.300</td>
      <td>0.199</td>
      <td>0.280</td>
      <td>0.265</td>
      <td>0.339</td>
      <td>0.236</td>
      <td>0.326</td>
      <td>0.206</td>
      <td>0.288</td>
      <td>0.220</td>
      <td>0.299</td>
      <td>0.229</td>
      <td>0.320</td>
      <td>0.232</td>
      <td>0.322</td>
      <td>0.238</td>
      <td>0.316</td>
      <td>0.404</td>
      <td>0.485</td>
      <td>1.108</td>
      <td>0.772</td>
    </tr>
    <tr>
      <td>192</td>
      <td>0.254</td>
      <td>0.322</td>
      <td>0.275</td>
      <td>0.334</td>
      <td>0.256</td>
      <td>0.316</td>
      <td>0.310</td>
      <td>0.362</td>
      <td>0.306</td>
      <td>0.373</td>
      <td>0.264</td>
      <td>0.324</td>
      <td>0.311</td>
      <td>0.361</td>
      <td>0.394</td>
      <td>0.361</td>
      <td>0.291</td>
      <td>0.357</td>
      <td>0.298</td>
      <td>0.349</td>
      <td>0.479</td>
      <td>0.521</td>
      <td>1.317</td>
      <td>0.850</td>
    </tr>
    <tr>
      <td>336</td>
      <td>0.315</td>
      <td>0.350</td>
      <td>0.339</td>
      <td>0.371</td>
      <td>0.318</td>
      <td>0.353</td>
      <td>0.373</td>
      <td>0.399</td>
      <td>0.380</td>
      <td>0.423</td>
      <td>0.334</td>
      <td>0.367</td>
      <td>0.338</td>
      <td>0.366</td>
      <td>0.378</td>
      <td>0.427</td>
      <td>0.478</td>
      <td>0.517</td>
      <td>0.353</td>
      <td>0.380</td>
      <td>0.552</td>
      <td>0.555</td>
      <td>1.415</td>
      <td>0.879</td>
    </tr>
    <tr>
      <td>720</td>
      <td>0.421</td>
      <td>0.421</td>
      <td>0.464</td>
      <td>0.441</td>
      <td>0.460</td>
      <td>0.436</td>
      <td>0.478</td>
      <td>0.454</td>
      <td>0.674</td>
      <td>0.583</td>
      <td>0.454</td>
      <td>0.432</td>
      <td>0.509</td>
      <td>0.465</td>
      <td>0.523</td>
      <td>0.510</td>
      <td>0.553</td>
      <td>0.538</td>
      <td>0.475</td>
      <td>0.445</td>
      <td>0.701</td>
      <td>0.627</td>
      <td>1.822</td>
      <td>0.984</td>
    </tr>
    <tr>
      <td>Avg.</td>
      <td>0.296</td>
      <td>0.342</td>
      <td>0.325</td>
      <td>0.361</td>
      <td>0.308</td>
      <td>0.346</td>
      <td>0.356</td>
      <td>0.388</td>
      <td>0.399</td>
      <td>0.426</td>
      <td>0.314</td>
      <td>0.352</td>
      <td>0.344</td>
      <td>0.372</td>
      <td>0.381</td>
      <td>0.404</td>
      <td>0.388</td>
      <td>0.433</td>
      <td>0.341</td>
      <td>0.372</td>
      <td>0.534</td>
      <td>0.547</td>
      <td>1.415</td>
      <td>0.871</td>
    </tr>
  </tbody>
</table>
## Appendix E. Ablation Studies and Parameter Sensitivity

We provide the t-SNE and PCA visualization of semantic anchor and prefix-prompted time series embeddings with different $V'$ in Figure 6. We observe semantic anchor embeddings display a continued spanning pattern among the joint space, whereas the prompted time series representation shows only a slight visual difference. It is reasonable since it is primarily controlled by the scaling factor $\lambda$.

<table>
  <thead>
    <tr>
      <th rowspan="2">Ablation Setting</th>
      <th colspan="4">Long-term Forecasting</th>
    </tr>
    <tr>
      <th>ETTh2-96</th>
      <th>ETTh2-192</th>
      <th>ETTm2-96</th>
      <th>ETTm2-192</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>w/o Prompt & Alignment and w/o Decomposition</td>
      <td>0.289</td>
      <td>0.358</td>
      <td>0.170</td>
      <td>0.231</td>
    </tr>
    <tr>
      <td>w/ Prompt & Alignment and w/o Decomposition</td>
      <td>0.287</td>
      <td>0.353</td>
      <td>0.166</td>
      <td>0.228</td>
    </tr>
    <tr>
      <td>w/ Prompt & Alignment and w/ Decomposition</td>
      <td>0.278</td>
      <td>0.346</td>
      <td>0.165</td>
      <td>0.222</td>
    </tr>
  </tbody>
</table>
> Table 10 (see PDF p. 17). Ablation studies on ETTh2 and ETTm2 in predicting 96 and 192 steps (MSE reported).

> Figure 6 (see PDF p. 18). The t-SNE and PCA plots of semantic anchor and prefix-prompted time series embeddings with different $V'$.

## Appendix F. Visualization

In this section, we provide the visualizations of the forecasting cases of S²IP-LLM on ETTm2, Electricity, and Weather datasets under the input-512-predict-96 setting. As shown in Figure 7, S²IP-LLM achieves exceptionally good forecasting results across various datasets.

> Figure 7 (see PDF p. 19). Long-term forecasting visualization cases for ETTm2, Electricity, and Weather. Blue lines are the ground truths and orange lines are the model predictions. The vertical line indicates where the prediction starts.

[^tslib]: `https://github.com/thuml/Time-Series-Library`

[^timellm]: We reproduced results through public available code: `https://github.com/KimMeen/Time-LLM`. For Time-LLM variants, `L` denotes the LLaMA backbone, and `G` refers to the GPT-2 backbone.
