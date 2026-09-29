# Empowering Time Series Forecasting with LLM-Agents

Chin-Chia Michael Yeh*, Vivian Lai, Uday Singh Saini, Xiran Fan, Yujie Fan, Junpeng Wang, Xin Dai, Yan Zheng

Visa Research  
Foster City, CA, USA

*Correspondence: `miyeh@visa.com`*

*arXiv:2508.04231v2 [cs.LG], November 26, 2025.*

**Abstract.** Large Language Model (LLM) powered agents have emerged as effective planners for Automated Machine Learning (AutoML) systems. While most existing AutoML approaches focus on automating feature engineering and model architecture search, recent studies in time series forecasting suggest that lightweight models can often achieve state-of-the-art performance. This observation led us to explore improving data quality, rather than model architecture, as a potentially fruitful direction for AutoML on time series data. We propose DCATS, a Data-Centric Agent for Time Series. DCATS leverages metadata accompanying time series to clean data while optimizing forecasting performance. We evaluated DCATS using four time series forecasting models on a large-scale traffic volume forecasting dataset. Results demonstrate that DCATS achieves an average 6% error reduction across all tested models and time horizons, highlighting the potential of data-centric approaches in AutoML for time series forecasting. The source code is available at: https://sites.google.com/view/ts-agent.

**CCS Concepts**

- Information systems -> Spatial-temporal systems
- Computing methodologies -> Multi-agent planning

**Keywords**

Agentic AI, Time Series, Forecasting, Spatial-Temporal

## 1. Introduction

Time series data pervades diverse real-world applications, including traffic monitoring, financial transactions, and ride-share demand forecasting [11]. Accurate time series forecasting, in particular, is a critical research area with extensive applications [7, 11, 27]. Concurrently, Large Language Models (LLMs) have demonstrated considerable success in powering Automatic Machine Learning (AutoML) systems for various general machine learning tasks [1, 8, 16, 19]. This confluence motivates our exploration into developing an LLM-powered AutoML system specifically designed for the unique challenges of time series forecasting.

However, directly applying general AutoML principles to time series forecasting can overlook domain-specific optimization opportunities. Recent findings indicate that lightweight time series models can achieve state-of-the-art performance [3, 9, 22, 27], shifting focus towards data quality. This aligns with the principles of data-centric AI [18, 28], which prioritizes data refinement over complex model architectures. We identify a significant opportunity in leveraging LLMs to automate this data-centric approach for time series forecasting, an area that remains underexplored. Therefore, this paper introduces DCATS (Data-Centric Agent for Time Series), an LLM-powered agent that focuses on intelligently refining training data rather than solely optimizing model architectures.

DCATS operates by strategically enriching the training dataset through the selection of relevant auxiliary time series. The LLM-agent formulates a dataset expansion plan by reasoning over the metadata associated with available time series. For instance, as illustrated in Fig. 1, to forecast traffic volume for a highway entrance near San Mateo, California, DCATS might identify and incorporate historical data from proximate locations like Burlingame or from geographically distant areas exhibiting similar temporal patterns. Furthermore, the agent iteratively refines this data selection plan by evaluating the impact of different enrichment strategies, thereby optimizing the final dataset.

> Figure 1 (see PDF p. 1). The proposed DCATS framework focuses on refining data quality rather than model design. Note, the LLM-agent makes decisions based on validation errors.

To validate our approach, we conducted a preliminary study employing a large-scale traffic volume forecasting dataset [11], notable for its extensive metadata. Our findings demonstrate that DCATS achieves a 6% reduction in forecasting error compared to a baseline model trained using all available time series. This result underscores the efficacy of leveraging LLM-agents for data-centric optimization in time series forecasting and signals a promising avenue for future research across various domains.

Our key contributions are as follows:

- We introduce DCATS, a novel data-centric agentic framework designed specifically for time series forecasting problems.
- We present a preliminary study using traffic volume time series, showcasing a 6% performance improvement over alternative solutions that do not employ LLM-agents.
- We showcase the ability of LLMs to perform reasoning over time series metadata to formulate effective data enrichment strategies for improved forecasting accuracy.

## 2. Related Work

Traditional AutoML systems primarily focus on model selection, hyperparameter tuning, and pipeline configuration [5, 13, 17], employing various optimization and meta-learning techniques to achieve automation. A parallel development has been neural architecture search, which specifically targets the automated design of neural networks using reinforcement learning or evolutionary algorithms [4, 10, 14, 15, 29]. However, these approaches predominantly emphasize model optimization rather than data quality enhancement. Recent advancements in LLMs have enabled more sophisticated AutoML approaches, with systems like AIDE [16] employing LLM-agents for solving general machine learning problems. While AIDE demonstrates the potential of LLM-powered AutoML, it does not fully exploit the unique characteristics of time series data. The emerging paradigm of data-centric AI suggests that improving data quality can yield greater performance gains than model refinement alone [18, 28], particularly relevant for time series forecasting where recent research shows lightweight models with high-quality data can match or exceed complex architectures [3, 9, 22, 27]. Despite these findings, few AutoML systems have been designed with a data-centric focus for time series forecasting, which presents unique challenges including temporal dependencies and domain-specific metadata [7]. To the best of our knowledge, DCATS represents the first LLM-powered data-centric AutoML framework designed specifically for time series forecasting problems, leveraging rich metadata to intelligently select and augment training data.

## 3. Methodology

The proposed DCATS framework comprises four core components, as illustrated in Fig. 2. We will use the traffic volume prediction use case to demonstrate the application of the proposed framework for time series forecasting. The key components of DCATS are: 1) a time series dataset consisting of multiple univariate time series, 2) an accompanying metadata database containing shared background information and additional details about each univariate time series, 3) an LLM-agent responsible for driving the automatic forecasting model building process, and 4) a forecasting module that builds models and validates performance based on the agent's requests.

> Figure 2 (see PDF p. 2). Overall design of the proposed DCATS.

The framework is initiated when users submit a query expressing their intent to build a forecasting model for specific time series within the dataset. Upon receiving the query, the LLM-agent retrieves background information about the entire time series dataset and detailed information about each time series from the metadata database. Based on this information, the agent generates a list of proposals, each containing instructions for creating a sub-dataset from the time series dataset and an explanation of the strategy behind it. The proposal generation process is detailed in Section 3.1. Each proposal is then evaluated by the forecasting module, which trains models using the sub-dataset and reports performance on the validation dataset. The forecasting module is described in detail in Section 3.2. Using the validation performance, the LLM-agent provides a new set of proposals for the next evaluation round, typically refining the winning strategy from the previous iteration. The refinement process is explained in Section 3.3. This iterative process continues until no proposals in the current round show improvement over the best-so-far solution.

### 3.1. Initial Proposal

The LLM-agent generates initial proposals using a five-section prompt: 1) background, 2) task, 3) guidelines, 4) neighbor sets, and 5) output format. The background section provides context for the time series dataset. For the traffic volume forecasting problem, the prompt is shown in Listing 1.

**Listing 1.** The background.

```text
# Background
We have a spatio-temporal dataset containing 8,600 locations, each with a
unique `location_id` (integer from 0 to 8,599). Each location has a
univariate time series representing traffic volume changes over time,
split into training and validation datasets. Our goal is to build time
series forecasting models for specific locations.
```

The task section includes information about the target time series and the specific task, i.e. providing a number of proposals. An example task section for location 1201 is shown in Listing 2, which includes historical total volume, city name, county name, population, and freeway information.

**Listing 2.** The task.

```text
# Task
Construct a time series forecasting model for location `location_id=1201`.
Location details: location_id=1201, historical_total_volume=2229867. This
location is in Campbell, a city located in Santa Clara County, California.
Campbell has a population of approximately 41,700 residents. The location
is on freeway SR87-N, which has 3 lanes.

While we could use only data from `location_id=1201`, including data from
other locations may improve the model's performance. We request 5
proposals, each suggesting a list of `location_id`s from the neighbors of
`location_id=1201`.
```

Additional guidelines are provided to the agent, as shown in Listing 3, detailing what is known about each time series and how sub-datasets should be created. Sub-datasets are created by selecting time series from the "neighbors" of the target time series, defined by similarity in road network distance, local correlation, or geodetic distance. These "neighbors" serve a similar function to retrieved documents in a RAG system [2, 6].

An example of how neighbors are presented to the agent is shown in Listing 4, providing various details about each time series or location. For brevity, the paper shows only the nearest neighbor for location 1201 based on local correlation or pattern similarity.

The output format section specifies the format for each proposal, as shown in Listing 5.

**Listing 3.** The guidelines.

```text
## Guidelines:
- Ensure each location is selected only once per proposal.
- Utilize the provided neighbor sets based on different criteria (road
  network, temporal pattern similarity, and geodetic distance).
- Consider the additional details provided for each location, including:
  - Similarity or Distance
  - Historical Total Volume
  - City
  - County
  - Population
  - Freeway
  - Number of Lanes
- Balance the selection of neighbors across different criteria to create
  diverse and informative proposals.
- Explain the rationale behind each proposal, highlighting how the
  selected neighbors might contribute to improving the forecasting model.
```

**Listing 4.** The neighbor sets.

```text
## Neighbor Sets:
- Nearest Neighbors Selected Based on Temporal Pattern Similarity.
Neighbors are selected based on the Pearson correlation coefficient
between the most similar patterns observed at two locations. This
correlation ranges from -1 to 1, indicating the strength and direction of
the linear relationship between patterns. This neighbor selection method
is particularly valuable because similar patterns across locations suggest
that people passing by exhibit comparable behaviors. Consequently, sharing
data between these locations when training a model can provide crucial
insights into common temporal trends and significantly enhance the model's
predictive capabilities. By focusing on temporal similarities rather than
geographical proximity, this approach can uncover hidden relationships
between seemingly unrelated locations, potentially leading to more nuanced
and accurate predictions in various applications such as urban planning,
traffic management, or consumer behavior analysis.
1. location_id=1205, similarity=0.9849, historical_total_volume=2598232.
   This location is in San Jose, a city located in Santa Clara County,
   California. San Jose has a population of approximately 969,655 residents.
   The location is on freeway SR87-N, which has 2 lanes.
... (omitted for brevity)
```

**Listing 5.** The output format.

```text
# Output Format
Please output each proposal using the following format:

Proposal {proposal_number}
Explanation: {reasoning_behind_the_proposal}
Neighbors: [{location_id_for_neighbor_1}, {location_id_for_neighbor_2},
{location_id_for_neighbor_3}, ..., {location_id_for_last_neighbor}]
```

### 3.2. Forecasting Module

The forecasting module trains time series forecasting models using the sub-dataset and evaluates their performance on the validation set. The DCATS framework implements four different time series forecasting models:

- Linear [27]: A simple yet effective forecasting model.
- MLP [3, 23]: A natural extension of the linear model with increased learning capability due to additional parameters.
- SparseTSF [9]: A compact and effective forecasting model.
- UltraSTF [22]: Another compact model with improved cost/learning capability trade-off.

To enhance model convergence speed, the framework first trains a foundation model [12, 20, 24, 25] using all available time series data before user queries. It then uses the sub-dataset proposed by the LLM-agent to fine-tune the foundation model when testing different proposals. To further refine the data before fine-tuning, it removes the 10% most anomalous data from the sub-dataset using discord-based anomaly detection methods [21, 26].

### 3.3. Proposal Refinement

The proposal refinement agent uses a six-section prompt: 1) objective, 2) background, 3) experiment results, 4) task, 5) additional considerations, and 6) output format. The overall objective statement and background for the refinement agent are shown in Listing 6, providing information about the target time series and current best-so-far validation performance.

**Listing 6.** The objective.

```text
# Objective
Develop an improved time series forecasting model for `location_id=1201`,
leveraging data from other relevant locations.

# Background
- Target location: `location_id=1201`
- Target location information: location_id=1201,
  historical_total_volume=2229867. This location is in Campbell, a city
  located in Santa Clara County, California. Campbell has a population of
  approximately 41,700 residents. The location is on freeway SR87-N, which
  has 3 lanes.
- Best performance achieved (Mean Absolute Error): 7.4190
```

The experiment section lists all sub-dataset setups and their corresponding performance, as shown in Listing 7. For brevity, the paper shows only the experiment result for one of the proposals. Explanations are included to help the agent reason about the main factors behind good performance, and results are ordered based on performance.

**Listing 7.** The experiment results.

```text
# Previous Experiment Results (Ranked from Best to Worst)
Proposal 1
Explanation: This proposal aims to create a well-rounded selection by
integrating one neighbor from each of the three criteria: road network
similarity, temporal pattern similarity, and geodetic proximity. Each of
these neighbors comes from a different set, ensuring a diverse set of data
inputs to help the model generalize better over different types of
proximity and similarity.
Neighbors: [1200, 1202, 1204, 1205, 1223]
Performance (Mean Absolute Error): 7.4190
... (omitted for brevity)
```

Detailed tasks are given to the agent in Listing 8, along with additional considerations to aid in reasoning. The output format section again follows Listing 5.

**Listing 8.** The task.

```text
# Task
Based on the experiment results, baseline performance, and best-so-far
performance, provide a new set of proposals to further enhance the
forecasting model. Each proposal should:
1. Include a list of `location_id`s selected from the neighbors of
   `location_id=1201`
2. Ensure no duplicate selections within a single proposal
3. Aim to minimize the Mean Absolute Error (MAE)

# Additional Considerations
- Analyze the characteristics of the target location and its neighbors
- Identify patterns in successful proposals from previous experiments
- Explore diverse combinations of locations that may capture various
  aspects of time series behavior
```

## 4. Experiment

This experiment demonstrates the effectiveness of the DCATS framework in improving time series forecasting performance and providing valuable insights through explanations. Key findings show consistent performance improvements across multiple models and the framework's ability to leverage diverse metadata for optimal sub-dataset selection.

### 4.1. Dataset and Experiment Setup

The authors utilized the LargeST dataset [11], comprising traffic time series from 8,600 sensors across California. The dataset includes metadata such as location coordinates, county, freeway name, and number of lanes. They augmented this metadata with city names, populations, and historical transaction volumes. Sensor readings were aggregated into 15-minute intervals, resulting in 96 intervals per day over 35,040 time steps. Each sub-dataset was split into training, validation, and test sets using a 6:2:2 ratio. The framework was applied to the training and validation sets, with performance reported on the test set. The authors tested 60 user queries with randomly selected locations, forecasting the next 12 intervals for each sensor at each timestamp. Performance was evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE). GPT-4 Turbo was employed as the LLM agent, generating five proposals per round.

### 4.2. Experiment Results

The authors applied the DCATS framework to four time series forecasting models, see Section 3.2, with results shown in Table 1. Performance metrics were averaged across all 12 time steps and 60 queries.

**Table 1.** The DCATS framework improves performance across all scenarios.

| Method | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| Linear | 37.31 | 74.01 | 15.12% |
| Linear+DCATS | 35.91 | 72.91 | 14.20% |
| % improvement | 3.77% | 1.48% | 6.14% |
| MLP | 34.07 | 67.68 | 13.33% |
| MLP+DCATS | 31.26 | 63.34 | 12.34% |
| % improvement | 8.26% | 6.41% | 7.42% |
| SparseTSF | 37.92 | 74.19 | 16.83% |
| SparseTSF+DCATS | 34.88 | 69.20 | 15.46% |
| % improvement | 8.02% | 6.73% | 8.14% |
| UltraSTF | 29.77 | 60.92 | 10.26% |
| UltraSTF+DCATS | 28.61 | 57.78 | 9.79% |
| % improvement | 3.91% | 5.16% | 4.55% |

The DCATS framework consistently improved performance across all tested methods, with an average improvement of 6% across all queries, models, and metrics. This model-agnostic improvement suggests that the framework's data selection process enhances forecast quality regardless of the underlying model. SparseTSF showed the lowest performance, while UltraSTF combined with the DCATS framework achieved the best results.

To understand the agent's proposal generation process, the authors created a word cloud, Fig. 3, using explanations associated with the best proposal for all user queries. The word cloud reveals that terms related to road networks, patterns, and geodetic information have similar prominence, indicating balanced utilization of different neighbor selection criteria across queries.

> Figure 3 (see PDF p. 4). Word cloud generated from explanations associated with the best proposal for each user query.

Next, examining individual query explanations reveals how the framework adapts neighbor selection criteria based on specific query characteristics. The paper selected three proposals that demonstrate how the best proposal for different queries requires neighbors selected using different criteria:

- Concentrates on mixing the top-performing neighbors from road network similarity and temporal patterns, which are crucial for understanding structural and sequential traffic volume dynamics.
- This proposal focuses on refining the successful aspects of the first proposal from previous experiments, which utilized geodetic proximity... (omitted for brevity).
- This proposal includes neighbors selected based on high similarity in traffic volume and temporal patterns, specifically focusing on the freeway SR60-W in Diamond Bar, ... (omitted for brevity).

These examples highlight how the agent uses neighbors from different neighbor sets in different proposals, sometimes utilizing specific metadata such as freeway names, cities, e.g. Diamond Bar, or historical traffic volumes to create each proposal. This observation further motivates the proposed system, as it would be highly labor-intensive for a human to manually analyze the metadata 60 times for the 60 queries. The DCATS framework automates this process, efficiently building the best possible forecast model for each time series within a dataset.

## 5. Conclusion

This paper presents a preliminary study on utilizing LLMs for building time series forecasting models automatically, employing the principles of data-centric AI. The authors propose the DCATS framework and demonstrate its capability on a traffic volume forecasting dataset, where it improves upon existing systems by 6%. The results suggest that LLMs have significant potential in enhancing time series analysis and forecasting tasks. For future work, it would be valuable to investigate the performance of the DCATS framework when applied to time series from diverse domains and compare agents powered by different LLMs.

## GenAI Usage Disclosure

The proposed agent is powered by a large language model (LLM). Additionally, the authors use LLMs to proofread and polish their writing.

## References

[1] Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, et al. 2024. MLE-bench: Evaluating machine learning agents on machine learning engineering. *arXiv preprint arXiv:2410.07095* (2024).

[2] Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh, Menghai Pan, Chin-Chia Michael Yeh, Guanchu Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Mahashweta Das, et al. 2024. MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation. *arXiv preprint arXiv:2501.00332* (2024).

[3] Si-An Chen, Chun-Liang Li, Nate Yoder, Sercan O Arik, and Tomas Pfister. 2023. TSMixer: An all-mlp architecture for time series forecasting. *arXiv preprint arXiv:2303.06053* (2023).

[4] Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. 2019. Neural architecture search: A survey. *Journal of Machine Learning Research* 20, 55 (2019), 1-21.

[5] Matthias Feurer, Katharina Eggensperger, Stefan Falkner, Marius Lindauer, and Frank Hutter. 2022. Auto-sklearn 2.0: Hands-free automl via meta-learning. *Journal of Machine Learning Research* 23, 261 (2022), 1-61.

[6] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented generation for large language models: A survey. *arXiv preprint arXiv:2312.10997* 2 (2023).

[7] Rakshitha Godahewa, Christoph Bergmeir, Geoffrey I Webb, Rob J Hyndman, and Pablo Montero-Manso. 2021. Monash time series forecasting archive. *arXiv preprint arXiv:2105.06643* (2021).

[8] Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. 2023. MLAgentBench: Evaluating language agents on machine learning experimentation. *arXiv preprint arXiv:2310.03302* (2023).

[9] Shengsheng Lin, Weiwei Lin, Wentai Wu, Haojun Chen, and Junjie Yang. 2024. SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters. *arXiv preprint arXiv:2405.00946* (2024).

[10] Hanxiao Liu, Karen Simonyan, and Yiming Yang. 2018. DARTS: Differentiable architecture search. *arXiv preprint arXiv:1806.09055* (2018).

[11] Xu Liu, Yutong Xia, Yuxuan Liang, Junfeng Hu, Yiwei Wang, Lei Bai, Chao Huang, Zhenguang Liu, Bryan Hooi, and Roger Zimmermann. 2023. LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting. *arXiv preprint arXiv:2306.08259* (2023).

[12] John A Miller, Mohammed Aldosari, Farah Saeed, Nasid Habib Barna, Subas Rana, I Budak Arpinar, and Ninghao Liu. 2024. A survey of deep learning and foundation models for time series forecasting. *arXiv preprint arXiv:2401.13912* (2024).

[13] Randal S Olson and Jason H Moore. 2016. TPOT: A tree-based pipeline optimization tool for automating machine learning. In *Workshop on automatic machine learning*. PMLR, 66-74.

[14] Hieu Pham, Melody Guan, Barret Zoph, Quoc Le, and Jeff Dean. 2018. Efficient neural architecture search via parameters sharing. In *International conference on machine learning*. PMLR, 4095-4104.

[15] Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V Le. 2019. Regularized evolution for image classifier architecture search. In *Proceedings of the AAAI conference on artificial intelligence*, Vol. 33. 4780-4789.

[16] Dominik Schmidt, Zhengyao Jiang, and Yuxiang Wu. 2024. AIDE: Human-Level Performance on Data Science Competitions. https://www.weco.ai/blog/technical-report.

[17] Chris Thornton, Frank Hutter, Holger H Hoos, and Kevin Leyton-Brown. 2013. Auto-WEKA: Combined selection and hyperparameter optimization of classification algorithms. In *Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining*. 847-855.

[18] Junpeng Wang, Shixia Liu, and Wei Zhang. 2024. Visual analytics for machine learning: A data perspective survey. *IEEE Transactions on Visualization and Computer Graphics* 30, 12 (2024), 7637-7656.

[19] Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. 2024. OpenHands: An open platform for ai software developers as generalist agents. In *The Thirteenth International Conference on Learning Representations*.

[20] Chin-Chia Michael Yeh, Xin Dai, Huiyuan Chen, Yan Zheng, Yujie Fan, Audrey Der, Vivian Lai, Zhongfang Zhuang, Junpeng Wang, Liang Wang, et al. 2023. Toward a foundation model for time series data. In *Proceedings of the 32nd ACM International Conference on Information and Knowledge Management*. 4400-4404.

[21] Chin-Chia Michael Yeh, Audrey Der, Uday Singh Saini, Vivian Lai, Yan Zheng, Junpeng Wang, Xin Dai, Zhongfang Zhuang, Yujie Fan, Huiyuan Chen, et al. 2024. Matrix Profile for Anomaly Detection on Multidimensional Time Series. *arXiv preprint arXiv:2409.09298* (2024).

[22] Chin-Chia Michael Yeh, Xiran Fan, Zhimeng Jiang, Yujie Fan, Huiyuan Chen, Uday Singh Saini, Vivian Lai, Xin Dai, Junpeng Wang, Zhongfang Zhuang, Liang Wang, and Yan Zheng. 2025. A Compact Model for Large-Scale Time Series Forecasting. *arXiv preprint arXiv:2502.20634* (2025).

[23] Chin-Chia Michael Yeh, Yujie Fan, Xin Dai, Uday Singh Saini, Vivian Lai, Prince Osei Aboagye, Junpeng Wang, Huiyuan Chen, Yan Zheng, Zhongfang Zhuang, et al. 2024. RPMixer: Shaking up time series forecasting with random projections for large spatial-temporal data. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 3919-3930.

[24] Chin-Chia Michael Yeh, Uday Singh Saini, Xin Dai, Xiran Fan, Shubham Jain, Yujie Fan, Jiarui Sun, Junpeng Wang, Menghai Pan, Yuzhong Dou, Yingtong Chen, Vineeth Rakesh, Liang Wang, Yan Zheng, and Mahashweta Das. 2025. TREASURE: A Transformer-Based Foundation Model for High-Volume Transaction Understanding. *arXiv preprint arXiv:2511.19693* (2025).

[25] Chin-Chia Michael Yeh, Uday Singh Saini, Junpeng Wang, Xin Dai, Xiran Fan, Yujie Sun, Jiarui Fan, and Yan Zheng. 2025. TiCT: A Synthetically Pre-Trained Foundation Model for Time Series Classification. *arXiv preprint arXiv:2511.19694* (2025).

[26] Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, Yifei Ding, Hoang Anh Dau, Diego Furtado Silva, Abdullah Mueen, and Eamonn Keogh. 2016. Matrix profile I: all pairs similarity joins for time series: a unifying view that includes motifs, discords and shapelets. In *2016 IEEE 16th international conference on data mining (ICDM)*. IEEE, 1317-1322.

[27] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. 2023. Are transformers effective for time series forecasting?. In *Proceedings of the AAAI conference on artificial intelligence*, Vol. 37. 11121-11128.

[28] Daochen Zha, Zaid Pervaiz Bhat, Kwei-Herng Lai, Fan Yang, Zhimeng Jiang, Shaochen Zhong, and Xia Hu. 2025. Data-centric artificial intelligence: A survey. *Comput. Surveys* 57, 5 (2025), 1-42.

[29] Barret Zoph and Quoc V Le. 2016. Neural architecture search with reinforcement learning. *arXiv preprint arXiv:1611.01578* (2016).
