# Transformer Models in Time Series Forecasting

## Introduction

The rapidly evolving field of Large Language Models (LLMs) like ChatGPT, stemming from transformer models, has been highly influential in the Natural Language Processing (NLP) domain. While they are highly effective in handling complex data structures, especially in the context of languages and texts, the application of transformer models in time series forecasting remains under-explored.

This research aims to answer the question, "Are transformer models in time series forecasting worth the hassle? A data-driven approach." By investigating various models, including single layer perceptron and several deep learning models, and their performance on complex data sets like wind turbine data, we seek to uncover if the inherent complexity of transformer models justifies their utilization in time series forecasting.

## Objectives

1. **Comparative Analysis**: Compare transformer models with existing linear models.
2. **Performance Evaluation**: Assess the models on diverse metrics, such as accuracy, computational costs, and resource usage.
3. **Applications in Renewable Energy**: Understand how predictive models can contribute to the stability of the electricity grid and aid in the planning of electricity consumption and production.

## Data

The study utilizes two distinct data sets:

1. **SCADA Wind Farm Data**: 10-minute SCADA and event data from 6 Senvion MM92s at the Kelmarsh wind farm and 13 Senvion MM92s at the Penmanshiel wind farm, grouped by year from 2016 to mid-2021.

The data is not presented here, due to the file size.

## Methodology

1. **Selection of Models**: Several transformer models, an linear model as a benchmark.
2. **Pre-processing**: Detect outliers & stationarity and handle missing values.
3. **Model Training and Tuning**: Utilize various publicly available libraries, fine-tuning via hyperparameter search.
4. **Evaluation**: Assess on forecast horizons of 10 minutes, 1 hour, 12 hours, and 24 hours, and rate based on total computational time and resources.
5. **Result Presentation**: Print and visualize the results for better comparability.

## Research Questions

The primary research focus is to understand if transformer models justify their complexity and potential benefits over other linear models. Specific areas of exploration include:

- Direct comparison in terms of performance, time, and resources.
- Understanding if the depth of the network affects the results.
- Quantifying the benefits of increased predictive accuracy versus development costs.

## Conclusion

The deep learning models outperform the linear model, but the computational costs are several times higher.

| Forecast Horizon | Model       | Avg Epoch Duration (s) | Avg RAM (GB) | Avg GPU (GB) | Test Loss  |
|------------------|-------------|------------------------|--------------|--------------|------------|
| 1                | Linear      | 111.42                 | 8.19         | 0.42         | 706.81     |
| 1                | eFormer     | 258.91                 | 8.35         | 3.38         | 694.00     |
| 1                | Transformer | 224.38                 | 7.68         | 1.22         | 705.59     |
| 1                | Informer    | 606.45                 | 13.91        | 1.10         | 706.73     |
|                  |             |                        |              |              |            |
| 6                | Linear      | 107.84                 | 8.29         | 0.42         | 707.83     |
| 6                | eFormer     | 251.89                 | 8.35         | 3.38         | 711.03     |
| 6                | Transformer | 231.44                 | 13.83        | 1.76         | 703.80     |
| 6                | Informer    | 546.68                 | 8.40         | 1.49         | 695.16     |
|                  |             |                        |              |              |            |
| 72               | Linear      | 147.70                 | 24.99        | 0.42         | 705.61     |
| 72               | eFormer     | 420.58                 | 25.91        | 3.38         | 712.72     |
| 72               | Transformer | 500.17                 | 25.52        | 9.69         | 688.56     |
|                  |             |                        |              |              |            |
| 144              | Linear      | 213.82                 | 44.92        | 0.42         | 705.36     |
| 144              | eFormer     | 754.87                 | 46.62        | 3.38         | 702.41     |
| 144              | Transformer | 882.95                 | 46.44        | 28.54        | 705.18     |

This table details average duration for an epoch, RAM and GPU usage, and the test loss across different forecast horizons.


## How to Run the Code

The .ipynb files for the respective models contain all the necessary code to be run on their own. After setting the hyperparameters and the Loss functions and forecast horizons to investigate. Simply run the files.
For the visualizations the .ipynb file reads in all results automatically and prints the desired LaTeX tables and graphs.

## Contact

For any questions or collaboration, please contact me at jan.besler@student.uni-tuebingen.de .
