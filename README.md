# Transformer Models in Time Series Forecasting

## Introduction

The rapidly evolving field of Large Language Models (LLMs) like ChatGPT, stemming from transformer models, has been highly influential in the Natural Language Processing (NLP) domain. While they are highly effective in handling complex data structures, especially in the context of languages and texts, the application of transformer models in time series forecasting remains under-explored.

This research aims to answer the question, "Are transformer models in time series forecasting worth the hassle? A data-driven approach." By investigating various models, including LSTM and Random Forest, and their performance on complex and less complex data sets like wind turbine data and electricity consumption, we seek to uncover if the inherent complexity of transformer models justifies their utilization in time series forecasting.

## Objectives

1. **Comparative Analysis**: Compare transformer models with existing models like LSTM and Random Forest, focusing on different data complexity levels.
2. **Performance Evaluation**: Assess the models on diverse metrics, such as accuracy, computational costs, and resource usage.
3. **Applications in Renewable Energy**: Understand how predictive models can contribute to the stability of the electricity grid and aid in the planning of electricity consumption and production.
4. **Probabilistic Approach**: Explore the use of probabilistic models to quantify uncertainty and its benefits and challenges.

## Data

The study utilizes two distinct data sets:

1. **SCADA Wind Farm Data**: 10-minute SCADA and event data from 6 Senvion MM92s at the Kelmarsh wind farm, grouped by year from 2016 to mid-2021 (more complex).
2. **Electricity Consumption Data**: Data from 4 manufacturing companies in Germany provided by Fraunhofer IPA, available in 15-minute intervals (less complex).

The data is not presented here, due to disclosure by the companies and file size.

## Methodology

1. **Selection of Models**: Several transformer models, an LSTM model, and Random Forest as a benchmark.
2. **Pre-processing**: Detect outliers & stationarity and handle missing values.
3. **Model Training and Tuning**: Utilize various publicly available libraries, fine-tuning via hyperparameter search.
4. **Evaluation**: Assess on forecast horizons of 1 hour, 24 hours, and 96 hours, and rate based on total computational time and resources.
5. **Result Presentation**: Print and visualize the results for better comparability.

## Research Questions

The primary research focus is to understand if transformer models justify their complexity and potential benefits over other models like LSTM. Specific areas of exploration include:

- Direct comparison in terms of performance, time, and resources.
- Understanding if the depth of the network affects the results.
- Quantifying the benefits of increased predictive accuracy versus development costs.

## Conclusion (To be Added)

This section will provide a summary of the findings and implications.

## How to Run the Code (If Applicable)

Instructions on how to run any associated code or access the datasets.

## Acknowledgements

Acknowledging any contributors, datasets, or resources used.

## Contact

For any questions or collaboration, please contact me at jan.besler@student.uni-tuebingen.de .
