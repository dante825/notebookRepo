# Credit Risk Model
<p>The purpose of these scripts are for practice purpose.
Understanding how to train machine learning model in Spark.</p>

## Biased dataset
<p>The credit risk dataset is a biased dataset.</p>

| Risk | RiskIndex | Frequency |
|------|-----------|-----------|
| Good | 0     |  700 |
| Bad  | 1      | 300 |


<p>Training models with a biased dataset would be resulting in a biased model and inaccurate result.</p>


## The first model (train with the biased data as it is)

Precision: 0.83</br>
Recall: 0.90</br>
f1 score: 0.77</br>

|     | precision | recall | f1-score | support |
|-----|-----------|--------|----------|---------|
| 0.0 | 0.83      |   0.90 | 0.86     | 144     |
| 1.0 | 0.58      |   0.44 |  0.50    | 48      |
|     |           |        |  0.78    | 192     |
| accuracy| 0.71  |   0.67 |   0.68   | 192     |
| macro avg | 0.71 | 0.67   |  0.68  | 192    |
| weighted avg | 0.77 | 0.78  | 0.77 | 192   |


## Rebalace the dataset
