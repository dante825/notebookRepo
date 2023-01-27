# Credit Risk Model
<p>The purpose of these scripts are for practice purpose.
Understanding how to train machine learning model in Spark.</p>

<p>Note: Shift + CMD + V to preview the markdown file</p>

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
|     |           |        |      |     |
| accuracy|   |   |   0.76   | 192     |
| macro avg | 0.71 | 0.67   |  0.68  | 192    |
| weighted avg | 0.77 | 0.78  | 0.77 | 192   |


## Rebalancing using RandomOverSampler
|     | precision | recall | f1-score | support |
|-----|-----------|--------|----------|---------|
| 0.0 | 0.44      |   0.92 | 0.60     | 59      |
| 1.0 | 0.94      |   0.52 |  0.67    | 141     |
|     |           |        |          |         |
| accuracy|       |        |   0.64   | 200     |
| macro avg | 0.69 | 0.72   |  0.63   | 200     |
| weighted avg |   0.79 | 0.64  | 0.65 | 200    |

<p>Although the overall accuracy of the model decreased. The recall of the model increases,
the rebalanced model has a higher recall rate which is less false negative rate.
More loan can be given to trustworthy customers.
</p>


## Rebalancing with SMOTE
|     | precision | recall | f1-score | support |
|-----|-----------|--------|----------|---------|
| 0.0 | 0.45      |   0.92 | 0.60     | 59      |
| 1.0 | 0.94      |   0.52 |  0.67    | 141     |
|     |           |        |          |         |
| accuracy|       |        |   0.64   | 200     |
| macro avg | 0.69 | 0.72   |  0.64   | 200     |
| weighted avg |   0.79 | 0.64  | 0.65 | 200    |

<p>The 2 rebalancing methods have similar outputs, SMOTE has a slight improvement on precision of 0 class.</p>