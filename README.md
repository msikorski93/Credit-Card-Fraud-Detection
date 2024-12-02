# Credit-Card-Fraud-Detection
![ alt text ](https://img.shields.io/badge/license-MIT-green?style=&logo=)
![ alt text ](https://img.shields.io/badge/-Jupyter-F37626?logo=Jupyter&logoColor=white)
![ alt text ](https://img.shields.io/badge/-NumPy-013243?logo=Numpy&logoColor=white)
![ alt text ](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=TensorFlow&logoColor=white)
![ alt text ](https://img.shields.io/badge/-Keras-D00000?logo=Keras&logoColor=white)
![ alt text ](https://img.shields.io/badge/-pandas-150458?logo=pandas&logoColor=white)
![ alt text ](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

### Part 1 Notebook
A binary classification task performed with commonly used `scikit-learn` algorithms. The dataset's target distribution was heavily imbalanced.

We need to keep in mind that accuracy score is a misleading evaluation metric in such case (normal transactions will be correctly classified and outnumbered, while fraud will not be). Traditional classifiers tend to favor the majority class, neglecting the minority class due to its lower representation. Each model performance was evaluated with F1 score.

|                      | Logisitic Regression | Naïve Bayes | K-Neighbors | LightGBM |
|---------------------:|---------------------:|------------:|------------:|---------:|
|      No Sampling     | 0.7344               | 0.1068      | 0.8152      | 0.2998   |
|        K-Means       | 0.9999               | 0.7744      | 0.9999      | 0.9999   |
|        ADASYN        | 0.8497               | 0.6093      | 0.9522      | 0.9370   |
|       SMOTE-ENN      | 0.9449               | 0.8993      | 0.9997      | 0.9994   |
| Random Undersampling | 0.9394               | 0.9011      | 0.9295      | 0.9375   |
|       Near Miss      | 0.9703               | 0.9819      | 0.9512      | 0.9736   |
|  Random Oversampling | 0.9447               | 0.9027      | 0.9996      | 0.9998   |
|   Balanced Bagging   | 0.7319               | 0.1076      | 0.8145      | 0.5682   |

While k-means achieves very high F1 scores (close to 1.0) across all models, this may indicate that the technique is overly biased towards achieving very high scores, which might not reflect real-world performance. The dramatic difference between the scores of Naïve Bayes (0.7744) and the others (close to 1.0) suggests that k-means may have produced imbalanced results for certain classifiers. When reviewing the evaluation results, we should favor the resampling technique that provides balanced results across all four models, rather than focusing solely on the overall highest score in the table. Thus, the optimal resampling based on the F1 scores is **SMOTE-ENN** with k-nearest neighbors.

### Part 2 Notebook
The same binary classification task was repeated with no sampling, SMOTE-ENN, and random undersampling. Convolution networks were trained on each of them with the same architecture in `TensorFlow`. Suprisingly, providing **no sampling** is the optimal strategy. It has the best balance overall, especially in terms of precision, recall, and log loss.

|                      | Log Loss | F1-Score | Precision | Recall | Accuracy |
|---------------------:|---------:|---------:|----------:|-------:|---------:|
|      No Sampling     | 0.0235   | 0.7956   | 0.8372    | 0.7579 | 0.9993   |
|       SMOTE-ENN      | 0.0750   | 0.5874   | 0.4398    | 0.8842 | 0.9979   |
| Random Undersampling | 1.4800   | 0.0725   | 0.0376    | 0.9579 | 0.9589   |
