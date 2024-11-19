# Credit-Card-Fraud-Detection
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
