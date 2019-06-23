# <a href='https://www.kaggle.com/c/titanic/overview'>kaggle_competition_titanic </a>
Supervised learning for binary classification using Logistic regression

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<img src='img/titanic.jpg'/>

## Introduction to dataset

__Training dataset__
<img src='img/stats/train_dataset.png'/>

__Testing dataset__
<img src='img/stats/test_dataset.png'/>

__Preparing and cleaning data for analysis:__

Given all `NaN` values have been replaced by a placeholder value of `-1000`, we get the following Kernel Density Plots for the numeric variables in the dataset.

<img src='img/stats/kde_nan_values.png' /><br/>

Percent of Null Values           |  Number of Unique Values
:-------------------------------:|:----------------------:
![](img/stats/percent_null.png)  |  ![](img/stats/num_non_null.png)

    We drop the rows with NaN values for the Embarked and the Fare column in the training dataset as they only have 0.15% and 0.07% of missing values.
    We drop the Cabin column completely as it has 77.5% of values missing and it does not make sense to impute this proportion of missing values.
    (We could use a isCabinPresent boolean variable but we don't to limit the number of features)
    The NaN values in the Age column are filled with imputed values.
    These imputed values are the age values randomly selected within one std dev of the mean of the age of the respective ticket class and sex from the tables given below.

Mean of Age segmented by Passenger Class and Sex  |  Std Dev of Age segmented by Passenger Class and Sex
:-------------------------------:|:----------------------:
![](img/stats/mean_of_age_by_pclass_sex.png)  |  ![](img/stats/std_of_age_by_pclass_sex.png)  |

## Statistical analysis

General statistics for the entire dataset (train and test)  |  Correlation Matrix of numerical columns in test dataset
:-------------------------------:|:----------------------:
![](img/stats/dataset_description.png)  |  ![](img/stats/correl_matrix.png)  |

    There is a negative correlation between Passenger Class and survival rate as expected as more Passengers from the better classes (Smaller the better) are expected to survive.
    This is also true for Fare, which is positively correlated with the survival rate.

__Boxplot of Age segmented by Passenger Class and Sex__
<img src='img/stats/boxplot_age_pclass_sex.png' />

Survival rate by Passenger Class and Sex  |  No. of Survivors(1) and Deceased(0) by Sex
:-------------------------------:|:----------------------:
![](img/stats/survival_rate_sex_pclass.png)  |  ![](img/stats/num_survived_deceased_sex.png)  |

    The survival rate and the number of passengers who survived was higher for females than males.

__Average fare of passengers by Sex and Class__
<img src='img/stats/avg_fare_by_pclass_sex.png' />

    Women in almost all Passenger Classes were paying on average a higher fare than men which also meant they had a higher chance of survival as Fare is positively correlated with survivability.

## Logistic Regression results

    For our logistic regression we use a linear logistic model with no polynomial features.

__The training dataset after cleaning and normalization.__
<img src='img/result/X_feat.png' />
   * Note that the categorical features `Sex` and `Embarked` have been converted into numerical features by the columns `isMale`, `isFemale`, `emS`, `emC` and `emQ`.

__Loss curve after 200000 iterations with an alpha(learning rate) of 0.019 and lambda(regularization factor) of 0.012`__
<img src='img/result/loss_overtime.png' />

__Final loss and accuracy on test data__
<img src='img/result/loss_accuracy_model.png' />

__Accuracy on test data from Kaggle using a linear logistic model__
<img src='img/submission01.png' />

### Acknowledgements
-   Kaggle dataset available from <https://www.kaggle.com/c/titanic/overview>
-   Utility.py utility function library created from another repository on custom Linear and Logistic regression created from scratch using numpy for calculation and matplotlib for plotting.
