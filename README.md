# Heart disease prediction by using machine learning algorithms

## Table of contents
* Introduction
* Dataset Description
* Data Wrangling
* Exploratory Data Analysis
* Modeling
* Selecting the Best Model
* Conclusion

## Introduction

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs. In this project, We will predict whether or not an individual will suffer a possible heart disease by using statistical or machine learning models: logistic regression, decision tree, random forest, and gradient boosting, and also evaluate and compare these models.

## Dataset Description

The heart disease dataset contains 11 features:
* Age: age of the patient [years]
* Sex: sex of the patient [M: Male, F: Female]
* ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
* RestingBP: resting blood pressure [mm Hg]
* Cholesterol: serum cholesterol [mm/dl]
* FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
* RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
* MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
* ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
* Oldpeak: oldpeak = ST [Numeric value measured in depression]
* ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
* HeartDisease: output class [1: heart disease, 0: Normal]

## Data Wrangling

We checked the missing values and duplicated values. This dataset is quite clean. It does not have missing values and duplicated values. We also checked the categorical features.

## Exploratory Data Analysis

* The relation between Sex and HeartDisease

![image](https://user-images.githubusercontent.com/115129335/218932215-25af18e7-837c-4ef6-86ff-cfe8b158c94b.png)

The number of male who has heart disease is much lager than the number of male who does not have heart disease. While the number of female who has heart disease is much smaller than the number of female who does not have heart disease. 63.2% of male has heart disease, while the number of female who has heart disease is only 25.9%. Men is more likely to have heart disease than women.

* The relation among Sex, Age and HeartDisease

![image](https://user-images.githubusercontent.com/115129335/218933532-ba8b5823-114c-475f-80e4-02382c0c9e0d.png)

The number of male who has heart disease is much more than the number of female who has heart disease. Women who have heart disease are almost in the range of 55 to 65, while men's range is much wider.

* The relation between ChestPainType and HeartDisease

![image](https://user-images.githubusercontent.com/115129335/218933852-162ee32d-b4ac-4195-9fc1-59110788910f.png)

Silent heart attacks are much more than those who has symptoms. They are described as “silent” because when they occur, their symptoms lack the intensity of a classic heart attack, such as extreme chest pain and pressure; stabbing pain in the arm, neck, or jaw; sudden shortness of breath; sweating, and dizziness.

* The relation between FastingBS and HeartDisease

![image](https://user-images.githubusercontent.com/115129335/218934259-a20b0743-dfbc-4f37-a700-92aa0792e45b.png)

When fasting blood sugar > 120 mg/dl, the number of people who have heart disease are far more than the number of people who do not have heart disease.

* The relation between RestingECG and HeartDisease

![image](https://user-images.githubusercontent.com/115129335/218934581-4a8fee83-a7b1-4ba6-81b1-49da64c29b39.png)

When patients' resting electrocardiogram results show abnormal, the number of patients who have heart disease are much larger than the number of patients who do not have heart disease.

* The relation between MaxHR and HeartDisease

![image](https://user-images.githubusercontent.com/115129335/218935128-b3f33184-f8c0-4ebe-a35d-9cfbc513681d.png)

Maximum heart rate achieved of patients who have heart disease are lower than patients who do not have heart disease.

* The relation between ExerciseAngina and HeartDisease

![image](https://user-images.githubusercontent.com/115129335/218935336-36b42182-fe05-488d-a29c-28054cf676d6.png)

Patients who have heart disease have exercise-induced angina are more than those do not have angina. Very little patients who do not have heart disease but have angina.

* The relation between Oldpeak and HeartDisease

![image](https://user-images.githubusercontent.com/115129335/218935590-6d012573-49df-4ea6-9a7c-baa29eff15da.png)

The number of ST depression induced by exercise relative to rest spread more widely when patients have heart diseases, which means patients who have heart disease may have extreme high or low oldpeak number.

![image](https://user-images.githubusercontent.com/115129335/218935744-50cf7424-9a32-4715-982a-9da4674669f0.png)

The number of flat result in slope of the peak exercise ST segment is much larger than up and down result when patients have heart disease.

### Balanced and Imbalanced Classes

![image](https://user-images.githubusercontent.com/115129335/218935989-f47d353c-a9d3-49eb-9caf-164a8312b87f.png)

For a binary classification problem (two classes), the problem is called balanced if the number of elements of each class is about the same--in other words, each class would have a size that is about 50% of the total number of elements in the dataset. In this particular case study, one would say that this problem is slightly imbalanced, since the difference in the percentages is about 10%. When there is a imbalance among the classes in a binary classification problem, one usually refers to them as the minority class, and the majority class.

In practice, Imbalanced Classification Problems (ICP) are very common in situations where one is modeling events that are not common, and thus these events would be instances of the minority class. Examples of these problems include: study of diseases such as Cancer, study of processes such as fraud, and--in general--the study of rare anomalies within a system.

Since this particular case study deals with a slight imbalance among the classes, we might be able to build useful models with the given dataset--without introducing additional interventions.

## Modeling

### Building a Logistic Regression Model with Only Two Features
<img width="436" alt="Screen Shot 2023-02-14 at 9 02 52 PM" src="https://user-images.githubusercontent.com/115129335/218936624-b8483c08-f2ab-48a7-a434-badd8257a5fc.png">
<img width="436" alt="Screen Shot 2023-02-14 at 9 03 53 PM" src="https://user-images.githubusercontent.com/115129335/218936769-567c7f29-c868-4876-b584-410568e72900.png">

The model with only two features isn't very good! The accuracy on the training data is only 61%, and the accuracy on the testing data is barely better than random chance (54%)--where random chance if 50%. This isn't surprising since we are using only two features.

Moreover, notice that the classification report shows that the model performs poorly when trying to recognize inputs that belong to class "No", which is indicated by the poor values of precision, recall, and f1-score for class "No"--for the training set and test set.

Since the gap between training and testing accuracy is about 7%, one might say that the model is slightly over-fitting the data. Thus, in general, one says that a model is over-fitting (or just overfitting), when there is an important gap between its training performance and its test performance.

### Building a Logistic Regression Model with all Features

<img width="425" alt="Screen Shot 2023-02-14 at 9 10 40 PM" src="https://user-images.githubusercontent.com/115129335/218937931-30c052b8-5b24-4f76-a040-420fe6f2ab71.png">

<img width="425" alt="Screen Shot 2023-02-14 at 9 11 24 PM" src="https://user-images.githubusercontent.com/115129335/218938041-33223625-9419-4776-a8d2-26a632789b4d.png">

<img width="406" alt="Screen Shot 2023-02-14 at 9 12 04 PM" src="https://user-images.githubusercontent.com/115129335/218938131-2996f0bb-c9f5-44b7-969f-25feace7a92e.png">

From the chart we can see that while the overall accuracy was 85%, when we predict heart disease, 10% (10 of 100) of the time we are predicting a false positive, while the false negatives (predicting no disease when in fact there is heart disease) is about 20% (17 of 84). This information can be discussed with stakeholders to decide which is more important, reducing false positives or false negatives, assuming overall accuracy is acceptable.

Some comments for this model:

* The model's training accuracy (0.87) is pretty good (meaning, close to 1--or 100%), then one says there is only a small "bias" in the model.
* Since the model's test accuracy (0.85) is decently close to the training accuracy, we would say that there is a small "variance" between the training accuracy and the test accuracy. This is an indication that the model will "generalize well", which means that the model will be well-behaved when new data is presented to it.
* Overfitting in this model is diminished.

### Descion Tree

<img width="1103" alt="Screen Shot 2023-02-14 at 9 14 02 PM" src="https://user-images.githubusercontent.com/115129335/218938464-bc3ee084-0491-499f-bfdc-50030dbd0ba0.png">

<img width="466" alt="Screen Shot 2023-02-14 at 9 14 43 PM" src="https://user-images.githubusercontent.com/115129335/218938597-b42b31ce-cfb7-4e04-8595-cbde714e2c25.png">

Decision tree models does not improved compared to logistic regression model.You might have noticed an important fact about decision trees. Each time we run a given decision tree algorithm to make a prediction (such as whether customers will buy the Hidden Farm coffee) we will actually get a slightly different result. This might seem weird, but it has a simple explanation: machine learning algorithms are by definition stochastic, in that their output is at least partly determined by randomness.

To account for this variability and ensure that we get the most accurate prediction, we might want to actually make lots of decision trees, and get a value that captures the centre or average of the outputs of those trees. Luckily, there's a method for this, known as the Random Forest.

### Random Forest

<img width="466" alt="Screen Shot 2023-02-14 at 9 16 01 PM" src="https://user-images.githubusercontent.com/115129335/218938837-a5af48af-3d8d-43e4-92f9-477bea895726.png">

The popularity of random forest is primarily due to how well it performs in a multitude of data situations. It tends to handle highly correlated features well, where as a linear regression model would not. In this case study we demonstrate the performance ability even with only a few features and almost all of them being highly correlated with each other. Random Forest is also used as an efficient way to investigate the importance of a set of features with a large data set. Consider random forest to be one of your first choices when building a decision tree, especially for multiclass classifications.

Feature importance

<img width="695" alt="Screen Shot 2023-02-14 at 9 16 37 PM" src="https://user-images.githubusercontent.com/115129335/218939082-7ad592be-1774-49f8-819f-3e9f3b0b0cbd.png">

### Gradient boosting

<img width="456" alt="Screen Shot 2023-02-14 at 9 18 54 PM" src="https://user-images.githubusercontent.com/115129335/218939350-2cbd4855-c4a7-40fe-a2fa-51887b83b536.png">

Feature importance

<img width="695" alt="Screen Shot 2023-02-14 at 9 19 53 PM" src="https://user-images.githubusercontent.com/115129335/218939514-5693c5ad-d772-417c-9e8a-c3e9cd5d0319.png">

## Selecting the Best Model

### Build the model by dropping some less important features

* Random Forest

<img width="494" alt="Screen Shot 2023-02-14 at 9 22 03 PM" src="https://user-images.githubusercontent.com/115129335/218939864-d09be983-91bd-4912-a313-01d3c1d054ee.png">

* Gradient boosting

<img width="494" alt="Screen Shot 2023-02-14 at 9 22 49 PM" src="https://user-images.githubusercontent.com/115129335/218939972-d989bbf3-a05d-436d-ab7c-3089978ef247.png">

After dropping the less important features, the model does not improved. So we still choose the model with all features.

## Conclusion

How many patients will have heart disease?

56.94 % of patients will get heart disease.






