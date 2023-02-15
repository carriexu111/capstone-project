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
