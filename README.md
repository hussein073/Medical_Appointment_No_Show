
# Module 3 Final Project
----------------
## Predict Medical Appointment No Shows
Why do 30% of patients miss their scheduled appointments?

### 2019.05.31

### Hussein Sajid & Anna Zubova 

[Jupyter Notebook](https://github.com/AnnaLara/mod_3_project_classification/blob/master/Index_Part_1.ipynb)

[Slides](https://docs.google.com/presentation/d/1f0TGXD4iM-Tzlq4XLiDlYQ1kuPP1vhl447IFY7emRrQ/edit?usp=sharing)

**Project Goal**

The goal of this project is to test our ability to gather information from a real-world database and use our knowledge of statistical analysis and classification to generate analytical insights, build and interpret a classification model that can be meaningful to the company/stakeholder.

**Classification Model Requirements**

The goal of our project is to query the database to get the data needed to perform a statistical analysis and build a classification models. In this classification model, we will need to apply different classifier on the different models to answer at least one of the questions from the dataset we choose. 

For each classification model, be sure to specify the training set score and accuracy score of each classifier. 

## Dataset: Medical Appointments No Show
Found [here (link to kaggle.com)](https://www.kaggle.com/joniarroba/noshowappointments)

### Context
A person makes a doctor appointment, receives all the instructions and no-show. Who to blame? 

### Content
300k medical appointments and its 15 variables (characteristics) of each. The most important one if the patient show-up or no-show the appointment.

### Preliminary EDA

We first did quick EDA on the variables/features in the dataset to get familiar with them and to identify any that might need extra cleaning.

Here are the list of features in the dataset:

| Data Columns             | Entries      |
|--------------------------|--------------|
| PatientId                | 110527       | 
| AppointmentID            | 110527       | 
| Gender                   | 110527       |
| ScheduledDay             | 110527       | 
| AppointmentDay           | 110527       | 
| Age                      | 110527       | 
| Neighbourhood            | 110527       | 
| Scholarship              | 110527       |
| Hipertension             | 110527       | 
| Diabetes                 | 110527       | 
| Alcoholism               | 110527       | 
| Handcap                  | 110527       | 
| SMS_received             | 110527       |
| No-show                  | 110527       | 

## Data Wrangling

Firstly, I am going to try and explore the data to check for missing values/erroneous entries and also comment on redundant features and add additional ones, if needed.

* Data types
* No null values
* Imbalanced classes of predicted variable: only 20% No-shows
* Transform variables into binary (one-hot-encoding, Gender(0/1), No-show(0/1))

It is immediately apparent that some of the column names have typos, so let us clear them up before continuing further, so that I don't have to use alternate spellings everytime I need a variable. For convenience, I am going to convert the AppointmentDate and ScheduleDate columns into datetime64 format. It is interesting to note that the time portions have vanished from the Appointment Data timedeltas, because all appointment times were set exactly to 00:00:00. We also create a new feature called HourOfTheDay, which will indicate the hour of the day at which the appointment was booked. This will be derived off AppointmentDate. It is clear that we do not have any NaNs anywhere in the data. However, we do have some impossible ages such as -2 and -1, and some pretty absurd ages such as 100 and beyond. I do admit that it is possible to live 113 years and celebrate living so long, and some people do live that long, but most people don't.

## Exploring The Data

Now we are all set to explore the different features of the data and determine how good a feature it is for prediction whether a patient is likely to show up at an appointment.
First we will check how the likelihood that a person will show up at an appointment changes with respect to Age, HourOfTheDay, AwaitingTime. Clearly, HourOfTheDay and AwaitingTime are not good predictors of Status, since the probability of showing up depends feebly on the HourOfTheDay and not at all on the AwaitingTime. The significantly stronger dependency is observed with respect to Age.

## Baseline Modelling

We are going to start modeling to learn more above our variables! For this first run we are going to use ALL our non-categorical variables.

Following are the list of classifier used to predict the model accuracy:

* Logistic Regression
* Random Forest
* Support Vector Classifier 
* Class Imbalacing
* Decision Tree

For us, among all the classifier Random Forest gives better model Accuracy (80 %)

| No-show    | Precision    | Recall    | F1-Score |
|------------|--------------|-----------|----------|
| No         | 0.80         | 0.99      | 0.89     |
| Yes        | 0.32         | 0.02      |0.03      | 


--------------------
