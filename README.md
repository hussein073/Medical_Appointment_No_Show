
# Module 3 Final Project
----------------
## Predict Medical Appointment No Shows
Why do patients miss their scheduled appointments?

### Hussein Sajid & Anna Zubova 

[Jupyter Notebook Part 1](https://github.com/AnnaLara/mod_3_project_classification/blob/master/Index_Part_1.ipynb)

[Jupyter Notebook Part 2](https://github.com/AnnaLara/mod_3_project_classification/blob/master/index_part_2.ipynb)

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
We explored Kaggle dataset with data about medical appointments in Brazil during year 2016. The dataset contains data about approximately 110,000 appointments

## Part 1

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
## Part 2

### Feature engineering

I part 2 we did some feature engineering to see if we could predict better the no-show comparing to the models from part 1.

The features we added:

- How many days in advance the appointment was made
- Appointment month
- Appointment day of the week
- Number of prior appointments for each appointment
- Number of prior no-shows for each appointment

We ran several classification algorithms on the data, here is a comparison of different algorithms' evaluation metrics:

| **model**  |  **cv score** |**f1 0 / 1**| **precision 0 / 1**  |**recall 0 / 1**   |
|---|---|---|---|---|
| decision trees  |  0.75 | 0.76 / 0.41  | 0.89 / 0.30  |  0.66 / 0.66 |
| random forests  | 0.71  | 0.87 / 0.25  | 0.83 / 0.32  |  0.90 / 0.20 |
|  logistic regression (initial) | 0.69  | 0.84 / 0.33  |  0.85 / 0.32 |  0.84 / 0.34 |
|  logistic regression (dropped features) | 0.69 | 0.84 / 0.33  |  0.85 / 0.32 |  0.83 / 0.35 |

Decision trees was one of the most successful algorithms, but it is prone to overfitting, so we applied other algorithms too.

Logistic regression gave us much more reasonable recall than models from part 2 with a bit of loss in Cross Validation score.

After applying logistic regression for the first time, we had to revome variables `Scholarship`, `Diabetes`, `month_appointment_5` and `day_of_week_appointment_5` because of the large p-values. Here is the remaining variables with corresponding coefficients:
 
Gender	-0.123217
Age	-0.007556
Hipertension	-0.007325
Alcoholism	0.040789
Handcap	-0.146144
SMS_received	0.390198
days_in_advance	0.027498
month_appointment_6	-0.463698
day_of_week_appointment_1	-0.244623
day_of_week_appointment_2	-0.175504
day_of_week_appointment_3	-0.245840
day_of_week_appointment_4	-0.096902
number_of_previous_apptms	-0.078217
number_of_previous_noshows	0.601255

### Interpretation of coefficients

Continious variables:

- **age**: for every additional year log(odds of no-show) decreases by 0.007
- **number_of_previous_apptms**: for 1 additional number log(odds of no-show) decreases by 0.07
- **days_in_advance**: for every additional day log(odds of no-show) increases by 0.027
- **number_of_previous_noshows** for every additional previous no-show log(odds of no-show) increases by 0.6

Discreet variables:

The presence of feature increases/decreases log(odds of no-show) by coefficient value

Most significant features: `number_of_previous_noshows`, `month_appointment_6`, `SMS_received` 

### Simulation of change in most significant variables

Let's predict the probability of a no-show for a person with the following parameters:

- Gender:	Male
- Age:	30
- Hipertension:	0
- Alcoholism: 	0
- Handicap:	    0
- SMS_received:	0
- days_in_advance:	14
- Month_appointment_6:  	             0
- Day_of_week_appointment_1: 	0
- Day_of_week_appointment_2: 	1
- Day_of_week_appointment_3: 	0
- Day_of_week_appointment_4: 	0
- Number_of_previous_apptms: 	0
- number_of_previous_noshows:	0

We used out Logistic Regression model to predict the probability of a no-show of this imaginary person. This gave us our baseline probability: 46%

We will now try to increase the parameters for our 3 most important variables to see how the predicted probability will be changing.

1. Number of previous no-shows:  increase from 0 to 1 
Increase of probability from 46% to 61%

2. Number of previous no-shows:  increase from 1 to 2 
Increase of probability from 61% to 74%

3. Appointment in June:
Decrease of probability from  74% to 64%

3. Received SMS notification:
Increase of probability from 62% to 72%

It is counter-intuitive that recieving SMS will increase the probability of a no-show. The description of the dataset states that "SMS_received = 1 or more messages sent to the patient". That could mean that the value 0 indicates that more than 1 SMS was sent for those patient, which might explain the positive relationship between a no-show and SMS_recieved variable.
