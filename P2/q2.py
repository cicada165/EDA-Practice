import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.stats as stats
import statistics

data = pd.read_pickle('./Extrasensory_individual_data.p', compression='infer')
sensorData = pd.read_pickle('./Extrasensory_sensor_data.p', compression='infer')

## Case 1 Problem A code (and optional graph)

#Plot
plt.figure(figsize=(15,60))
plt.subplot(6,1,1)
plt.hist(data['age'])
plt.title('Histogram of Age')

plt.subplot(6,1,2)
plt.hist(data['gender'])
plt.title('Histogram of Gender')

plt.subplot(6,1,3)
plt.hist(data['system'])
plt.title('Histogram of System')

plt.subplot(6,1,4)
plt.hist(data['hours_in_study'])
plt.title('Histogram of Hours in Study')

plt.subplot(6,1,5)
plt.hist(data['perceived_average_screen_time'])
plt.title('Histogram of Perceived Average Screen Time')

plt.subplot(6,1,6)
plt.hist(data['actual_average_screen_time'])
plt.title('Histogram of Actual Average Screen Time')

# How are missing values represented for this feature
data.isnull().any()
#It looks like there are no null values
print(data['perceived_average_screen_time'].value_counts())
print(data['actual_average_screen_time'].value_counts())
#From the histogram, the missing value for perceived average screen time and actual average screen time is set to -1 

## Case 1 Problem B code and graph
tmp = data.query("actual_average_screen_time != -1")#Remove column that has missing value for actual_average_screen_time

#Histogram
# print(data_filtered['perceived_average_screen_time'].value_counts())
plt.hist(tmp ['actual_average_screen_time'])
plt.title('Histogram of Actual Average Screen Time')


#Does it have any outliers? If so, how many?
#Yes it has outliers, 10.78 and 11.63
tmp = data.query("actual_average_screen_time != -1")
q1 = tmp['actual_average_screen_time'].quantile(0.25)
q3 = tmp['actual_average_screen_time'].quantile(0.75)
iqr = q3-q1

low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

count = 0
outliers = []
for entry in tmp['actual_average_screen_time']:
    if entry > high or entry < low:
        count += 1
        outliers.append(entry)

print("There are " + str(count) + " outliers and they are: " + str(outliers) + '.')

# Is it skewed? If so, is it left skewed or right skewed? What’s the skewness?
skewness = tmp['actual_average_screen_time'].skew()
print(skewness)
#The data is highly skewed to the right with a skewness of 2.46. It is quite apparently by simply looking at the histogram previously.

## Case 1 Problem C code and graph
#Filling with mean

mean1 = data['perceived_average_screen_time'].mean()
mean2 = data['actual_average_screen_time'].mean()
data_meanFill = data.copy(deep=True)

for i in range(len(data_meanFill['perceived_average_screen_time'])):
    if data_meanFill['perceived_average_screen_time'][i] == -1:
        data_meanFill['perceived_average_screen_time'][i] = mean1

for i in range(len(data_meanFill['actual_average_screen_time'])):
    if data_meanFill['actual_average_screen_time'][i] == -1:
        data_meanFill['actual_average_screen_time'][i] = mean2

#Filling with median
median1 = data['perceived_average_screen_time'].median()
median2 = data['actual_average_screen_time'].median()
data_medianFill = data.copy(deep=True)

for i in range(len(data_medianFill['perceived_average_screen_time'])):
    if data_medianFill['perceived_average_screen_time'][i] == -1:
        data_medianFill['perceived_average_screen_time'][i] = median1

for i in range(len(data_medianFill['actual_average_screen_time'])):
    if data_medianFill['actual_average_screen_time'][i] == -1:
        data_medianFill['actual_average_screen_time'][i] = median2

#Filling with a random value
data_randFill = data.copy(deep=True)

for i in range(len(data_randFill['perceived_average_screen_time'])):
    if data_randFill['perceived_average_screen_time'][i] == -1:
        data_randFill['perceived_average_screen_time'][i] = random.randint(1,int(max(tmp['perceived_average_screen_time'])))

for i in range(len(data_randFill['actual_average_screen_time'])):
    if data_randFill['actual_average_screen_time'][i] == -1:
        data_randFill['actual_average_screen_time'][i] = random.randint(1,int(max(tmp['actual_average_screen_time'])))

# plt.figure(figsize=(15,20))
# plt.subplot(2,1,1)
# plt.hist(data['perceived_average_screen_time'], label = 'Original', alpha = 0.33, color = 'black')
# plt.hist(data_meanFill['perceived_average_screen_time'], label = 'Filling with Mean', alpha = 0.33, color = 'green')
# plt.hist(data_medianFill['perceived_average_screen_time'], label = 'Filling with Median', alpha = 0.33, color = 'blue')
# plt.hist(data_randFill['perceived_average_screen_time'], label = 'Filling with Random Value', alpha = 0.33, color = 'red')
# plt.legend(loc='upper left')
# plt.title('Histograms for Perceived Average Screen Time')

plt.figure(figsize=(15,20))
plt.subplot(2,1,2)
plt.hist(data['actual_average_screen_time'], label = 'Original', alpha = 0.33, color = 'black')
plt.hist(data_meanFill['actual_average_screen_time'], label = 'Filling with Mean', alpha = 0.33, color = 'green')
plt.hist(data_medianFill['actual_average_screen_time'], label = 'Filling with Median', alpha = 0.33, color = 'blue')
plt.hist(data_randFill['actual_average_screen_time'], label = 'Filling with Random Value', alpha = 0.33, color = 'red')
plt.legend(loc='upper left')
plt.title('Histograms for Actual Average Screen Time')

## Case 1 Problem D code and graph
np.random.seed(5) #Set a seed to ensure consistency in results

population = np.random.normal(3.75, 1.25, len(data))
d_meanFill = np.random.normal(statistics.mean(data_meanFill['actual_average_screen_time']), np.std(data_meanFill['actual_average_screen_time']), len(data_meanFill['actual_average_screen_time']))
d_medianFill = np.random.normal(statistics.mean(data_medianFill['actual_average_screen_time']), np.std(data_medianFill['actual_average_screen_time']), len(data_medianFill['actual_average_screen_time']))
d_randFill = np.random.normal(statistics.mean(data_randFill['actual_average_screen_time']), np.std(data_randFill['actual_average_screen_time']), len(data_randFill['actual_average_screen_time']))

t_meanFill, p_meanFill = stats.ttest_ind(population, d_meanFill)
t_medianFill, p_medianFill = stats.ttest_ind(population, d_medianFill)
t_randFill, p_randFill = stats.ttest_ind(population, d_randFill)

print("Mean-Fill vs population: "  + str(p_meanFill))
print("Median-Fill vs population: " + str(p_medianFill))
print("Random-Fill vs population: " + str(p_randFill))

## Case 2 Problem A code and histogram

tmp = data.query("perceived_average_screen_time != -1")#Remove column that has missing value for perceived_average_screen_time
plt.hist(tmp['perceived_average_screen_time'])

q1 = tmp['perceived_average_screen_time'].quantile(0.25)
q3 = tmp['perceived_average_screen_time'].quantile(0.75)
iqr = q3-q1

low = q1 - 1.5*iqr
high = q3 + 1.5*iqr

count = 0
outliers = []
for entry in tmp['perceived_average_screen_time']:
    if entry > high or entry < low:
        count += 1
        outliers.append(entry)

print("There are " + str(count) + " outliers and they are: " + str(outliers) + '.')

skewness = tmp['perceived_average_screen_time'].skew()
print('The skewness is: ' + str(skewness))

## Case 2 Problem B code
count = 0
for value in tmp['actual_average_screen_time']:
    if value >= statistics.mean(tmp['actual_average_screen_time']) + np.std(tmp['actual_average_screen_time']):
        count += 1


print("There are " + str(count) + ' intense phone users.')

## Case 2 Problem C code and graph
b_missing = []
b_power = []
for value in data['perceived_average_screen_time']:
    if value == -1:
        b_missing.append(0)
        b_power.append(1)
    elif value >= statistics.mean(tmp['perceived_average_screen_time']) + np.std(tmp['perceived_average_screen_time']):
        b_missing.append(1)
        b_power.append(0)
    else:
         b_missing.append(1)
         b_power.append(1)

chi_data = [b_missing, b_power]
stat, p, dof, expected = stats.chi2_contingency(chi_data)

print(p)


people_2 = dict()

# for key, value in sensorData.items():
#     count = 0
#     for index, row in value.iterrows():
#         if row['lf_measurements:battery_level'] < 0.15 and pd.isna(row['location:raw_latitude']):
#             count+=1
#             people_2[key] = count

for key, value in sensorData.items():
    count = 0
    b_count = 0
    for index, row in value.iterrows():
        if row['lf_measurements:battery_level'] < 0.15 and pd.isna(row['location:raw_latitude']):
            count+=1
            b_count+=1
            if (count/b_count) > 0.95:
                people_2[key] = count
        elif row['lf_measurements:battery_level'] < 0.15:
            b_count += 1
            
# for battery_percentage in entry['lf_measurements:battery_level']
# print(entry['lf_measurements:battery_level'])

print(people_2)
  
## Case 3 Problem B code and graph
subject = sensorData['F50235E0-DD67-4F2A-B00B-1F31ADA998B9']

subject_forward = subject.copy(deep=True)
subject_forward['location:raw_latitude'] = subject_forward['location:raw_latitude'].interpolate(method ='linear', limit_direction ='forward')
# prev = list(subject_forward['location:raw_latitude'])[0]
# for key, value in sensorData.items():
#     for index, row in value.iterrows():
#         if pd.isna(row['location:raw_latitude']):
#             row['location:raw_latitude'] = prev
#             prev = row['location:raw_latitude']
#         else:
#             prev = row['location:raw_latitude']


subject_backward = subject.copy(deep=True)
subject_backward['location:raw_latitude'] = subject_backward['location:raw_latitude'].interpolate(method ='linear', limit_direction ='backward')
# next = list(subject_forward['location:raw_latitude'])[1]
# index = 1
# for key, value in sensorData.items():
#     for index, row in value.iterrows():
#         if pd.isna(row['location:raw_latitude']):
#             row['location:raw_latitude'] = next
#             count+=1
#             next = list(subject_forward['location:raw_latitude'])[count]
#         else:
#             count+=1
#             next = list(subject_forward['location:raw_latitude'])[count]

subject_linear = subject.copy(deep=True)
subject_linear['location:raw_latitude'] = subject_linear['location:raw_latitude'].interpolate(method = 'linear')



plt.figure(figsize=(15,10))
plt.plot(subject['location:raw_latitude'], label = 'Original', alpha = 0.33, color = 'black')
plt.plot(subject_forward['location:raw_latitude'], label = 'Forward Filling', alpha = 0.33, color = 'red')
plt.plot(subject_backward['location:raw_latitude'], label = 'Backward Filling', alpha = 0.33, color = 'blue')
plt.plot(subject_linear['location:raw_latitude'], label = 'Linear Interpolation Filling', alpha = 0.33, color = 'green')
plt.legend(loc='upper left')
plt.title('Line Graphs Using Different Filling Methods')
