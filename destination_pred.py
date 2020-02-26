# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:50:13 2019

@author: zoed0

remember subplots
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from sklearn.cluster import DBSCAN
from geopy.distance import vincenty
from scipy.stats import circmean,circvar

#step1 read in xlsx,convert departure time into T, combine dep T,lat,long \
# with arrival lat,long
path = 'C:/Users/Administrator/Desktop/projects/intern/naive_bayes_destination_prediction/'
dat = pd.read_csv(path + 'parking_data.csv',
                  parse_dates= ['arvT','levT'])
dat2 = dat[['longitude','latitude','arvT','levT']]
dat3 = dat2.drop(dat2.index[0]).reset_index(drop=True)
dat_combine = pd.concat([dat2,dat3],axis=1)
dat_combine.columns = ['leave_long','leave_lat','arrive_time_nouse',\
                       'leave_time','arrive_long','arrive_lat','arrive_time','levT_nouse']
dat_combine = dat_combine.iloc[:,[0,1,3,4,5,6]]
dat_combine.arrive_time = pd.to_datetime(dat_combine.arrive_time)
dat_combine.leave_time = pd.to_datetime(dat_combine.leave_time)
judge_r = ((dat_combine.arrive_time - dat_combine.leave_time).dt.seconds/3600) 
dat_combine['duration'] = judge_r
dat_combine = dat_combine[(judge_r >0) & (judge_r <10)]
dat_combine['leave_hour'] = map(lambda a: a.hour,dat_combine['leave_time'])
dat_combine['leave_tz'] = 0
dat_combine.loc[dat_combine['leave_hour'].isin([0,1,2,3]),'leave_tz'] = 0
dat_combine.loc[dat_combine['leave_hour'].isin([4,5,6,7]),'leave_tz'] = 1
dat_combine.loc[dat_combine['leave_hour'].isin([8,9,10,11]),'leave_tz'] = 2
dat_combine.loc[dat_combine['leave_hour'].isin([12,13,14,15]),'leave_tz'] = 3
dat_combine.loc[dat_combine['leave_hour'].isin([16,17,18,19]),'leave_tz'] = 4
dat_combine.loc[dat_combine['leave_hour'].isin([20,21,22,23]),'leave_tz'] = 5

pivot_hour = dat_combine.groupby('leave_hour')[['leave_time']].agg('count')
plt.plot(pivot_hour)  #Basically normal dist
plt.xlabel("time in 24h")
plt.ylabel("count")
plt.show()

#step2 
#distance=(arrive - leave)
leave_tuple = [tuple(x) for x in dat_combine[['leave_lat','leave_long']].to_numpy()]
arrive_tuple =  [tuple(x) for x in dat_combine[['arrive_lat','arrive_long']].to_numpy()]
dat_combine['distance'] = map(lambda a,b:vincenty(a,b).miles*1609.344,\
           leave_tuple,arrive_tuple)
dat_combine['avg_speed'] = dat_combine['distance']/dat_combine['duration']*0.001
#delete incorrect records
dat_combine = dat_combine[dat_combine['distance'] > 100]
dat_combine = dat_combine[dat_combine['avg_speed'] < 200]       
dat_combine[~((dat_combine['avg_speed'] > 100) & (dat_combine['duration']<0.02))]

#cluster arrival lat,long within 500m, label them as the same arrival sites (y_i)
arrive_site_np = dat_combine[['arrive_long','arrive_lat']].to_numpy()
model = DBSCAN(eps=0.01,min_samples=1).fit(\
              dat_combine[['arrive_long','arrive_lat']])
dat_combine['y_i'] = model.fit_predict(arrive_site_np) #given distance within 500m, cluster into 103 clusters
dat_tt = dat_combine[['leave_hour','leave_long','leave_lat','y_i']]

#scatter plot of the spread out of arrival sites
plt.scatter(dat_combine[['arrive_long','arrive_lat']]['arrive_long'],\
            dat_combine[['arrive_long','arrive_lat']]['arrive_lat'],\
            c=dat_combine['y_i'])

#split train and test set, train: test = 9:1
len_dat = dat_tt.shape[0]
train_idx = random.sample(dat_tt.index.to_list(),int(len_dat*0.9))
train = dat_tt.loc[train_idx]
test = dat_tt.drop(train_idx)

#step3 calculate miu_i for T,lat,long
#plt.hist(dat_f['leave_long'])
#plt.hist(dat_f['leave_lat'])
plt.hist(train['leave_hour'])
plt.show()
#miu_lat = dat_f['leave_lat'].mean()
#miu_long = dat_f['leave_long'].mean()
###################
#try use scipy.stat to calculate mean and var



prob_table = []
for i in np.sort(train['y_i'].unique()):
    prob_T_yi = [0] * 6 
    data_list = train[train['y_i'] == i].leave_tz
    dat_mean = circmean(data_list,5,0)
    dat_var = circvar(data_list,5,0)
    #print(dat_mean,dat_var)
    if (abs(dat_var) <= 0.01):
        hour = train[train.y_i == i].iloc[0,0]
        prob_T_yi[hour] = 1
        #print(i,data_list,str(dat_var == 0))
    elif (dat_var > 0):
        for hour in range(6):
            if (abs(hour-dat_mean) <= 3):
                distance = abs(hour-dat_mean)
            else:
                distance = 6 - abs(hour-dat_mean)
            prob_T_yi[hour] = 1/(math.sqrt(2*math.pi*dat_var))*math.exp(-distance**2/(2*abs(dat_var)))
            #print(i,dat_mean,dat_var,distance,prob_T_yi)
    else:        
        print(dat_var,str('############'))
    prob_table.append(prob_T_yi) 
prob_df = pd.DataFrame(prob_table)

#step4 p(T|Y=y_i) ~ N(miu_i,sigma_i)
#step5 calculate p(y_i) = count(y_i)/sum(count(y_j))
count_yi = train.groupby('y_i')[['leave_tz']].agg('count')
p_yi = count_yi/count_yi.sum()
p_yi = p_yi.reset_index(drop=True)


#step6 Bayes rule for p(y_i|T,lat,long)
prob_mul = prob_df.mul(p_yi['leave_tz'],axis=0)
prob_final = prob_mul.div(prob_mul.sum())

#step7 sort p and give top1, top3 recommendation
#prd_top1 = {}
#for i in range(prob_final.shape[1]):
#    prd_top1[i] = prob_final.nlargest(n=1,columns=i).index.tolist()
prd_top3 = {}
for i in range(prob_final.shape[1]):
    prd_top3[i] = prob_final.nlargest(n=3,columns=i).index.tolist()

#step 8 calculate precison
#print(test)    
#for i in range(24):
#    for index, row in test.iterrows():
#        if (row.leave_hour == i):
#            #print(prd_top1[i])
#           test.loc[index,'esti_des'] = int(prd_top1[i][0])
for i in range(24):
    for index, row in test.iterrows():
        if (row.leave_tz == i):
            test.loc[index,'top1'],test.loc[index,'top2'],test.loc[index,'top3'] = \
                    int(prd_top3[i][0]),int(prd_top3[i][1]),int(prd_top3[i][2]) 
count = 0
for index, row in test.iterrows():
    if (row.y_i in [row.top1,row.top2,row.top3]):
        count += 1        
precision = float(count)/(test.shape[0])
print(test)
print('In 1000 m, the top3 precision = ' + str(precision))
