# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:46:27 2020

@author: anuju
"""

# In[1]:

import os
os.environ['JAVA_HOME'] = 'C:/Program Files/Java/jre1.8.0_241'
os.environ['PYSPARK_SUBMIT_ARGS'] = "--master local[2] pyspark-shell"

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
spark = SparkSession.builder.appName('k-means clustering').getOrCreate()


# Loading datafile "hack_data.txt"
dataset = spark.read.csv("c:/users/anuju/desktop/Internet_Of_Things/Assignments/hack_data.txt",header=True,inferSchema=True)
dataset.head()
dataset.show(10)
dataset.describe().show()

#UNDERSTANDING THE DATA

#plotting a histogram of the data
dataset.toPandas().hist(column=dataset.columns, bins=20, figsize=(7,7))

# Correlation Matrix plot
import seaborn as sns
import numpy as np
import matplotlib as plt

# Get Correlations
corr = dataset.toPandas()[dataset.columns].corr()

#plotting the heatmap
sns.heatmap(corr,annot=True)





# In[2]:

# ## Format the Data

#Converting Categorical Data

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
import pyspark.sql.functions as F

stage_string = StringIndexer().setInputCol("Location").setOutputCol("LocationIndex")
stage_one_hot = OneHotEncoder().setInputCol("LocationIndex").setOutputCol("LocationCoded")


ppl = Pipeline(stages = [stage_string , stage_one_hot])
df = ppl.fit(dataset).transform(dataset)
df.toPandas().to_csv('HackData_afterTransform.csv')

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

print(df.columns)

# Extracting Features without Location

Assembler_NoLocation = VectorAssembler(
  inputCols=['Session_Connection_Time',
             'Bytes Transferred',
             'Kali_Trace_Used',
             'Servers_Corrupted',
             'Pages_Corrupted',
             'WPM_Typing_Speed'],
    outputCol="features_NoLocation")

FeatureData_NoLocation= Assembler_NoLocation.transform(df)
FeatureData_NoLocation.toPandas().to_csv('HackDataFeatures_NoLocation.csv')
FeatureData_NoLocation.show(5)

# Extracting Features with Location after String Indexer

Assembler_LocationIndex = VectorAssembler(
  inputCols=['Session_Connection_Time',
             'Bytes Transferred',
             'Kali_Trace_Used',
             'Servers_Corrupted',
             'Pages_Corrupted',
             'WPM_Typing_Speed',
             'LocationIndex'],
    outputCol="features_LocationIndex")

FeatureData_LocationIndex= Assembler_LocationIndex.transform(df)
FeatureData_LocationIndex.toPandas().to_csv('HackDataFeatures_LocationIndex.csv')
FeatureData_LocationIndex.show(5)

# Extracting Features with Location after StringIndexer and  OneHotEncoder

Assembler_LocationCoded = VectorAssembler(
  inputCols=['Session_Connection_Time',
             'Bytes Transferred',
             'Kali_Trace_Used',
             'Servers_Corrupted',
             'Pages_Corrupted',
             'WPM_Typing_Speed',
             'LocationCoded'],
    outputCol="features_LocationCoded")

FeatureData_LocationCoded= Assembler_LocationCoded.transform(df)
FeatureData_LocationCoded.toPandas().to_csv('HackDataFeatures_LocationCoded.csv')
FeatureData_LocationCoded.show(5)



# In[3]:

#Scaling the data without Location

from pyspark.ml.feature import StandardScaler

scaler_NoLocation = StandardScaler(inputCol="features_NoLocation", outputCol="scaledFeatures_NoLocation", withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler

scalerModel_NoLocation = scaler_NoLocation.fit(FeatureData_NoLocation)

# Normalize each feature to have unit standard deviation.

FinalData_NoLocation = scalerModel_NoLocation.transform(FeatureData_NoLocation)

FinalData_NoLocation.toPandas().to_csv('HackDataFinal_NoLocation.csv')

###################################################################################################################


#Scaling the data with Location after String Indexer


scaler_LocationIndex = StandardScaler(inputCol="features_LocationIndex", outputCol="scaledFeatures_LocationIndex", withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler

scalerModel_LocationIndex = scaler_LocationIndex.fit(FeatureData_LocationIndex)

# Normalize each feature to have unit standard deviation.

FinalData_LocationIndex = scalerModel_LocationIndex.transform(FeatureData_LocationIndex)

FinalData_LocationIndex.toPandas().to_csv('HackDataFinal_LocationIndex.csv')


##################################################################################################################


#Scaling the data with Location after OneHotEncoder


scaler_LocationCoded = StandardScaler(inputCol="features_LocationCoded", outputCol="scaledFeatures_LocationCoded", withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler

scalerModel_LocationCoded = scaler_LocationCoded.fit(FeatureData_LocationCoded)

# Normalize each feature to have unit standard deviation.

FinalData_LocationCoded = scalerModel_LocationCoded.transform(FeatureData_LocationCoded)

FinalData_LocationCoded.toPandas().to_csv('HackDataFinal_LocationCoded.csv')


# In[4]:

# ## Train the Model and Evaluate

from pyspark.ml.clustering import KMeans

#Importing matplotlib to plot a graph

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

# Finding the Appropriate K Value by Elbow Method

# Without Location

wssse= []
KMin = 2
KMax = 34
KStep = 1
index = 0

for iLoop in range(KMin,KMax,KStep):
    kmeans = KMeans(featuresCol='scaledFeatures_NoLocation', k=iLoop)
    model = kmeans.fit(FinalData_NoLocation) 
    wssse.append(model.computeCost(FinalData_NoLocation))
    print("When k=" + str(iLoop) +" Within Set Sum of Squared Errors = " + str(wssse[index]))
    index += 1

    
fig, ax = plt.subplots(1,1, figsize =(8,8))
ax.plot(range(KMin,KMax, KStep),wssse[0:math.ceil(KMax/KStep)],'--bo')
ax.xaxis.set_major_locator(ticker.MultipleLocator(KStep))
ax.set_xlabel('k-Value',fontsize = 20)
ax.set_ylabel('WSSSE',fontsize = 20)
ax.grid()   
ax.set_title("The Elbow Method to find the k-value (NoLocation)",fontsize = 24)

# With LocationIndex

wssse= []
KMin = 2
KMax = 30
KStep = 1
index = 0

for iLoop in range(KMin,KMax,KStep):
    kmeans = KMeans(featuresCol='scaledFeatures_LocationIndex', k=iLoop)
    model = kmeans.fit(FinalData_LocationIndex) 
    wssse.append(model.computeCost(FinalData_LocationIndex))
    print("When k=" + str(iLoop) +" Within Set Sum of Squared Errors = " + str(wssse[index]))
    index += 1

    
fig, ax = plt.subplots(1,1, figsize =(8,8))
ax.plot(range(KMin,KMax, KStep),wssse[0:math.ceil(KMax/KStep)],'--bo')
ax.xaxis.set_major_locator(ticker.MultipleLocator(KStep))
ax.set_xlabel('k-Value',fontsize = 20)
ax.set_ylabel('WSSSE',fontsize = 20)
ax.grid()   
ax.set_title("The Elbow Method to find the k-value (With LocationIndex)",fontsize = 24)


# With LocationCoded

wssse= []
KMin = 2
KMax = 300
KStep = 10
index = 0

for iLoop in range(KMin,KMax,KStep):
    kmeans = KMeans(featuresCol='scaledFeatures_LocationCoded', k=iLoop)
    model = kmeans.fit(FinalData_LocationCoded) 
    wssse.append(model.computeCost(FinalData_LocationCoded))
    print("When k=" + str(iLoop) +" Within Set Sum of Squared Errors = " + str(wssse[index]))
    index += 1

    
fig, ax = plt.subplots(1,1, figsize =(8,8))
ax.plot(range(KMin,KMax, KStep),wssse[0:math.ceil(KMax/KStep)],'--bo')
ax.xaxis.set_major_locator(ticker.MultipleLocator(KStep))
ax.set_xlabel('k-Value',fontsize = 20)
ax.set_ylabel('WSSSE',fontsize = 20)
ax.grid()   
ax.set_title("The Elbow Method to find the k-value (With LocationCoded)",fontsize = 24)

#######################################################################################################################################

#Training the K-Means Model

#Without Location
    
Final_KMeans_NoLocation = KMeans(featuresCol='scaledFeatures_NoLocation', k=6)
Final_Model_NoLocation = Final_KMeans_NoLocation.fit(FinalData_NoLocation)

#Evaluation of Model

FinalModelWSSSE_NoLocation = Final_Model_NoLocation.computeCost(FinalData_NoLocation)
print ("Within Set Sum of Suared Errors = " + str(FinalModelWSSSE_NoLocation))


#With LocationIndex

Final_KMeans_LocationIndex = KMeans(featuresCol='scaledFeatures_LocationIndex', k=9)
Final_Model_LocationIndex = Final_KMeans_LocationIndex.fit(FinalData_LocationIndex)

#Evaluation of Model

FinalModelWSSSE_LocationIndex = Final_Model_LocationIndex.computeCost(FinalData_LocationIndex)
print ("Within Set Sum of Suared Errors = " + str(FinalModelWSSSE_LocationIndex))


#With LocationCoded

Final_KMeans_LocationCoded = KMeans(featuresCol='scaledFeatures_LocationCoded', k=192)
Final_Model_LocationCoded = Final_KMeans_LocationCoded.fit(FinalData_LocationCoded)

#Evaluation of Model

FinalModelWSSSE_LocationCoded = Final_Model_LocationCoded.computeCost(FinalData_LocationCoded)
print ("Within Set Sum of Suared Errors = " + str(FinalModelWSSSE_LocationCoded))


# In[5]:


# Shows the result.

#Without Location

centers = Final_Model_NoLocation.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# Predict the label of each hacking attempt
    
Final_Model_NoLocation.transform(FinalData_NoLocation).select('prediction').show(10)


#With LocationIndex

centers = Final_Model_LocationIndex.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# Predict the label of each hacking attempt
    
Final_Model_LocationIndex.transform(FinalData_LocationIndex).select('prediction').show(10)


#With LocationCoded

centers = Final_Model_LocationCoded.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# Predict the label of each hacking attempt
    
Final_Model_LocationCoded.transform(FinalData_LocationCoded).select('prediction').show(10)


# In[6]:

#formingClusters

#Without Location

clusters_NoLocation = Final_Model_NoLocation.transform(FinalData_NoLocation).select('*')
clusters_NoLocation.groupBy("prediction").count().orderBy(F.desc("count")).show()
clusters_NoLocation.show()
clusters_NoLocation_pd = clusters_NoLocation.toPandas()
clusters_NoLocation_pd.to_csv("Clusters_NoLocation.csv")

#With LocationIndex

clusters_LocationIndex = Final_Model_LocationIndex.transform(FinalData_LocationIndex).select('*')
clusters_LocationIndex.groupBy("prediction").count().orderBy(F.desc("count")).show()
clusters_LocationIndex.show()
clusters_LocationIndex_pd = clusters_LocationIndex.toPandas()
clusters_LocationIndex_pd.to_csv("clusters_LocationIndex.csv")

#With LocationCoded

clusters_LocationCoded = Final_Model_LocationCoded.transform(FinalData_LocationCoded).select('*')
clusters_LocationCoded.groupBy("prediction").count().orderBy(F.desc("count")).show()
clusters_LocationCoded.show()
clusters_LocationCoded_pd = clusters_LocationCoded.toPandas()
clusters_LocationCoded_pd.to_csv("clusters_LocationCoded.csv")

# Plotting Clusters


# Pairwise Scatterplot

dataScatterPlot_NoLocation = clusters_NoLocation_pd[['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used','Servers_Corrupted', 'Pages_Corrupted','WPM_Typing_Speed','prediction']]
Variables_NoLocation =  clusters_NoLocation_pd[['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used','Servers_Corrupted', 'Pages_Corrupted','WPM_Typing_Speed']]
g_NoLocation = sns.pairplot( dataScatterPlot_NoLocation, vars = Variables_NoLocation,  hue = "prediction", diag_kind = 'kde',palette= "husl", plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'})


dataScatterPlot_LocationIndex = clusters_LocationIndex_pd[['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used','Servers_Corrupted', 'Pages_Corrupted','WPM_Typing_Speed','LocationIndex','prediction']]
Variables_LocationIndex =  clusters_LocationIndex_pd[['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used','Servers_Corrupted', 'Pages_Corrupted','WPM_Typing_Speed','LocationIndex']]
g_LocationIndex = sns.pairplot( dataScatterPlot_LocationIndex, vars = Variables_LocationIndex,  hue = "prediction", diag_kind = 'kde',palette= "husl", plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'})


dataScatterPlot_LocationCoded = clusters_LocationCoded_pd[['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used','Servers_Corrupted', 'Pages_Corrupted','WPM_Typing_Speed','prediction']]
Variables_LocationCoded =  clusters_LocationCoded_pd[['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used','Servers_Corrupted', 'Pages_Corrupted','WPM_Typing_Speed']]
g_LocationCoded = sns.pairplot( dataScatterPlot_LocationCoded, vars =  Variables_LocationCoded, hue = "prediction", diag_kind = 'kde',palette= "husl", plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'})







