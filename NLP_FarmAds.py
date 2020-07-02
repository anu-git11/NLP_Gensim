# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:48:24 2020

@author: anujuneja
"""

import os
os.environ['JAVA_HOME'] = 'C:/Program Files/Java/jre1.8.0_241'
os.environ['PYSPARK_SUBMIT_ARGS'] = "--master local[2] pyspark-shell"


# In[1]:

# Read Text Data


from pyspark.sql import SparkSession
import pyspark.sql.functions as PySparkFunc

spark = SparkSession.builder.appName('text mining').getOrCreate()
data = spark.read.csv("C:/Users/anuju/desktop/Internet_Of_Things/Assignments/Assignment3/farm-ads.csv", inferSchema=True, sep=',')
data.show(5)

data = data.withColumnRenamed('_c0','class').withColumnRenamed('_c1','text')


# Removing the extra space in front of every ad 

data = data.withColumn('text', (PySparkFunc.trim (PySparkFunc.col("text"))))

data.show(5,truncate=False)



# In[2]:

# Count number of Words in each Text


from pyspark.sql.functions import length
data = data.withColumn('length', length(data['text']))
data.show()


# In[3]:

# Compare the length difference between -1(not relevant) and 1(relevant)


data.groupby('class').mean().show()


# In[4]:

# Treat TF-IDF features for each text
# TF: Term Frequency
# IDF: Inverse Document Frequency


from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer, IDF, VectorAssembler

# Tokenizer is used to split the text to words

tokenizer = Tokenizer(inputCol="text", outputCol="token_text")

# StopWordsRemover is used to filter out the stop words i.e. the commonly used words

stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')

#Word Term Frequency (TF)

count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')

# Compute the Inverse Document Frequency(IDF) for a given collection of documents

idf = IDF(inputCol="c_vec", outputCol="tf_idf")

# Transforming the class column to labels 0 for relevant ads and 1 for non relevant ads 

String_Coded = StringIndexer(inputCol='class',outputCol='label')


final_feature = VectorAssembler(inputCols=['tf_idf', 'length'],outputCol='features')

from pyspark.ml import Pipeline

data_prep_pipe = Pipeline(stages=[String_Coded, tokenizer,stopremove,count_vec,idf,final_feature])
clean_data = data_prep_pipe.fit(data).transform(data)

clean_data.show()

## Selecting the first row of the data to check the elements

clean_data.take(1)

## Selecting the last column i.e. the Features column from the first row to examine the Tf-IDf Matrix

clean_data.take(1)[0][-1]


###----------------------------------------------------------------------------------------------------------###

## Calculating the Density and Sparsity of the Tf-IDf Feature Matrix

from numpy import count_nonzero

sparsity = (1.0 - count_nonzero(clean_data.take(1)[0][-1]) / clean_data.take(1)[0][-1].size)*100
print ("Sparsity is" ,round( sparsity,2),"%")

density = (count_nonzero(clean_data.take(1)[0][-1]) / clean_data.take(1)[0][-1].size)*100
print ("Density is" , round(density,2),"%")


# In[5]:

# ##Calculating Word Frequencies:

import matplotlib.pyplot as plt
from collections import OrderedDict
from wordcloud import WordCloud
from pyspark.sql.functions import col


counts = clean_data.where(col("label") == 1).select(PySparkFunc.explode('token_text').alias('col')).groupBy('col').count().collect()
FARMcounts = clean_data.where(col("label") == 0).select(PySparkFunc.explode('token_text').alias('col')).groupBy('col').count().collect()

WordFreqDict = {row['col']: row['count'] for row in counts}
FARMWordFreqDict = {row['col']: row['count'] for row in FARMcounts}


# Let us sort the freq dictionary
SortedWordFreqDict = OrderedDict(sorted(WordFreqDict.items(), key=lambda x: x[1], reverse=True)[:75])
FARMSortedWordFreqDict = OrderedDict(sorted(FARMWordFreqDict.items(), key=lambda x: x[1], reverse=True)[:75])

# Generate text for building WordCloud
WordCloudText = " ".join([(k + " ")*v for k,v in SortedWordFreqDict.items()])
FARMWordCloudText = " ".join([(k + " ")*v for k,v in FARMSortedWordFreqDict.items()])

# Generate a word cloud image
wordcloud = WordCloud(collocations=False, max_font_size=300, 
                      colormap='gist_heat', background_color='white', 
                      width=1920, height=1200 ).generate(WordCloudText)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


FARMwordcloud = WordCloud(collocations=False, max_font_size=300, 
                          colormap='gist_heat', background_color='white', 
                          width=1920, height=1200 ).generate(FARMWordCloudText)
plt.figure()
plt.imshow(FARMwordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()




# In[6]: 

# ## Split data into training and test datasets

training, test = clean_data.randomSplit([0.6, 0.4], seed=12345)

# Build Logistic Regression Model

from pyspark.ml.classification import LogisticRegression

log_reg = LogisticRegression(featuresCol='features', labelCol='label')

model = log_reg.fit(training)

results = model.transform(test)

results.select('label','prediction').show(20)


# In[7]:

# #### Confusion Matrix

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


y_true = results.select("label")
y_true = y_true.toPandas()

y_pred = results.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)

print(cnf_matrix)

print("Prediction Accuracy is ", (cnf_matrix[0,0]+cnf_matrix[1,1])/sum(sum(cnf_matrix)) )

# PLOT CONFUSION MATRIX

fig, ax = plt.subplots(1)
ax.xaxis.set_label_position('top') 

ax.xaxis.tick_top()
sns.heatmap(cnf_matrix, annot=True, fmt='d', ax=ax)

plt.xlabel("PREDICTED")
plt.ylabel("ACTUAL")



# In[8]:

# GETTING THE CLASSIFICATION REPORT

target_names = ['Class 0' , 'Class 1']

print ('Classification Report:')

print(classification_report(y_true, y_pred, target_names = target_names))



# In[9]:

# Model Evaluation

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Select (prediction, true label) and compute test error

evaluator = MulticlassClassificationEvaluator (labelCol="label", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(results)

print ("Test Accuracy = %g" %accuracy)

print("Test Error = %g" % (1.0 - accuracy))



