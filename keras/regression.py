#!/usr/bin/env python
# coding: utf-8

# <a href="https://cognitiveclass.ai"><img src = "https://ibm.box.com/shared/static/9gegpsmnsoo25ikkbl4qzlvlyjbgxs5x.png" width = 400> </a>
# 
# <h1 align=center><font size = 5>Regression Models with Keras</font></h1>

# ## Introduction

# As we discussed in the videos, despite the popularity of more powerful libraries such as PyToch and TensorFlow, they are not easy to use and have a steep learning curve. So, for people who are just starting to learn deep learning, there is no better library to use other than the keras library. 
# 
# Keras is a high-level API for building deep learning models. It has gained favor for its ease of use and syntactic simplicity facilitating fast development. As you will see in this lab and the other labs in this course, building a very complex deep learning network can be achieved with Keras with only few lines of code. You will appreciate Keras even more, once you learn how to build deep models using PyTorch and TensorFlow in the other courses.
# 
# So, in this lab, you will learn how to use the Keras library to build a regression model.

# ## Table of Contents
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <font size = 3> 
# 1. <a href="#item1">Download and Clean Dataset</a>  
# 2. <a href="#item2">Import Keras</a>  
# 3. <a href="#item3">Build a Neural Network</a>  
# 4. <a href="#item4">Train and Test the Network</a>  
# </font>
# </div>

# <a id="item1"></a>

# ## Download and Clean Dataset

# Let's start by importing the <em>pandas</em> and the Numpy libraries.

# In[ ]:


import pandas as pd
import numpy as np


# We will be playing around with the same dataset that we used in the videos.
# 
# <strong>The dataset is about the compressive strength of different samples of concrete based on the volumes of the different materials that were used to make them. Ingredients include:</strong>
# 
# <strong>1. Cement</strong>
# 
# <strong>2. Blast Furnace Slag</strong>
# 
# <strong>3. Fly Ash</strong>
# 
# <strong>4. Water</strong>
# 
# <strong>5. Superplasticizer</strong>
# 
# <strong>6. Coarse Aggregate</strong>
# 
# <strong>7. Fine Aggregate</strong>

# Let's download the data and read it into a <em>pandas</em> dataframe.

# In[ ]:


concrete_data = pd.read_csv('https://ibm.box.com/shared/static/svl8tu7cmod6tizo6rk0ke4sbuhtpdfx.csv')
concrete_data.head()


# So the first concrete sample has 540 cubic meter of cement, 0 cubic meter of blast furnace slag, 0 cubic meter of fly ash, 162 cubic meter of water, 2.5 cubic meter of superplaticizer, 1040 cubic meter of coarse aggregate, 676 cubic meter of fine aggregate. Such a concrete mix which is 28 days old, has a compressive strength of 79.99 MPa. 

# #### Let's check how many data points we have.

# In[ ]:


concrete_data.shape


# So, there are approximately 1000 samples to train our model on. Because of the few samples, we have to be careful not to overfit the training data.

# Let's check the dataset for any missing values.

# In[ ]:


concrete_data.describe()


# In[ ]:


concrete_data.isnull().sum()


# The data looks very clean and is ready to be used to build our model.

# #### Split data into predictors and target

# In[ ]:


concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column


# <a id="item2"></a>

# Let's do a quick sanity check of the predictors and the target dataframes.

# In[ ]:


predictors.head()


# In[ ]:


target.head()


# Finally, the last step is to normalize the data by substracting the mean and dividing by the standard deviation.

# In[ ]:


predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()


# In[ ]:


n_cols = predictors_norm.shape[1] # number of predictors


# <a id="item1"></a>

# ## Import Keras

# Recall from the videos that Keras normally runs on top of a low-level library such as TensorFlow. This means that to be able to use the Keras library, you will have to install TensorFlow first and when you import the Keras library, it will be explicitly displayed what backend was used to install the Keras library. In CC Labs, we used TensorFlow as the backend to install Keras, so it should clearly print that when we import Keras.

# #### Let's go ahead and import the Keras library

# In[ ]:


import tensorflow.keras


# As you can see, the TensorFlow backend was used to install the Keras library.

# Let's import the rest of the packages from the Keras library that we will need to build our regressoin model.

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ## Build a Neural Network

# Let's define a function that defines our regression model for us so that we can conveniently call it to create our model.

# In[ ]:


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# <a id="item4"></a>

# ## Train and Test the Network

# In[ ]:


# build the model
model = regression_model()

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)


# <strong>You can refer to this [link](https://keras.io/models/sequential/) to learn about other functions that you can use for prediction or evaluation.</strong>

# Feel free to vary the following and note what impact each change has on the model's performance:
# 
# 1. Increase or decreate number of neurons in hidden layers
# 2. Add more hidden layers
# 3. Increase number of epochs

# ### Thank you for completing this lab!
# 
# This notebook was created by [Alex Aklson](https://www.linkedin.com/in/aklson/). I hope you found this lab interesting and educational. Feel free to contact me if you have any questions!

# This notebook is part of a course on **edX** called *Deep Learning Fundamentals with Keras*. If you accessed this notebook outside the course, you can take this course online by clicking [here](http://cocl.us/DL0101EN_edX_Week3_LAB1).

# <hr>
# 
# Copyright &copy; 2018 [IBM Developer Skills Network](https://cognitiveclass.ai/?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).
