#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


import joblib

# In[2]:


import spacy


# In[3]:


df = pd.read_csv('learn-ai-bbc/BBC News Train.csv')


# In[5]:


df.head()


# In[7]:


df.Category.value_counts()


# In[10]:


df['Cat']=df['Category'].apply(lambda x: 1 if x=='sport' else 2 if x=='business' else 3 if x=='politics' else 4 if x=='entertainment' else 5 )


# In[11]:


df.head()


# In[12]:


df.drop(['Category'],axis=1)


# In[13]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(df.Text, df.Cat, test_size=0.3)


# In[16]:


X_train.shape


# In[ ]:





# In[17]:


from sklearn.feature_extraction.text import CountVectorizer


# In[20]:


v = CountVectorizer()

cv=v.fit_transform(X_train.values)

joblib.dump(v, 'pre_fitted_vectorizer.pkl')
# In[19]:


X_train


# In[21]:


cv


# In[22]:


cv.toarray()


# In[23]:


cv.shape


# In[29]:


v.get_feature_names_out()[1300:1329]


# In[30]:


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(cv, Y_train)


# In[31]:

joblib.dump(model, 'model1.pkl')


X_test_cv = v.transform(X_test)


# In[32]:


from sklearn.metrics import classification_report


# In[33]:


y_pred = model.predict(X_test_cv)


# In[34]:


print(classification_report(Y_test,y_pred))


# In[37]:


ch = ["I love to watch and play "]
ch_cv = v.transform(ch)


# In[38]:


model.predict(ch_cv)


# In[39]:


test = pd.read_csv('learn-ai-bbc/BBC News Test.csv')


# In[41]:


test_cv = v.transform(test.Text)


# 

# In[43]:


sol = model.predict(test_cv)


# In[44]:


sol


# In[45]:


len(sol)


# In[46]:


len(test)


# In[47]:


submit = pd.read_csv('learn-ai-bbc/BBC News Sample Solution.csv')


# In[63]:


submit['Category'] = sol


# In[64]:


submit


# In[65]:


submit['Category']=submit['Category'].apply(lambda x: 'sport' if x==1 else 'business' if x==2 else 'politics' if x==3 else 'entertainment' if x==4 else 'tech')


# In[66]:


submit


# In[67]:


submit.to_csv('submit.csv',index=False)


# In[ ]:




