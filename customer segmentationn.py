#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[3]:


data = pd.read_csv(r"C:\Users\sampada\OneDrive\Desktop\Mall_Customers.csv")
data.head()


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.dtypes


# In[7]:


data.isnull().sum()


# In[8]:


data.drop(["CustomerID"],axis=1 , inplace = True)


# In[9]:


data.head()


# In[10]:


plt.figure(1, figsize=(15, 6))
n = 0

for x in ["Age", "Annual Income (k$)", "Spending Score (1-100)"]:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace = 0.5 , wspace =0.5)
    sns.distplot(data[x], bins=20)
    plt.title('Distplot of {}'.format(x))

plt.show()


# In[11]:


plt.figure(figsize=(15,5))
sns.countplot(y='Gender' ,data = data)
plt.show()


# In[12]:


plt.figure(1, figsize=(15, 7))
n = 0

for cols in ["Age", "Annual Income (k$)", "Spending Score (1-100)"]:
    n += 1
    plt.subplot(1, 3, n)
    sns.set(style="whitegrid")
    plt.subplots_adjust(hspace = 0.5 , wspace =0.5)
    sns.violinplot(x= cols , y= 'Gender', data = data)
    plt.ylabel('Gender' if n==1 else '')
    plt.title('Violin plot')

plt.show()


# In[13]:


age_18_30 = data.Age[(data.Age>= 18) & (data.Age<=30)]
age_31_40 = data.Age[(data.Age>= 31) & (data.Age<=40)]
age_41_50 = data.Age[(data.Age>= 41) & (data.Age<=50)]
age_51_60 = data.Age[(data.Age>= 51) & (data.Age<=60)]
age_61above = data.Age[(data.Age >=61)]

agex=["18-30" , "31-40","41-50","51-60", "61+"]
agey=[len(age_18_30.values),len(age_31_40.values),len(age_41_50.values),len(age_51_60.values),len(age_61above.values)]

plt.figure(figsize=(15,7))
sns.barplot(x=agex,y=agey,palette ="mako")
plt.title("Number of Customers and Ages")
plt.xlabel("Age")
plt.ylabel("Numbver of customers")
plt.show()


# In[14]:


sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=data)

plt.show()


# In[15]:


ss_1_20 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 1) & (data["Spending Score (1-100)"] <= 20)]
ss_21_40 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 21) & (data["Spending Score (1-100)"] <= 40)]
ss_41_60 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 41) & (data["Spending Score (1-100)"] <= 60)]
ss_61_80 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 61) & (data["Spending Score (1-100)"] <= 80)]
ss_61_80 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 81) & (data["Spending Score (1-100)"] <= 100)]
ssx =["1-20", "21-40", "41-60", "61-80", "81-100"]
ssy = [len("ss_1_20.values"), len("ss_21_40.values"),len("ss_41_60.values"),len("ss_61_80.values"), len("ss_81_100.values")]

plt.figure(figsize=(15,6))
sns.barplot(x=ssx , y=ssy , palette = "rocket")
plt.title("Spending Scores")
plt.xlabel("Score")
plt.ylabel("Number of customers with scores:")
plt.show()



# In[16]:


ai1= data["Annual Income (k$)"][(data['Annual Income (k$)']>=0) &(data['Annual Income (k$)']<=30)] 
ai2= data["Annual Income (k$)"][(data['Annual Income (k$)']>=31) & (data['Annual Income (k$)']<=60)] 
ai3= data["Annual Income (k$)"][(data['Annual Income (k$)']>=61) & (data['Annual Income (k$)']<=90)] 
ai4= data["Annual Income (k$)"][(data['Annual Income (k$)']>=91) & (data['Annual Income (k$)']<=120)] 
ai5= data["Annual Income (k$)"][(data['Annual Income (k$)']>=121) & (data['Annual Income (k$)']<=150)] 

aix = [ "$ 0 - 30,000", "$ 30,3001 - 60,000", "$ 60,100 - 90,000", "$ 90,001 - 120,000", "$ 120,001 - 150,000"]
aiy = [ len(ai1.values),len(ai2.values),len(ai3.values),len(ai4.values),len(ai5.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=aix , y=aiy , palette = "Spectral")
plt.title("Annual Incomes")
plt.xlabel("Income")
plt.ylabel("Number of customers")
plt.show()



# In[17]:


X1 = data.loc[:, ["Age","Spending Score (1-100)"]].values

wcss = []

for k in range (1,11):
    kmeans = KMeans(n_clusters = k, init= "k-means++")
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color = "red", marker ="8")
plt.xlabel("K Values")
plt.ylabel("WCSS")
plt.show()


# In[18]:


kmeans = KMeans(n_clusters = 4)

label = kmeans.fit_predict(X1)

print(label)


# In[19]:


print(kmeans.cluster_centers_)


# In[20]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(X1)


labels = kmeans.predict(X1)

print(labels)
plt.scatter(X1[:,0],X1[:,1], c=kmeans.labels_, cmap="rainbow")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color ="black")
plt.title("clusters of customers")
plt.xlabel("Age")
plt.ylabel("Spending score(1-100)")
plt.show()


# In[21]:


X2 = data.loc[:, ["Annual Income (k$)","Spending Score (1-100)"]].values

wcss = []

for k in range (1,11):
    kmeans = KMeans(n_clusters = k, init= "k-means++")
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color = "red", marker ="8")
plt.xlabel("K Values")
plt.ylabel("WCSS")
plt.show()


# In[22]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(X2)


labels = kmeans.predict(X2)

print(labels)
print(kmeans.cluster_centers_)
plt.scatter(X2[:,0],X2[:,1], c=kmeans.labels_, cmap="rainbow")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color ="black")
plt.title("clusters of customers")
plt.xlabel("Annual Income($k)")
plt.ylabel("Spending score(1-100)")
plt.show()


# In[25]:


X3 = data.loc[:,:]

wcss = []

for k in range (1,11):
    kmeans = KMeans(n_clusters = k, init= "k-means++")
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color = "red", marker ="8")
plt.xlabel("K Values")
plt.ylabel("WCSS")
plt.show()


# In[28]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])
kmeans = KMeans(n_clusters = 5)

label = kmeans.fit_predict(X3)

print(label)


# In[29]:


print(kmeans.cluster_centers_)


# In[39]:


X = np.random.rand(100, 3)

kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(X)

data = pd.DataFrame(X, columns=["X1", "X2", "X3"])
data["label"] = clusters

fig = plt.figure(figsize=(25,15))
ax= fig.add_subplot(111, projection='3d')
  
ax.scatter(data["X1"][data.label==0], data["X2"][data.label==0], data["X3"][data.label ==0], c='blue')
ax.scatter(data["X1"][data.label==1], data["X2"][data.label==1], data["X3"][data.label ==1], c='red')
ax.scatter(data["X1"][data.label==2], data["X2"][data.label==2], data["X3"][data.label ==2], c='violet')
ax.scatter(data["X1"][data.label==3], data["X2"][data.label==3], data["X3"][data.label ==3], c='purple')
ax.scatter(data["X1"][data.label==4], data["X2"][data.label==4], data["X3"][data.label ==4], c='green')
ax.view_init(30,185)

plt.xlabel("Age")
plt.ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')

plt.show()


# In[ ]:




