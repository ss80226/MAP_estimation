## Part 1. Split the training and testing data

import numpy as np
import pandas as pd
import scipy



df=pd.read_csv('Wine.csv', sep=',',header=None)
df_array = df.values
df_array.shape


label_count = [0, 0, 0]
for i in range(df_array.shape[0]):
    label_count[int(df_array[i][0]-1)] += 1


np.random.shuffle(df_array[0:label_count[0]])
np.random.shuffle(df_array[label_count[0]:label_count[0]+label_count[1]])
np.random.shuffle(df_array[label_count[0]+label_count[1]:df_array.shape[0]])

testing_list = []
training_list = []
testing_list.append(df_array[0:18])
training_list.append(df_array[18:label_count[0]])
testing_list.append(df_array[label_count[0]:label_count[0]+18])
training_list.append(df_array[label_count[0]+18:label_count[0]+label_count[1]])
testing_list.append(df_array[label_count[0]+label_count[1]:label_count[0]+label_count[1]+18])
training_list.append(df_array[label_count[0]+label_count[1]+18:df_array.shape[0]])

testing_array = np.array(testing_list)
training_array = np.array(training_list)
testing_array = np.reshape(testing_array, (-1, 14))
tmp = np.concatenate((training_array[0], training_array[1]))
training_array = np.concatenate((tmp, training_array[2]))


testing_df = pd.DataFrame(testing_array)
testing_df[0] = testing_df[0].apply(int)
training_df = pd.DataFrame(training_array)
training_df[0] = training_df[0].apply(int)


pd.DataFrame(testing_df).to_csv("./test.csv", index=False) # testing data
pd.DataFrame(training_df).to_csv("./train.csv", index=False) # training data


# ## Part 2. Use the MAP to predict the label in testing data


count_list = [0, 0, 0]
for data in training_df.values:
    count_list[int(data[0]-1)] += 1
feature_1 = np.zeros(shape=[13, count_list[0]])
feature_2 = np.zeros(shape=[13, count_list[1]])
feature_3 = np.zeros(shape=[13, count_list[2]])
for index, element in enumerate(training_df.values[0:count_list[0]]):
    for j in range(13):
        feature_1[j][index] = training_df.values[index][j+1]
for index, element in enumerate(training_df.values[count_list[0]:count_list[0]+count_list[1]]):
    for j in range(13):
        feature_2[j][index] = training_df.values[count_list[0]+index][j+1]
for index, element in enumerate(training_df.values[count_list[0]+count_list[1]:sum(count_list)]):
    for j in range(13):
        feature_3[j][index] = training_df.values[count_list[0]+count_list[1]+index][j+1]



import scipy.stats as st
likelihood_distributions = []
features = []
features.append(feature_1)
features.append(feature_2)
features.append(feature_3)
for i in range(3): # each label
    distribution_tmp = []
    for f_idx in range(13): # each features
        mean = np.mean(features[i][f_idx])
        std = np.std(features[i][f_idx])
        distribution_tmp.append(st.norm(mean, std))
    likelihood_distributions.append(distribution_tmp)
        


priors = [0., 0., 0.]
for i in range(3):
    priors[i] = count_list[i]/sum(count_list)



test_array = testing_df.values
np.random.shuffle(test_array)



from scipy import integrate
delta = 1e-6
counter = 0
correct = 0
for data in test_array:
    posts = [1., 1., 1.]
    for label_idx in range(3):
        post = 1.* priors[label_idx] 
        for f_idx in range(13):
            likelihood = integrate.quad(likelihood_distributions[label_idx][f_idx].pdf, data[f_idx+1], data[f_idx+1]+delta)[0]
            post = post * likelihood
        posts[label_idx] = post
    label = np.argmax(posts)
    counter += 1
    if label == int(data[0]-1):
        correct += 1
    else:
        pass
print('accuracy : ', correct/counter)


# ## Part 3.  Discussion of characteristics and Ploting of visualized result of testing data 


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# split the label and the features from testing data
X = testing_df.drop(0, 1)
y = testing_df[0]
# X = training_df.drop(0, 1)
# y = training_df[0]


pca2 = PCA(n_components=2)
X_p = pca2.fit(X).transform(X)
markers = ['s', 'x', 'o']
wines = ['wine_1', 'wine_2', 'wine_3']
labels = [1, 2, 3]
fig = plt.figure(figsize=(12,6))

#plot 2D
plt2 = fig.add_subplot(1,2,1)

for c, i, target_name, m in zip('rgb', labels, wines, markers):
    plt2.scatter(X_p[y==i, 0], X_p[y==i, 1], c=c, label=target_name, marker=m)
plt2.set_xlabel('PCA-feature-1')
plt2.set_ylabel('PCA-feature-2')
plt.legend(loc='upper right')


# plt.show()

plt3 = fig.add_subplot(1,2,2, projection='3d')

pca3 = PCA(n_components=3)
X_p = pca3.fit(X).transform(X)

for c, i, target_name, m in zip('rgb', labels, wines, markers):
    plt3.scatter(X_p[y==i, 0], X_p[y==i, 1], X_p[y==i, 2], c=c, label=target_name, marker=m)

plt3.set_xlabel('PCA-feature-1')
plt3.set_ylabel('PCA-feature-2')
plt3.set_zlabel('PCA-feature-3')
plt.legend(loc='upper right')
plt.tight_layout()

plt.savefig('./pca.png', dpi=300)
# plt.show()


