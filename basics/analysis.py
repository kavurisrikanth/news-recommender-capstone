#!/usr/bin/env python
# coding: utf-8

# ### Intro

# Data analysis for the NLP capstone project of the Upgrad Data Science course.

# Code committed to: https://github.com/kavurisrikanth/news-recommender-capstone

# ### The Basics - Loading data

# In[1000]:


import pandas as pd
import numpy as np

import plotly.express as px


# In[1001]:


txns = pd.read_csv('../data/consumer_transanctions.csv')
cnt = pd.read_csv('../data/platform_content.csv')


# In[1002]:


txns.head()


# In[1003]:


cnt.head()


# In[1004]:


# shared_data.get_data().store_transactions(txns)
# shared_data.get_data().store_content(cnt)


# In[ ]:





# In[ ]:





# ### Data preparation

# #### Drop unnecessary columns

# In[1005]:


# Drop country, consumer_location, consumer_device_info, consumer_session_id from txns
txns.drop(columns=['country', 'consumer_location', 'consumer_device_info', 'consumer_session_id'], inplace=True)


# In[ ]:





# In[1006]:


# Drop producer_id, producer_session_id, producer_device_info, producer_location, producer_country, item_type from cnt
cnt.drop(columns=['producer_id', 'producer_session_id', 'producer_device_info', 'producer_location', 'producer_country', 'item_type'], inplace=True)


# In[1007]:


content = cnt


# In[1008]:


content.head()


# #### Remove all docs that are not in English

# In[1009]:


content.language.value_counts()


# In[1010]:


content.shape


# In[1011]:


content = content[content['language'] == 'en']


# In[1012]:


content.shape


# #### Handle articles with duplicated entries

# In[1013]:


no_dups = content.sort_values('event_timestamp').drop_duplicates(subset=['title', 'text_description'], keep='last')


# In[1014]:


no_dups.head()


# In[1015]:


no_dups.reset_index(inplace=True)


# In[1016]:


no_dups.interaction_type.value_counts()


# In[1017]:


no_dups[no_dups['title'] == "Ethereum, a Virtual Currency, Enables Transactions That Rival Bitcoin's"]


# In[1018]:


content[content['title'] == "Ethereum, a Virtual Currency, Enables Transactions That Rival Bitcoin's"]


# The entry in the no duplicates DataFrame is the one with the older timestamp. Makes sense.

# In[1019]:


cnt = no_dups


# In[ ]:





# #### Introduce keywords

# In[1020]:


# %pip install gensim


# In[1021]:


from gensim.utils import simple_preprocess


# In[1022]:


cnt['text_description_preprocessed'] = cnt['text_description'].apply(lambda x: simple_preprocess(x, deacc=True))


# In[1023]:


cnt.head()


# In[1024]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_en = stopwords.words('english')


# In[1025]:


cnt['text_description_no_stopwords'] = cnt['text_description_preprocessed'].apply(lambda x: [word for word in x if word not in stopwords_en])


# In[1026]:


cnt.head()


# In[1027]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[1028]:


cnt['text_description_lemmatized'] = cnt['text_description_no_stopwords'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])


# In[1029]:


cnt.head()


# In[1030]:


# Drop the columns we don't need anymore
cnt.drop(['text_description_preprocessed', 'text_description_no_stopwords'], axis=1, inplace=True)


# #### Introduce a ratings column

# In[1031]:


def to_rating(val):
    if val == 'content_followed':
        return 5
    if val == 'content_commented_on':
        return 4
    if val == 'content_saved':
        return 3
    if val == 'content_liked':
        return 2
    return 1


# In[1032]:


txns.interaction_type.value_counts()


# In[1033]:


txns['rating'] = txns.interaction_type.apply(lambda x: to_rating(x))


# In[1034]:


txns.head()


# #### Adjust IDs

# The user and document IDs in the data make no sense. So create new IDs that start from 1.

# In[1035]:


class IdHelper:
    _map = {}
    _id = 1

    def translate(self, id):
        # If a mapping exists for id, then return the mapping
        # Otherwise, create a new mapping, store it, and return it
        if id in self._map:
            return self._map[id]
        new_id = self.__new_id__()
        self._map[id] = new_id
        return new_id

    def __new_id__(self):
        num = self._id
        self._id += 1
        return num


# In[1036]:


consumer_helper = IdHelper()
item_helper = IdHelper()


# In[1037]:


txns['consumer_id_adj'] = txns['consumer_id'].map(lambda x: consumer_helper.translate(x))


# In[1038]:


txns.head()


# In[1039]:


txns['item_id_adj'] = txns['item_id'].map(lambda x: item_helper.translate(x))


# In[1040]:


# Drop item_id and consumer_id from txns
txns.drop(columns=['item_id', 'consumer_id'], inplace=True)


# In[1041]:


txns.head()


# Same for content.

# In[1042]:


cnt.head()


# In[1043]:


cnt['item_id_adj'] = cnt['item_id'].map(lambda x: item_helper.translate(x))


# In[1044]:


# Drop item_id from cnt
cnt.drop(columns=['item_id'], inplace=True)


# In[1045]:


cnt.head()


# ### EDA

# #### Checking for missing values

# In[1046]:


txns.isna().sum()


# In[1047]:


txns.shape


# In[1048]:


cnt.isna().sum()


# In[1049]:


cnt.shape


# #### Checking for duplicated ratings

# In[1050]:


txns.head()


# In[1051]:


txns_2 = txns[['consumer_id_adj', 'item_id_adj', 'rating']]


# In[1052]:


txns_2.head()


# In[1053]:


duplicates = txns[txns.duplicated(subset=['consumer_id_adj', 'item_id_adj'], keep=False)]


# In[1054]:


duplicates.sort_values(by=['consumer_id_adj', 'item_id_adj', 'rating'], inplace=True)


# In[1055]:


duplicates.head()


# There are duplicated entries i.e., the same user has interacted with the same article multiple times.

# Since multiple interactions could mean that a user liked an article, the duplicates must be considered in the analysis.

# #### For "duplicated" transactions, calculate the average rating of the user for that article

# In[1056]:


grp = duplicates.groupby(by=['consumer_id_adj', 'item_id_adj'])['rating'].mean()


# In[1057]:


grp.head()


# In[1058]:


grp_df = pd.DataFrame(grp)


# In[1059]:


grp_df.head()


# Renaming the rating column to avoid any potential clash when merged with the original

# In[1060]:


grp_df.columns = ['rating_sum']


# In[1061]:


grp_df.head()


# In[1062]:


grp_df.reset_index(inplace=True)


# In[1063]:


grp_df.head()


# Check distributions of ratings

# In[1064]:


grp_df.describe(percentiles=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])


# In[1065]:


fig = px.box(grp_df, y='rating_sum')
fig.show()


# A majority of the articles are rated 2 or lower. Only a very small number of transactions have a high rating. However, these are not outliers. This is expected, as users would only like a small percentage of the articles in the system.

# #### Add the adjusted rating back to the original transactions DataFrame

# In[1066]:


no_dups = txns.drop_duplicates(subset=['consumer_id_adj', 'item_id_adj'])


# In[1067]:


no_dups.head()


# In[1068]:


no_dups.sort_values(by=['consumer_id_adj', 'item_id_adj', 'rating'], inplace=True)


# In[1069]:


no_dups.head()


# In[1070]:


duplicates.shape


# In[1071]:


no_dups.shape


# In[1072]:


txns.shape


# Merge the two DataFrames

# In[1073]:


txns_merged = pd.merge(left=no_dups, right=grp_df, left_on=['consumer_id_adj', 'item_id_adj'], right_on=['consumer_id_adj', 'item_id_adj'], how='left')


# In[1074]:


txns_merged.sort_values(by=['consumer_id_adj', 'item_id_adj', 'rating'], inplace=True)


# In[1075]:


txns_merged.head(25)


# Rows that have rating_sum as NaN were not duplicated in the original. So, the summed rating would just be the rating for these rows.

# In[1076]:


txns_merged['ratings_merged'] = txns_merged.rating_sum.fillna(txns_merged.rating)


# In[1077]:


txns_merged.head(25)


# In[1078]:


txns_merged.ratings_merged.describe()


# The rating is between 1 and 5, so that is good enough.

# In[1079]:


txns_merged.drop(columns=['rating_sum'], inplace=True)


# In[1080]:


txns_merged.head()


# In[1081]:


txns_merged.rename(columns={'rating': 'rating_original'}, inplace=True)


# In[1082]:


txns_merged.head()


# In[1083]:


# txns_merged.rename(columns={'ratings_scaled': 'rating'}, inplace=True)
txns_merged.rename(columns={'ratings_merged': 'rating'}, inplace=True)


# In[1084]:


txns_merged.head()


# In[1085]:


txns = txns_merged


# In[1086]:


txns.head()


# In[1087]:


txns.drop(columns=['interaction_type'], inplace=True)


# In[1088]:


txns.head()


# In[1089]:


txns.describe()


# Consolidated Ratings are between 1 and 4.5, which is expected.

# In[ ]:





# #### Plotting

# In[1090]:


px.histogram(txns, x='rating')


# In[1091]:


cnt.head()


# In[1092]:


px.histogram(cnt, x='language')


# In[ ]:





# ### Topic Modelling

# Try to create some basic topics under which each article may be categorized

# In[1093]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


# #### Feature extraction

# In[1094]:


vec = TfidfVectorizer(stop_words='english')
X = vec.fit_transform(cnt['text_description'])


# In[1095]:


test_df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())


# In[1096]:


test_df.head()


# #### NMF Decomposition

# In[1097]:


num_topics = 10
nmf = NMF(n_components=num_topics, random_state=42)
doc_topic = nmf.fit_transform(X)
topic_term = nmf.components_


# In[1098]:


# Getting the top 10 words for each topic

words = np.array(vec.get_feature_names())
topic_words = pd.DataFrame(
    np.zeros((num_topics, 10)),
    index=['topic_{}'.format(i + 1) for i in range(num_topics)],
    columns=['word_{}'.format(i + 1) for i in range(10)]
).astype(str)


# In[1099]:


topic_words


# Populating topic_words

# In[1100]:


for i in range(num_topics):
    idx = topic_term[i].argsort()[::-1][:10]
    topic_words.iloc[i] = words[idx]


# In[1101]:


topic_words


# In[1102]:


# Create a topic mapping for topic_words
# The topics in order are: 'Digital Marketing', 'E-Commerce', 'Cloud Computing', 'Data Science & Machine Learning', 'Cryptocurrency', 'Google', 'Apple', 'Facebook', 'Operating Systems & Runtimes', 'Computer Programming'
topic_mapping = {
    'topic_1': 'Digital Marketing',
    'topic_2': 'E-Commerce',
    'topic_3': 'Cloud Computing',
    'topic_4': 'Data Science & Machine Learning',
    'topic_5': 'Cryptocurrency',
    'topic_6': 'Google',
    'topic_7': 'Apple',
    'topic_8': 'Facebook',
    'topic_9': 'Operating Systems & Runtimes',
    'topic_10': 'Computer Programming'
}


# In[1103]:


doc_topic_df = pd.DataFrame(doc_topic, columns=['topic_{}'.format(i + 1) for i in range(num_topics)])


# In[1104]:


# Get the 5 topics with the highest probabilities for each document
doc_topic_df['top_topics'] = doc_topic_df.apply(lambda x: x.sort_values(ascending=False).index[:5].tolist(), axis=1)


# In[1105]:


doc_topic_df.head()


# In[1106]:


# Get the mapping for doc_topic_df.top_topics from topic_mapping and create a new column
doc_topic_df['top_topics_mapped'] = doc_topic_df.top_topics.apply(lambda x: [topic_mapping[i] for i in x])


# In[1107]:


doc_topic_df.head()


# In[1108]:


doc_topic_df.shape


# In[1109]:


# Add doc_topic_df.top_topics_mapped to cnt
cnt = pd.concat([cnt, doc_topic_df.top_topics_mapped], axis=1)


# In[1110]:


cnt.head()


# In[1111]:


# Rename cnt.top_topics_mapped to cnt.topics
cnt.rename(columns={'top_topics_mapped': 'topics'}, inplace=True)


# In[1112]:


cnt.head()


# With this, we have some idea of what topics each article is talking about.

# ## Getting articles for a User

# Consider user-based collaborative filtering, and ALS. Whichever gives the best result would be the model to use.

# ### User-based collaborative filtering

# In[1113]:


n_users = txns.consumer_id_adj.nunique()


# In[1114]:


n_articles = txns.item_id_adj.nunique()


# In[1115]:


# txns.consumer_id.values


# In[1116]:


print(f'Num users: {n_users}, Num articles: {n_articles}')


# ### Train test split

# In[1117]:


import sklearn
train, test = sklearn.model_selection.train_test_split(txns, test_size=0.3, random_state=42)


# In[1118]:


train.shape


# In[1119]:


test.shape


# In[1120]:


train.describe(percentiles=[0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])


# In[ ]:





# ### User-Article matrix

# Since this is collaborative filtering, we will consider the transactions matrix. From this, we construct a matrix of the ratings given by users for each product.

# Populate the training matrix

# In[1121]:


def create_and_populate_user_article_matrix(data):
    data_matrix = np.zeros((n_users, n_articles))

    for line in data.itertuples():
        # print(line)
        # print(type(line))
        # print(f'UserId: {line.consumer_id_adj}, ArticleId: {line.item_id_adj}, Rating: {line.rating}')
        # break
        user_id = line.consumer_id_adj
        article_id = line.item_id_adj
        rating = line.rating

        data_matrix[user_id - 1, article_id - 1] = rating
    
    return data_matrix


# Fill the training matrix with rating values

# In[1122]:


data_matrix = create_and_populate_user_article_matrix(train)


# In[1123]:


data_matrix


# In[1124]:


data_matrix.shape


# Dimensions match the number of unique users & articles

# Populate the testing matrix

# In[1125]:


data_matrix_test = create_and_populate_user_article_matrix(test)


# In[1126]:


data_matrix_test


# In[1127]:


data_matrix_test.shape


# ### Pairwise Distance

# In[1128]:


from sklearn.metrics.pairwise import pairwise_distances


# In[1129]:


user_similarity = 1 - pairwise_distances(data_matrix, metric='cosine')


# In[1130]:


user_similarity


# In[1131]:


user_similarity.shape


# Take the transpose of the data matrix in order to calculate the article similarity. Will be used later.

# In[1132]:


# data_matrix.shape


# In[1133]:


# data_matrix.T.shape


# In[1134]:


article_similarity = 1 - pairwise_distances(data_matrix.T, metric='cosine')


# In[1135]:


article_similarity


# In[1136]:


article_similarity.shape


# ### Get dot product of data matrix with similarity matrix

# In[1137]:


user_similarity.shape


# In[1138]:


data_matrix_test.shape


# In[1139]:


article_prediction = np.dot(user_similarity, data_matrix_test)


# In[1140]:


article_prediction.shape


# In[1141]:


article_pred_df = pd.DataFrame(article_prediction)


# In[1142]:


article_pred_df.head()


# In[1143]:


txns.consumer_id_adj.value_counts()


# ### Test for one user

# In[1144]:


test.head()


# In[1145]:


test_user_id = 962
test_user_idx = test_user_id - 1


# In[1146]:


test_user_id in test.consumer_id_adj.values


# In[1147]:


article_pred_df.iloc[test_user_idx]


# In[1148]:


article_recommendation = pd.DataFrame(article_pred_df.iloc[test_user_idx].sort_values(ascending=False))


# In[1149]:


article_recommendation


# In[1150]:


article_recommendation.reset_index(inplace=True)


# In[1151]:


article_recommendation.head()


# Since the matrix is zero-based, the article ID index that we get is also zero-based. However, our IDs are one-based. So, convert the article ID to one-based by adding 1.

# In[1152]:


article_recommendation['index'] = article_recommendation['index'] + 1


# In[1153]:


article_recommendation.head()


# In[1154]:


article_recommendation.rename(columns={'index': 'article_id', test_user_idx: 'score'}, inplace=True)


# In[1155]:


article_recommendation.head()


# Merging with the content dataframe to get the article title.

# In[1156]:


merged = pd.merge(article_recommendation, cnt, left_on='article_id', right_on='item_id_adj', how='left')


# In[1157]:


merged.columns


# In[1158]:


keep = ['article_id', 'score', 'title', 'interaction_type']


# In[1159]:


merged = merged.drop(columns=[col for col in merged if col not in keep])


# In[1160]:


merged.head(10)


# In[1161]:


cnt[cnt['item_id_adj'] == 203]


# Some articles have title as NaN. This is because they do not exist in the content DataFrame, meaning they were pulled out of the system, or that data was somehow lost.
# 
# These entries can be used for analysis. However, they must not be included in any results.

# In[1162]:


merged.shape


# In[1163]:


merged = merged[~(merged['title'].isna())]


# In[1164]:


merged.shape


# Of the remaining suggestions, some might have been pulled out of the system. Filter those out.

# In[1165]:


merged[merged['interaction_type'] == 'content_pulled_out']


# In[1166]:


merged = merged[merged['interaction_type'] != 'content_pulled_out']


# In[1167]:


merged.shape


# In[1168]:


merged.head()


# ### Evaluate the predictions of the Collaborative User-based model

# In[1169]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from math import sqrt


# In[1170]:


data_matrix


# In[1171]:


data_matrix_test


# In[1172]:


article_prediction


# In[1173]:


data_matrix_test_nz = data_matrix_test.nonzero()


# In[1174]:


prediction = article_prediction[data_matrix_test_nz]


# In[1175]:


ground_truth = data_matrix_test[data_matrix_test_nz]


# #### Mean Absolute Error

# In[1176]:


mean_absolute_error(prediction, ground_truth)


# #### Root Mean Square Error

# In[1177]:


sqrt(mean_squared_error(prediction, ground_truth))


# #### Precision

# Out of the recommended items, how many did the user like?

# In[1178]:


num_pred = 10


# In[1179]:


predicted_article_ids_for_user = merged['article_id'].values[:num_pred]


# In[1180]:


predicted_article_ids_for_user


# In[1181]:


def get_articles_that_user_liked(user_id):
    # For this, we get all the articles that user has given a rating of more than the average rating
    # Get the average rating for the user
    avg = txns[txns['consumer_id_adj'] == user_id].rating.mean()

    user_interactions = txns[(txns['consumer_id_adj'] == user_id) & (txns['rating'] > avg)].sort_values(by='rating', ascending=False)

    if (len(user_interactions) == 0):
        user_interactions = txns[(txns['consumer_id_adj'] == user_id)].sort_values(by='rating', ascending=False)

    return user_interactions[['item_id_adj', 'rating']]


# Since in the txns DataFrame, all IDs are 1-indexed, we can use the test user ID as it is.

# In[1182]:


user_interactions = get_articles_that_user_liked(test_user_id)


# In[1183]:


user_interactions.head()


# In[1184]:


actual_article_ids_for_user = user_interactions['item_id_adj'].values


# In[1185]:


set(predicted_article_ids_for_user)


# In[1186]:


set(actual_article_ids_for_user)


# Get intersection of predictions and user interactions

# In[1187]:


set(predicted_article_ids_for_user) & set(actual_article_ids_for_user)


# In[1188]:


correctly_predicted_article_ids = set(predicted_article_ids_for_user) & set(actual_article_ids_for_user)


# Some of the articles that user liked are identified

# Precision = #Correct predictions / #Predictions

# In[1189]:


precision = len(correctly_predicted_article_ids) / len(predicted_article_ids_for_user)


# In[1190]:


precision


# #### Recall

# Recall is the ratio of liked articles that the system is able to identify correctly

# Recall = #Correct Predictions / #Liked Articles

# In[1191]:


recall = len(correctly_predicted_article_ids) / len(actual_article_ids_for_user)


# In[1192]:


recall


# In order to evaluate the filtering method over the entire test data, get the metrics as defined above, and take the average

# In[1193]:


# Helper methods
def evaluate_user_based_filtering(test):
    # For each unique consumer_id_adj in the test DataFrame, we will evaluate the precision and recall
    # of the user-based filtering algorithm
    total_precision = 0
    total_recall = 0

    test_user_ids = test.consumer_id_adj.unique()
    num_users = len(test_user_ids)
    for test_user_id in test_user_ids:
        # Get the articles that the user has liked
        user_interactions = get_articles_that_user_liked(test_user_id)
        actual_article_ids_for_user = user_interactions['item_id_adj'].values

        if (len(actual_article_ids_for_user) == 0):
            # If the user has not liked any articles, we will skip this user
            # Print the user id so that we can keep track of the progress
            print('Skipping user: ', test_user_id)
            num_users -= 1
            continue

        # Get the articles that the user-based filtering algorithm has recommended
        test_user_idx = test_user_id - 1
        article_recommendation = pd.DataFrame(article_pred_df.iloc[test_user_idx].sort_values(ascending=False))
        article_recommendation.reset_index(inplace=True)
        article_recommendation['index'] = article_recommendation['index'] + 1
        article_recommendation.rename(columns={'index': 'article_id', test_user_idx: 'score'}, inplace=True)
        merged = pd.merge(article_recommendation, cnt, left_on='article_id', right_on='item_id_adj', how='left')
        keep = ['article_id', 'score', 'title', 'interaction_type']
        merged = merged.drop(columns=[col for col in merged if col not in keep])
        merged = merged[~(merged['title'].isna())]
        merged = merged[merged['interaction_type'] != 'content_pulled_out']
        predicted_article_ids_for_user = merged['article_id'].values[:num_pred]

        # Calculate precision and recall
        correctly_predicted_article_ids = set(predicted_article_ids_for_user) & set(actual_article_ids_for_user)
        precision = len(correctly_predicted_article_ids) / len(predicted_article_ids_for_user)
        recall = len(correctly_predicted_article_ids) / len(actual_article_ids_for_user)
        
        total_precision += precision
        total_recall += recall
    
    # Return the average precision and recall as a tuple
    return (total_precision / num_users, total_recall / num_users)


# In[1194]:


# Evaluate the user-based filtering algorithm and store the results in 2 variables
avg_precision, avg_recall = evaluate_user_based_filtering(test)


# In[1195]:


# Round the results to 3 decimal places and print them
print('Average precision: ', round(avg_precision, 3))
print('Average recall: ', round(avg_recall, 3))


# Check if ALS does better.

# Expose method to get recommendations for a user

# In[1196]:


def get_articles_for_user_from_user_based(user_id, n=-1):
    user_idx = user_id - 1

    recommendation = pd.DataFrame(article_pred_df.iloc[user_idx].sort_values(ascending=False))

    recommendation.reset_index(inplace=True)

    recommendation['index'] = recommendation['index'] + 1

    recommendation.rename(columns={'index': 'article_id', user_idx: 'score'}, inplace=True)

    merged = pd.merge(recommendation, cnt, left_on='article_id', right_on='item_id_adj', how='left')

    keep = ['article_id', 'title', 'score', 'topics', 'interaction_type']

    merged = merged.drop(columns=[col for col in merged if col not in keep])

    merged = merged[merged['interaction_type'] != 'content_pulled_out']

    # Drop rows with NaN values
    merged.dropna(inplace=True)

    # Reset the index
    merged.reset_index(inplace=True, drop=True)

    # Drop interaction_type
    merged = merged.drop(columns=['interaction_type'])

    # Sort by score
    merged = merged.sort_values(by='score', ascending=False)

    # Return the top n articles if n is specified
    if (n > 0):
        return merged[:n]

    return merged


# In[1197]:


get_articles_for_user_from_user_based(test_user_id)


# In[1198]:


get_articles_for_user_from_user_based(test_user_id, 10)


# ## Alternating Least Squares method

# #### Create sparse User-Article matrix

# In[1199]:


from scipy.sparse import csr_matrix


# Random values in CSR matrix will be filled with alpha value

# In[1200]:


txns.head()


# In[1201]:


keep = ['consumer_id_adj', 'item_id_adj', 'rating']


# In[1202]:


txns_mod = txns.drop(columns=[col for col in txns.columns if col not in keep])


# In[1203]:


txns_mod.head()


# In[1204]:


txns_mod.describe()


# In[1205]:


alpha = 40


# In[1206]:


txns_mod.shape


# In[ ]:





# In[1207]:


txns_mod.shape[0]


# In[1208]:


x = [alpha] * txns_mod.shape[0]


# In[1209]:


len(x)


# In[1210]:


sparse_user_article = csr_matrix( ([alpha]*txns_mod.shape[0], (txns_mod['consumer_id_adj'], txns_mod['item_id_adj']) ))


# In[1211]:


sparse_user_article


# In[1212]:


n_users


# In[1213]:


n_articles


# Matrix dimensions match with the number of users & articles, accounting for the extra row at index 0

# Convert to array

# In[1214]:


csr_user_array = sparse_user_article.toarray()


# In[1215]:


csr_user_array


# In[1216]:


n_users


# In[1217]:


len(csr_user_array), len(csr_user_array[0])


# Dimensions match with the matrix

# In[1218]:


max(csr_user_array[1])


# Create article-user sparse matrix

# In[1219]:


sparse_article_user = sparse_user_article.T.tocsr()


# In[1220]:


sparse_article_user


# Shape matches

# In[1221]:


csr_article_array = sparse_article_user.toarray()


# #### Create train & test data

# In[1222]:


get_ipython().run_line_magic('pip', 'install implicit')


# In[1223]:


from implicit.evaluation import train_test_split


# In[1224]:


sparse_article_user


# In[1225]:


train, test = train_test_split(sparse_user_article, train_percentage=0.8)


# In[1226]:


train


# In[1227]:


test


# #### Building the ALS Model

# In[1228]:


from implicit.als import AlternatingLeastSquares


# In[1229]:


model = AlternatingLeastSquares(factors=60, regularization=0.1, iterations=60, calculate_training_loss=False)


# In[1230]:


# model


# Training

# In[1231]:


model.fit(train)


# In[1232]:


# test


# In[ ]:





# In[1233]:


# test_user_id = 114


# In[1234]:


user_interactions = get_articles_that_user_liked(test_user_id)


# New Implicit API expects (user, item) sparse matrix as input

# In[1235]:


model.recommend(test_user_id, sparse_user_article[test_user_id], N=20, filter_already_liked_items=False)


# In[1236]:


ids, scores = model.recommend(test_user_id, sparse_user_article[test_user_id], N=20, filter_already_liked_items=False)


# In[1237]:


out = pd.DataFrame({'article_id': ids, 'als_score': scores})


# In[1238]:


# out


# In[1239]:


out.head(num_pred)


# In[1240]:


out.shape


# In[1241]:


user_interactions.head(10)


# In[1242]:


user_interactions.shape


# In[1243]:


actual_article_ids_for_user = set(user_interactions['item_id_adj'].values)


# In[1244]:


predicted_article_ids_for_user = set(out['article_id'].values)


# In[1245]:


correctly_predicted_article_ids = actual_article_ids_for_user & predicted_article_ids_for_user


# In[1246]:


precision = len(correctly_predicted_article_ids) / len(predicted_article_ids_for_user)


# In[1247]:


recall = len(correctly_predicted_article_ids) / len(actual_article_ids_for_user)


# In[1248]:


# Print the precision and recall
print('Precision: ', precision)
print('Recall: ', recall)


# Similar to user-based collaborative filtering, evaluate ALS

# implicit.evaluation already contains a mean_average_precision_at_k method

# In[1249]:


from implicit.evaluation import precision_at_k


# In[1250]:


p_at_k = precision_at_k(model, train, test, K=10)


# In[1251]:


# Round the results to 3 decimal places and print them
print('Precision at k: ', round(p_at_k, 3))


# Check if better precision@k is possible with hyperparameter tuning

# In[1252]:


import itertools


# In[1253]:


if False:
    factors = [60, 80, 85, 87, 90, 92, 95, 100]
    regularization = [0.1, 0.11, 0.115, 0.12, 0.125]
    iterations = [30, 35, 40, 45, 50, 60]

    # Create a DataFrame to store the results
    results = pd.DataFrame(columns=['factors', 'regularization', 'iterations', 'precision_at_k'])
    for (f, r, i) in itertools.product(factors, regularization, iterations):
        model = AlternatingLeastSquares(factors=f, regularization=r, iterations=i, calculate_training_loss=False)
        model.fit(train, show_progress=False)
        p_at_k = precision_at_k(model, train, test, K=10, show_progress=False)

        # Append the results to the DataFrame
        # Create a temp DataFrame to store the results
        temp_results = pd.DataFrame([[f, r, i, p_at_k]], columns=['factors', 'regularization', 'iterations', 'precision_at_k'])
        
        # Concatenate the temp DataFrame to the results DataFrame
        results = pd.concat([results, temp_results], ignore_index=True)


# In[1254]:


if False:
    # Sort the results by precision_at_k and print the top 5
    results.sort_values(by='precision_at_k', ascending=False, inplace=True)
    results.head()


# Got best params from tuning

# precision@k = 0.144

# In[1255]:


best_user_based_f = 92
best_user_based_r = 0.115
best_user_based_i = 40


# In[1256]:


best_user_based_als = AlternatingLeastSquares(
    factors=best_user_based_f, 
    regularization=best_user_based_r, 
    iterations=best_user_based_i, 
    calculate_training_loss=False
)
best_user_based_als.fit(train)


# In[1257]:


ids, scores = best_user_based_als.recommend(test_user_id, sparse_user_article[test_user_id], N=20, filter_already_liked_items=True)


# precision@k is higher than that of User-based collaborative filtering, so ALS can be used for getting articles for a user.

# Expose method

# In[1258]:


def get_articles_for_user_from_als(user_id, n=20):
    global best_user_based_als
    if not best_user_based_als:
        best_user_based_als = AlternatingLeastSquares(
            factors=best_user_based_f, 
            regularization=best_user_based_r, 
            iterations=best_user_based_i, 
            calculate_training_loss=False
        )
        best_user_based_als.fit(train)
    id, scores = best_user_based_als.recommend(user_id, sparse_user_article[user_id], N=50, filter_already_liked_items=True)

    out = pd.DataFrame({'item_id_adj': id, 'score': scores})

    # Merge out with cnt on item_id_adj
    merged = out.merge(cnt, how='left', on='item_id_adj')

    # Keep only item_id_adj, title, score, and topics
    merged = merged[['item_id_adj', 'title', 'score', 'topics']]

    # Drop rows with NaN values
    merged.dropna(inplace=True)

    # Reset index
    merged.reset_index(drop=True, inplace=True)

    # Round score to 3 decimal places
    merged['score'] = merged['score'].apply(lambda x: round(x, 3))

    # Sort by score
    merged.sort_values(by='score', ascending=False, inplace=True)

    return merged[:n]


# In[1259]:


get_articles_for_user_from_als(test_user_id, n=10)


# ## Getting articles matching another article

# Consider item-based collaborative filtering and content-based filtering

# ### Item-based collaborative filtering

# Use article_similarity matrix constructed earlier

# In[1260]:


article_similarity.shape


# In[1261]:


n_articles


# In[ ]:





# In[1262]:


n_articles


# In[1263]:


data_matrix_test.shape


# In[1264]:


data_matrix_test.T.shape


# In[1265]:


other_article_prediction = np.dot(article_similarity, data_matrix_test.T)


# In[1266]:


other_article_prediction.shape


# In[1267]:


other_article_pred_df = pd.DataFrame(other_article_prediction)


# In[1268]:


other_article_pred_df.head()


# #### Test for one article

# In[1269]:


test_article_id = 1190


# In[1270]:


test_article_idx = test_article_id - 1


# In[1271]:


article_similarity[test_article_idx]


# In[1272]:


df = pd.DataFrame(article_similarity[test_article_idx], columns=['score'])


# In[1273]:


df.head()


# In[1274]:


df.reset_index(inplace=True)


# In[1275]:


df.head()


# In[1276]:


df['index'] = df['index'] + 1


# In[1277]:


df.head()


# In[1278]:


df.rename(columns={'index': 'item_id_adj'}, inplace=True)


# In[1279]:


df.sort_values(by='score', ascending=False, inplace=True)


# In[1280]:


df.head()


# In[1281]:


cnt[(cnt['item_id_adj'] == 1190) | (cnt['item_id_adj'] == 918)][['item_id_adj', 'title', 'text_description', 'topics']]


# In[ ]:





# Expose method

# In[1282]:


def get_articles_matching_article_from_item_based(article_id, n=-1, all=False):
    article_idx = article_id - 1

    out = pd.DataFrame(article_similarity[article_idx], columns=['score'])

    out.reset_index(inplace=True)

    out['index'] = out['index'] + 1

    out.rename(columns={'index': 'item_id_adj'}, inplace=True)

    out.sort_values(by='score', ascending=False, inplace=True)

    # Merge out with cnt on item_id_adj
    merged = out.merge(cnt, how='left', on='item_id_adj')

    # Keep only item_id_adj, title, score, and topics
    merged = merged[['item_id_adj', 'title', 'score', 'topics']]

    # Drop rows with NaN values
    merged.dropna(inplace=True)

    # Reset index
    merged.reset_index(drop=True, inplace=True)

    # Round score to 3 decimal places
    merged['score'] = merged['score'].apply(lambda x: round(x, 3))

    # Sort by score
    merged.sort_values(by='score', ascending=False, inplace=True)

    if n == -1 or all:
        return merged

    return merged[:n]


# In[1283]:


get_articles_matching_article_from_item_based(test_article_id, n=10)


# ### ALS for Articles

# Use sparse_article_user created earlier

# In[1284]:


item_train, item_test = train_test_split(sparse_article_user, train_percentage=0.8, random_state=42)


# In[1285]:


model = AlternatingLeastSquares(factors=60, regularization=0.1, iterations=60, calculate_training_loss=False)


# In[1286]:


model.fit(item_train)


# In[1287]:


precision_at_k(model, item_train, item_test, K=10)


# Precision@k value is 0.193. Check for a better value with Hyperparameter tuning.

# In[1288]:


def item_based_hyperparameter_tuning():
    factors = [10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    regularization = [0.7, 0.8, 0.9, 0.95, 1, 1.1, 1.2, 1.5]
    iterations = [80, 90, 100, 110, 120, 130, 140, 150]

    # Create a DataFrame to store the results
    results = pd.DataFrame(columns=['factors', 'regularization', 'iterations', 'precision_at_k'])
    for (f, r, i) in itertools.product(factors, regularization, iterations):
        model = AlternatingLeastSquares(factors=f, regularization=r, iterations=i, calculate_training_loss=False)
        model.fit(train, show_progress=False)
        p_at_k = precision_at_k(model, item_train, item_test, K=10, show_progress=False)

        # Append the results to the DataFrame
        # Create a temp DataFrame to store the results
        temp_results = pd.DataFrame([[f, r, i, p_at_k]], columns=['factors', 'regularization', 'iterations', 'precision_at_k'])
        
        # Concatenate the temp DataFrame to the results DataFrame
        results = pd.concat([results, temp_results], ignore_index=True)

    # Sort the results by precision_at_k and print the top 5
    results.sort_values(by='precision_at_k', ascending=False, inplace=True)
    return results


# In[1289]:


if False:
    results = item_based_hyperparameter_tuning()
    print(results.head())


# In[ ]:





# After hyperparameter tuning

# In[1290]:


best_article_als_f = 20
best_article_als_r = 1.2
best_article_als_i = 120


# In[1291]:


best_item_als = AlternatingLeastSquares(
    factors=best_article_als_f, 
    regularization=best_article_als_r, 
    iterations=best_article_als_i, 
    calculate_training_loss=False
)
best_item_als.fit(item_train)


# Test for one article

# In[1292]:


test_article_id = 1190


# In[1293]:


ids, scores = best_item_als.recommend(test_article_id, sparse_article_user[test_article_id], N=20, filter_already_liked_items=False)


# In[1294]:


# Create a DataFrame of the recommended article ids and scores
collab_out = pd.DataFrame({'article_id': ids, 'Score': scores})


# In[1295]:


collab_out.head()


# In[ ]:





# In[1296]:


# Define a function to get the article title from the article id
def get_article_title(article_id):
    # If the article id is not in the article dataframe, log that it is missing
    if article_id not in cnt['item_id_adj'].values:
        print('Missing article id: ', article_id)
        return None
    return cnt[cnt['item_id_adj'] == article_id]['title'].values[0]


# In[1297]:


def get_article_topics(article_id):
    # If the article id is not in the article dataframe, log that it is missing
    if article_id not in cnt['item_id_adj'].values:
        print('Missing article id: ', article_id)
        return None
    return cnt[cnt['item_id_adj'] == article_id]['topics'].values[0]


# In[1298]:


# Get the article title from the article ids
collab_out['title'] = collab_out['article_id'].apply(lambda x: get_article_title(x))


# In[1299]:


collab_out.head()


# In[1300]:


# Get the article topics from the article ids
collab_out['topics'] = collab_out['article_id'].apply(lambda x: get_article_topics(x))


# In[1301]:


collab_out.head()


# In[1302]:


# Drop rows with missing article titles
collab_out.dropna(inplace=True)


# In[1303]:


collab_out


# Expose method

# In[1304]:


def get_articles_matching_article_from_als(article_id, n=20, all=False):
    ids, scores = best_item_als.similar_items(
        article_id, item_users=sparse_article_user, N=50 if not all else n_articles)

    out = pd.DataFrame({'item_id_adj': ids, 'score': scores})

    merged = pd.merge(out, cnt, how='left', on='item_id_adj')

    keep = ['item_id_adj', 'score', 'title', 'topics']

    merged = merged.drop(columns=[col for col in merged if col not in keep])

    merged.dropna(inplace=True)

    # reset index
    merged.reset_index(drop=True, inplace=True)

    # round score to 3 decimal places
    merged['score'] = merged['score'].apply(lambda x: round(x, 3))

    # sort by score
    merged.sort_values(by='score', ascending=False, inplace=True)

    if all:
        return merged

    return merged[:n]


# In[1305]:


get_articles_matching_article_from_als(test_article_id, n=10)


# ### Content-based filtering

# #### Derive keywords from the article text

# In[1306]:


cnt.columns


# In[1307]:


cnt.head()


# In[1308]:


cnt.text_description


# In[1314]:


# Join cnt.text_description_lemmatized into a single list
words_list = []
for doc in cnt.text_description_lemmatized:
    words_list.append(doc)


# In[1315]:


len(words_list)


# In[1316]:


words_list[0][:10]


# In[1317]:


cnt.shape


# In[1318]:


words_list[0]


# In[1319]:


len(words_list), len(words_list[0]), len(words_list[1])


# In[ ]:





# #### Create Dictionary, Bag of Words, tfidf model & Similarity matrix

# In[1320]:


# %pip install gensim


# In[1321]:


from gensim.corpora.dictionary import Dictionary


# In[1322]:


# create a dictionary from words list
dictionary = Dictionary(words_list)


# In[1323]:


dictionary


# In[1324]:


len(dictionary)


# In[1325]:


number_words = 0
for word in words_list:
    number_words = number_words + len(word)


# In[1326]:


number_words


# In[1327]:


dictionary.get(0), dictionary.get(1), dictionary.get(2)


# ##### Generating Bag of Words

# In[1328]:


bow = dictionary.doc2bow(words_list[0])


# In[1329]:


len(words_list[0]), len(bow)


# Some words are repeated

# ##### Generating a corpus

# In[1330]:


#create corpus where the corpus is a bag of words for each document
corpus = [dictionary.doc2bow(doc) for doc in words_list] 


# In[1331]:


len(corpus), len(corpus[0]), len(corpus[1])


# All the articles are in the corpus, and the length of the first matches the count in the Bag of Words above

# ##### Use the TfIdf model on the corpus

# In[1332]:


from gensim.models.tfidfmodel import TfidfModel


# In[1333]:


#create tfidf model of the corpus
tfidf = TfidfModel(corpus) 


# In[1334]:


tfidf


# In[1335]:


len(tfidf[corpus[0]])


# In[1336]:


len(tfidf[corpus[1]])


# Again, the lengths are matched

# ##### Generate Similarity matrix

# In[1337]:


from gensim.similarities import MatrixSimilarity

# Create the similarity matrix. This is the most important part where we get the similarities between the movies.
sims = MatrixSimilarity(tfidf[corpus], num_features=len(dictionary))


# In[1338]:


len(dictionary)


# In[1339]:


# Flatten words_list into a set of unique words
words_set = set([word for doc in words_list for word in doc])


# In[1340]:


len(set(words_set))


# 

# In[1341]:


sims


# In[1342]:


sims[corpus[0]]


# In[1343]:


len(sims[corpus[0]])


# In[1344]:


len(sims)


# #### Generating recommendations

# In[1345]:


def article_recommendation(content):
    # get a bag of words from the content
    query_doc_bow = dictionary.doc2bow(content) 

    #convert the regular bag of words model to a tf-idf model
    query_doc_tfidf = tfidf[query_doc_bow] 

    # get similarity values between input movie and all other movies
    similarity_array = sims[query_doc_tfidf] 

    #Convert to a Series
    similarity_series = pd.Series(similarity_array.tolist(), index=cnt['item_id_adj']) 

    #get the most similar movies 
    # similarity_output = similarity_series.sort_values(ascending=False)
    similarity_output = similarity_series
    return similarity_output


# In[1346]:


test_article_id


# In[1347]:


cnt[cnt['item_id_adj'] == test_article_id]


# In[1348]:


test_desc = cnt[cnt['item_id_adj'] == test_article_id]['text_description_lemmatized'].values[0]


# In[1349]:


recs = article_recommendation(test_desc)


# In[1350]:


recs[:10]


# In[1351]:


recs_df = pd.DataFrame(recs, columns=['Score'])


# In[1352]:


recs_df.head()


# In[1353]:


recs_df.reset_index(inplace=True)


# In[1354]:


recs_df.head()


# In[1355]:


recs_df.isna().sum()


# In[1356]:


recs_df = cnt.merge(recs_df, on='item_id_adj', how='left')


# In[1357]:


recs_df.head()


# In[1358]:


recs_df.sort_values(by='Score', ascending=False, inplace=True)


# In[1359]:


recs_df.isna().sum()


# In[1360]:


keep = ['Score', 'title', 'text_description', 'topics', 'item_id_adj']


# In[1361]:


recs_df.drop(columns=[col for col in recs_df if col not in keep], inplace=True)


# In[1362]:


recs_df.head()


# Expose method

# In[1363]:


def get_articles_matching_article_from_content_based(article_id, n=-1):
    lemmatized_desc = cnt[cnt['item_id_adj'] == article_id]['text_description_lemmatized'].values[0]

    recommendations = article_recommendation(lemmatized_desc)

    recommendations_df = pd.DataFrame(recommendations, columns=['score'])

    recommendations_df.reset_index(inplace=True)

    recommendations_df = cnt.merge(recommendations_df, on='item_id_adj', how='left')

    recommendations_df.sort_values(by='score', ascending=False, inplace=True)

    keep = ['score', 'title', 'topics', 'item_id_adj']

    recommendations_df.drop(columns=[col for col in recommendations_df if col not in keep], inplace=True)

    # Drop rows with NaN
    recommendations_df.dropna(inplace=True)

    # Reset index
    recommendations_df.reset_index(drop=True, inplace=True)

    if n > 0:
        recommendations_df = recommendations_df[:n]

    return recommendations_df


# In[1364]:


get_articles_matching_article_from_content_based(test_article_id, n=10)


# 
# #### Comparing item-based and content-based filtering

# In[1365]:


num_articles = len(collab_out)


# In[1366]:


num_articles


# In[1367]:


# Assign the first num_articles rows from recs_df to content_out
content_out = recs_df.iloc[:num_articles]


# In[1368]:


content_out.reset_index(inplace=True)


# In[1369]:


content_out.head()


# In[1370]:


cnt[cnt['item_id_adj'] == test_article_id][['title', 'topics']]


# In[1371]:


# Rename index to article_id
content_out.rename(columns={'item_id_adj': 'article_id'}, inplace=True)
content_out.drop(columns=['index'], inplace=True)


# In[1372]:


content_out.head()


# In[1373]:


collab_out.head()


# In[1374]:


# Left join the content_out and collab_out DataFrames on article_id
out = pd.merge(collab_out, content_out, on='article_id', how='left')


# In[1375]:


content_out


# In[1376]:


collab_out


# In[1377]:


out


# In[1378]:


content_out.shape


# There isn't much overlap between the item-based collaborative, content-based, and ALS results.

# Check if combining with ALS improves the results

# In[ ]:





# ### Combining item-based filterings

# In[1379]:


item_als_result = get_articles_matching_article_from_als(test_article_id, n=50, all=True)


# In[1380]:


item_als_result.shape


# In[1381]:


item_collab_result = get_articles_matching_article_from_item_based(test_article_id)


# In[1382]:


item_collab_result.shape


# In[1383]:


item_content_result = get_articles_matching_article_from_content_based(test_article_id)


# In[1384]:


item_content_result.shape


# #### Normalizing the similarity scores using Min-Max normalization

# In[1385]:


# Normalize the scores in item_als_result
item_als_result['normalized_score_als'] = (item_als_result['score'] - min(item_als_result['score'])) / (max(item_als_result['score']) - min(item_als_result['score']))


# In[1386]:


min(item_als_result['score'])


# In[1387]:


item_als_result.head()


# In[1388]:


# Normalize the scores in item_collab_result
item_collab_result['normalized_score_collab'] = (item_collab_result['score'] - min(item_collab_result['score'])) / (max(item_collab_result['score']) - min(item_collab_result['score']))


# In[1389]:


min(item_collab_result['score'])


# In[1390]:


item_collab_result.head()


# In[1391]:


# Normalize the scores in item_content_result
item_content_result['normalized_score_content'] = (item_content_result['score'] - min(item_content_result['score'])) / (max(item_content_result['score']) - min(item_content_result['score']))


# In[1392]:


min(item_content_result['score'])


# In[1393]:


item_content_result.head()


# #### Item-based & content-based

# In[1394]:


item_collab_result.head()


# In[1395]:


item_content_result.head()


# In[1396]:


item_content_hybrid = pd.merge(item_content_result, item_collab_result, on='item_id_adj', how='left')


# In[1397]:


item_content_hybrid.shape


# In[1398]:


item_content_hybrid.isna().sum()


# In[1399]:


item_content_hybrid.dropna(inplace=True)


# In[1400]:


item_content_hybrid.head()


# In[1401]:


# Drop title_y and topics_y
item_content_hybrid.drop(columns=['title_y', 'topics_y'], inplace=True)


# In[1402]:


item_content_hybrid.head()


# In[1403]:


# Store the average of the normalized scores in a new column
item_content_hybrid['final_score'] = item_content_hybrid[['normalized_score_content', 'normalized_score_collab']].mean(axis=1)


# In[1404]:


item_content_hybrid.head()


# In[1405]:


# Sort the DataFrame by final_score in descending order
item_content_hybrid.sort_values(by='final_score', ascending=False, inplace=True)

# Reset the index
item_content_hybrid.reset_index(drop=True, inplace=True)


# In[1406]:


item_content_hybrid.head()


# In[1407]:


# Drop the score_x, score_y, normalized_score_content and normalized_score_collab columns
item_content_hybrid.drop(columns=['score_x', 'score_y', 'normalized_score_content', 'normalized_score_collab'], inplace=True)

# Rename title_x to title and topics_x to topics
item_content_hybrid.rename(columns={'title_x': 'title', 'topics_x': 'topics'}, inplace=True)


# In[1408]:


item_content_hybrid.head()


# Expose method

# In[1409]:


def get_articles_matching_article_from_item_content_hybrid(article_id, n=-1, ignore=[]):
    item_collab_result = get_articles_matching_article_from_item_based(article_id)

    # Normalize the scores in item_collab_result
    item_collab_result['normalized_score_collab'] = (item_collab_result['score'] - min(item_collab_result['score'])) / (max(item_collab_result['score']) - min(item_collab_result['score']))

    item_content_result = get_articles_matching_article_from_content_based(article_id)

    # Normalize the scores in item_content_result
    item_content_result['normalized_score_content'] = (item_content_result['score'] - min(item_content_result['score'])) / (max(item_content_result['score']) - min(item_content_result['score']))

    item_content_hybrid = pd.merge(item_content_result, item_collab_result, on='item_id_adj', how='left')

    item_content_hybrid.dropna(inplace=True)

    # Drop title_y and topics_y
    item_content_hybrid.drop(columns=['title_y', 'topics_y'], inplace=True)

    # Store the average of the normalized scores in a new column
    item_content_hybrid['final_score'] = item_content_hybrid[['normalized_score_content', 'normalized_score_collab']].mean(axis=1)

    # Drop the rows that have item_id_adj in ignore if ignore is not empty
    if len(ignore) > 0:
        item_content_hybrid = item_content_hybrid[~item_content_hybrid['item_id_adj'].isin(ignore)]

    # Sort the DataFrame by final_score in descending order
    item_content_hybrid.sort_values(by='final_score', ascending=False, inplace=True)

    # Reset the index
    item_content_hybrid.reset_index(drop=True, inplace=True)

    # Drop the score_x, score_y, normalized_score_content and normalized_score_collab columns
    item_content_hybrid.drop(columns=['score_x', 'score_y', 'normalized_score_content', 'normalized_score_collab'], inplace=True)

    # Rename title_x to title and topics_x to topics
    item_content_hybrid.rename(columns={'title_x': 'title', 'topics_x': 'topics'}, inplace=True)

    if n > 0:
        # Return only the first n articles
        return item_content_hybrid.head(n)

    return item_content_hybrid


# In[1410]:


get_articles_matching_article_from_item_content_hybrid(test_article_id, n=5)


# #### ALS & Item-based

# In[1411]:


item_als_result.head()


# In[1412]:


item_collab_result.head()


# In[1413]:


item_als_hybrid = pd.merge(item_collab_result, item_als_result, on='item_id_adj', how='left')


# In[1414]:


item_als_hybrid.shape


# In[1415]:


item_als_hybrid.isna().sum()


# In[1416]:


item_als_hybrid.head()


# In[1417]:


item_als_hybrid.score_y.value_counts()


# In[1418]:


item_als_hybrid.dropna(inplace=True)


# In[1419]:


item_als_hybrid.head()


# In[1420]:


# Drop title_y and topics_y
item_als_hybrid.drop(columns=['title_y', 'topics_y'], inplace=True)


# In[1421]:


item_als_hybrid.head()


# In[1422]:


# Calculate final score by multiplying normalized_score_collab by 2/3, and normalized_score_als by 1/3, and then adding them together
item_als_hybrid['final_score'] = (item_als_hybrid['normalized_score_collab'] * 2/3) + (item_als_hybrid['normalized_score_als'] * 1/3)


# In[1423]:


# Sort the DataFrame by final_score in descending order
item_als_hybrid.sort_values(by='final_score', ascending=False, inplace=True)


# In[1424]:


item_als_hybrid.head()


# Since ALS results are more diverse, we include them in the final results. However, item-based results are more intuitive, so we give them a higher weightage.

# Expose method

# In[1425]:


def get_articles_matching_article_from_als_item_hybrid(article_id, n=-1):
    item_als_result = get_articles_matching_article_from_als(article_id, all=True)

    # Normalize the scores in item_als_result
    item_als_result['normalized_score_als'] = (item_als_result['score'] - min(item_als_result['score'])) / (max(item_als_result['score']) - min(item_als_result['score']))

    item_collab_result = get_articles_matching_article_from_item_based(article_id)

    # Normalize the scores in item_collab_result
    item_collab_result['normalized_score_collab'] = (item_collab_result['score'] - min(item_collab_result['score'])) / (max(item_collab_result['score']) - min(item_collab_result['score']))

    item_als_hybrid = pd.merge(item_collab_result, item_als_result, on='item_id_adj', how='left')

    item_als_hybrid.dropna(inplace=True)

    # Drop title_y and topics_y
    item_als_hybrid.drop(columns=['title_y', 'topics_y'], inplace=True)

    # Calculate final score by multiplying normalized_score_collab by 2/3, and normalized_score_als by 1/3, and then adding them together
    item_als_hybrid['final_score'] = (item_als_hybrid['normalized_score_collab'] * 2/3) + (item_als_hybrid['normalized_score_als'] * 1/3)

    # Sort the DataFrame by final_score in descending order
    item_als_hybrid.sort_values(by='final_score', ascending=False, inplace=True)

    # Reset the index
    item_als_hybrid.reset_index(drop=True, inplace=True)

    # Drop the score_x, score_y, normalized_score_content and normalized_score_collab columns
    item_als_hybrid.drop(columns=['score_x', 'score_y', 'normalized_score_als', 'normalized_score_collab'], inplace=True)

    # Rename title_x to title and topics_x to topics
    item_als_hybrid.rename(columns={'title_x': 'title', 'topics_x': 'topics'}, inplace=True)

    if n > 0:
        # Return only the first n articles
        return item_als_hybrid.head(n)

    return item_als_hybrid


# In[1426]:


get_articles_matching_article_from_als_item_hybrid(test_article_id, n=5)


# #### ALS & Content-based

# In[1427]:


content_als_hybrid = pd.merge(item_als_hybrid, item_content_result, on='item_id_adj', how='left')


# In[1428]:


content_als_hybrid.shape


# In[1429]:


content_als_hybrid.isna().sum()


# In[1430]:


content_als_hybrid.head()


# Since the algorithm is similar, defining method directly here

# In[1431]:


def get_articles_matching_article_from_als_content_hybrid(article_id, n=-1, ignore=[]):
    item_als_result = get_articles_matching_article_from_als(article_id, all=True)

    # Normalize the scores in item_als_result
    item_als_result['normalized_score_als'] = (item_als_result['score'] - min(item_als_result['score'])) / (max(item_als_result['score']) - min(item_als_result['score']))

    item_content_result = get_articles_matching_article_from_content_based(article_id)

    # Normalize the scores in item_content_result
    item_content_result['normalized_score_content'] = (item_content_result['score'] - min(item_content_result['score'])) / (max(item_content_result['score']) - min(item_content_result['score']))

    content_als_hybrid = pd.merge(item_content_result, item_als_result, on='item_id_adj', how='left')

    content_als_hybrid.dropna(inplace=True)

    # Drop title_y and topics_y
    content_als_hybrid.drop(columns=['title_y', 'topics_y'], inplace=True)

    # If ignore is not empty, drop the rows with item_id_adj in ignore
    if len(ignore) > 0:
        content_als_hybrid = content_als_hybrid[~content_als_hybrid['item_id_adj'].isin(ignore)]

    # Calculate final score by multiplying normalized_score_content by 2/3, and normalized_score_als by 1/3, and then adding them together
    content_als_hybrid['final_score'] = (content_als_hybrid['normalized_score_content'] * 2/3) + (content_als_hybrid['normalized_score_als'] * 1/3)

    # Sort the DataFrame by final_score in descending order
    content_als_hybrid.sort_values(by='final_score', ascending=False, inplace=True)

    # Reset the index
    content_als_hybrid.reset_index(drop=True, inplace=True)

    # Drop the score_x, score_y, normalized_score_als and normalized_score_content columns
    content_als_hybrid.drop(columns=['score_x', 'score_y', 'normalized_score_als', 'normalized_score_content'], inplace=True)

    # Rename title_x to title and topics_x to topics
    content_als_hybrid.rename(columns={'title_x': 'title', 'topics_x': 'topics'}, inplace=True)

    if n > 0:
        # Return only the first n articles
        return content_als_hybrid.head(n)

    return content_als_hybrid


# In[1432]:


get_articles_matching_article_from_als_content_hybrid(test_article_id, n=5)


# Results are as expected.

# ### Final API

# #### Get top 10 articles for a user at the start of the day

# In[1433]:


def get_top_10_articles_for_user(user_id):
    return get_articles_for_user_from_als(user_id, n=10)


# In[1434]:


get_top_10_articles_for_user(test_user_id)


# #### Get more articles for a user when they read an article

# We will be using a hybrid of item-based collaborative filtering and ALS, since the results are intuitive and diverse.

# In[1435]:


def get_articles_read_by_user(user_id):
    return list(txns[txns['consumer_id_adj'] == user_id]['item_id_adj'].values)

def get_more_articles_for_user(article_id, user_id):
    to_filter = get_articles_read_by_user(user_id)

    # Append the article_id to to_filter
    to_filter.append(article_id)

    return get_articles_matching_article_from_als_content_hybrid(article_id, n=10, ignore=to_filter)


# In[1436]:


get_more_articles_for_user(test_article_id, test_user_id)


# In[ ]:





# 

# 

# ### Online evaluation for Item recommendations

# #### Evaluation method

# To check whether the recommendations are good, we will be using the following method:
# 
# If a user has scrolled through at least 75% of an article, we will consider it as a positive interaction. To measure this accurately, we should also keep track of the amount of time in which the user scrolls through the article. When the user clicks on an article and that page opens, we start a timer. The timer is stopped when the user either leaves the page, or has scrolled through 75% of the article.

# #### Further improvements

# To evaluate the results of getting articles similar to another article, we can use the article's topics. If the topics are similar, then the articles are similar.
# 
# In order to further personalize the recommendations, we can use the user's interests. This can also be broken down into a list of topics. The recommended articles should generally have the topics in which the user is interested. When serving articles to the user, we can also keep track of the topics in which the user has read the most articles.
# 
# That being said, diversifying the results is important in order to keep the user engaged. To do this, we could track topics that are similar to a user's favorite topics. If the user has not read many articles in such a similar topic, we can recommend articles from that topic.

# 
