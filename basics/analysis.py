#!/usr/bin/env python
# coding: utf-8

# ### Intro

# Data analysis for the NLP capstone project of the Upgrad Data Science course.

# Code committed to: https://github.com/kavurisrikanth/news-recommender-capstone

txns = None
cnt = None

load_done = False

# Helpers

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

def to_rating(val):
    # #### Introduce a ratings column
    if val == 'content_followed':
        return 5
    if val == 'content_commented_on':
        return 4
    if val == 'content_saved':
        return 3
    if val == 'content_liked':
        return 2
    return 1

def get_articles_that_user_liked(user_id):
    # For this, we get all the articles that user has given a rating of more than the average rating
    # Get the average rating for the user
    avg = txns[txns['consumer_id_adj'] == user_id].rating.mean()

    user_interactions = txns[(txns['consumer_id_adj'] == user_id) & (txns['rating'] > avg)].sort_values(by='rating', ascending=False)

    if (len(user_interactions) == 0):
        user_interactions = txns[(txns['consumer_id_adj'] == user_id)].sort_values(by='rating', ascending=False)

    return user_interactions[['item_id_adj', 'rating']]

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

# Main methods

def load():
    # ### The Basics - Loading data
    import pandas as pd
    import numpy as np

    import plotly.express as px

    txns = pd.read_csv('../data/consumer_transanctions.csv')
    cnt = pd.read_csv('../data/platform_content.csv')

    return txns, cnt

def prepare(txns, cnt):
    from gensim.utils import simple_preprocess
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    # ### Data preparation
    # #### Drop unnecessary columns
    # Drop country, consumer_location, consumer_device_info, consumer_session_id from txns
    txns.drop(columns=['country', 'consumer_location', 'consumer_device_info', 'consumer_session_id'], inplace=True)

    # Drop producer_id, producer_session_id, producer_device_info, producer_location, producer_country, item_type from cnt
    cnt.drop(columns=['producer_id', 'producer_session_id', 'producer_device_info', 'producer_location', 'producer_country', 'item_type'], inplace=True)

    content = cnt

    # #### Remove all docs that are not in English

    content.language.value_counts()

    content = content[content['language'] == 'en']

    # #### Handle articles with duplicated entries
    no_dups = content.sort_values('event_timestamp').drop_duplicates(subset=['title', 'text_description'], keep='last')

    no_dups.reset_index(inplace=True)

    cnt = no_dups

    # Introduce rating
    txns['rating'] = txns.interaction_type.apply(lambda x: to_rating(x))

    # #### Introduce keywords
    cnt['text_description_preprocessed'] = cnt['text_description'].apply(lambda x: simple_preprocess(x, deacc=True))

    stopwords_en = stopwords.words('english')

    cnt['text_description_no_stopwords'] = cnt['text_description_preprocessed'].apply(lambda x: [word for word in x if word not in stopwords_en])

    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    cnt['text_description_lemmatized'] = cnt['text_description_no_stopwords'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # Drop the columns we don't need anymore
    cnt.drop(['text_description_preprocessed', 'text_description_no_stopwords'], axis=1, inplace=True)
    
    return txns, cnt

def adjust_ids(txns, cnt):
    # #### Adjust IDs
    # The user and document IDs in the data make no sense. So create new IDs that start from 1.
    consumer_helper = IdHelper()
    item_helper = IdHelper()

    txns['consumer_id_adj'] = txns['consumer_id'].map(lambda x: consumer_helper.translate(x))

    txns['item_id_adj'] = txns['item_id'].map(lambda x: item_helper.translate(x))

    # Drop item_id and consumer_id from txns
    txns.drop(columns=['item_id', 'consumer_id'], inplace=True)

    # Same for content.
    cnt['item_id_adj'] = cnt['item_id'].map(lambda x: item_helper.translate(x))

    # Drop item_id from cnt
    cnt.drop(columns=['item_id'], inplace=True)

    return txns, cnt

def adjust_ratings(txns, cnt):
    # ### EDA
    # #### Checking for missing values
    # #### Checking for duplicated ratings
    import pandas as pd

    txns_2 = txns[['consumer_id_adj', 'item_id_adj', 'rating']]

    duplicates = txns[txns.duplicated(subset=['consumer_id_adj', 'item_id_adj'], keep=False)]

    duplicates.sort_values(by=['consumer_id_adj', 'item_id_adj', 'rating'], inplace=True)

    # There are duplicated entries i.e., the same user has interacted with the same article multiple times.
    # Since multiple interactions could mean that a user liked an article, the duplicates must be considered in the analysis.
    # #### For "duplicated" transactions, calculate the average rating of the user for that article

    grp = duplicates.groupby(by=['consumer_id_adj', 'item_id_adj'])['rating'].mean()

    grp_df = pd.DataFrame(grp)

    # Renaming the rating column to avoid any potential clash when merged with the original

    grp_df.columns = ['rating_sum']

    grp_df.reset_index(inplace=True)

    # Check distributions of ratings
    # fig = px.box(grp_df, y='rating_sum')
    # fig.show()


    # A majority of the articles are rated 2 or lower. Only a very small number of transactions have a high rating. However, these are not outliers. This is expected, as users would only like a small percentage of the articles in the system.

    # #### Add the adjusted rating back to the original transactions DataFrame
    no_dups = txns.drop_duplicates(subset=['consumer_id_adj', 'item_id_adj'])

    no_dups.sort_values(by=['consumer_id_adj', 'item_id_adj', 'rating'], inplace=True)

    # Merge the two DataFrames
    txns_merged = pd.merge(left=no_dups, right=grp_df, left_on=['consumer_id_adj', 'item_id_adj'], right_on=['consumer_id_adj', 'item_id_adj'], how='left')

    txns_merged.sort_values(by=['consumer_id_adj', 'item_id_adj', 'rating'], inplace=True)

    # Rows that have rating_sum as NaN were not duplicated in the original. So, the summed rating would just be the rating for these rows.
    txns_merged['ratings_merged'] = txns_merged.rating_sum.fillna(txns_merged.rating)

    # txns_merged.ratings_merged.describe()
    # The rating is between 1 and 5, so that is good enough.

    txns_merged.drop(columns=['rating_sum'], inplace=True)

    txns_merged.rename(columns={'rating': 'rating_original'}, inplace=True)

    # txns_merged.rename(columns={'ratings_scaled': 'rating'}, inplace=True)
    txns_merged.rename(columns={'ratings_merged': 'rating'}, inplace=True)

    txns = txns_merged

    txns.drop(columns=['interaction_type'], inplace=True)

    # Consolidated Ratings are between 1 and 4.5, which is expected.

    return txns, cnt

def do_topic_modeling(txns, cnt):
    # ### Topic Modelling

    # Try to create some basic topics under which each article may be categorized

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    import pandas as pd
    import numpy as np

    # #### Feature extraction

    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(cnt['text_description'])

    # #### NMF Decomposition
    num_topics = 10
    nmf = NMF(n_components=num_topics, random_state=42)
    doc_topic = nmf.fit_transform(X)
    topic_term = nmf.components_

    # Getting the top 10 words for each topic

    words = np.array(vec.get_feature_names())
    topic_words = pd.DataFrame(
        np.zeros((num_topics, 10)),
        index=['topic_{}'.format(i + 1) for i in range(num_topics)],
        columns=['word_{}'.format(i + 1) for i in range(10)]
    ).astype(str)

    # Populating topic_words

    for i in range(num_topics):
        idx = topic_term[i].argsort()[::-1][:10]
        topic_words.iloc[i] = words[idx]

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

    doc_topic_df = pd.DataFrame(doc_topic, columns=['topic_{}'.format(i + 1) for i in range(num_topics)])

    # Get the 5 topics with the highest probabilities for each document
    doc_topic_df['top_topics'] = doc_topic_df.apply(lambda x: x.sort_values(ascending=False).index[:5].tolist(), axis=1)

    # Get the mapping for doc_topic_df.top_topics from topic_mapping and create a new column
    doc_topic_df['top_topics_mapped'] = doc_topic_df.top_topics.apply(lambda x: [topic_mapping[i] for i in x])

    # Add doc_topic_df.top_topics_mapped to cnt
    cnt = pd.concat([cnt, doc_topic_df.top_topics_mapped], axis=1)

    # Rename cnt.top_topics_mapped to cnt.topics
    cnt.rename(columns={'top_topics_mapped': 'topics'}, inplace=True)

    return txns, cnt

def create_and_populate_user_article_matrix(data):
    import numpy as np
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

def main():
    if not load_done:
        txns, cnt = load()

        txns, cnt = prepare(txns, cnt)

        txns, cnt = adjust_ids(txns, cnt)

        txns, cnt = adjust_ratings(txns, cnt)

        txns, cnt = do_topic_modeling(txns, cnt)

        # With this, we have some idea of what topics each article is talking about.



        load_done = True



from math import sqrt

# ### User-based collaborative filtering
# ### Item-based collaborative filtering
class Collaborative:
    train = None
    test = None
    article_pred_df = None
    other_article_pred_df = None

    def __init__(self, txns):
        import sklearn
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import accuracy_score
        from sklearn.metrics.pairwise import pairwise_distances

        import numpy as np
        import pandas as pd
        # ### Train test split
        self.train, self.test = sklearn.model_selection.train_test_split(txns, test_size=0.3, random_state=42)

        # ### User-Article matrix
        # Since this is collaborative filtering, we will consider the transactions matrix. From this, we construct a matrix of the ratings given by users for each product.
        # Populate the training matrix
        # Fill the training matrix with rating values
        data_matrix = create_and_populate_user_article_matrix(self.train)

        # Populate the testing matrix
        data_matrix_test = create_and_populate_user_article_matrix(self.test)

        # ### Pairwise Distance
        user_similarity = 1 - pairwise_distances(data_matrix, metric='cosine')

        # Take the transpose of the data matrix in order to calculate the article similarity. Will be used later.
        self.article_similarity = 1 - pairwise_distances(data_matrix.T, metric='cosine')

        # ### Get dot product of data matrix with similarity matrix
        article_prediction = np.dot(user_similarity, data_matrix_test)

        self.article_pred_df = pd.DataFrame(article_prediction)

    def get_articles_for_user_from_user_based(self, user_id, n=-1):
        import pandas as pd
        user_idx = user_id - 1

        recommendation = pd.DataFrame(self.article_pred_df.iloc[user_idx].sort_values(ascending=False))

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

    def get_articles_matching_article_from_item_based(self, article_id, n=-1, all=False):
        import pandas as pd
        article_idx = article_id - 1

        out = pd.DataFrame(self.article_similarity[article_idx], columns=['score'])

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

# ## Getting articles for a User

# Consider user-based collaborative filtering, and ALS. Whichever gives the best result would be the model to use.

# ### User-based collaborative filtering
n_users = txns.consumer_id_adj.nunique()

n_articles = txns.item_id_adj.nunique()

print(f'Num users: {n_users}, Num articles: {n_articles}')


# ## Alternating Least Squares method
class ALS:
    best_user_based_als = None
    sparse_user_article = None

    def __init__(self):
        from scipy.sparse import csr_matrix

        # #### Create sparse User-Article matrix
        # Random values in CSR matrix will be filled with alpha value
        keep = ['consumer_id_adj', 'item_id_adj', 'rating']

        txns_mod = txns.drop(columns=[col for col in txns.columns if col not in keep])

        alpha = 40

        self.sparse_user_article = csr_matrix( ([alpha]*txns_mod.shape[0], (txns_mod['consumer_id_adj'], txns_mod['item_id_adj']) ))

        # Matrix dimensions match with the number of users & articles, accounting for the extra row at index 0
        # Convert to array
        csr_user_array = self.sparse_user_article.toarray()

        # Dimensions match with the matrix
        # Create article-user sparse matrix
        self.sparse_article_user = self.sparse_user_article.T.tocsr()

        # Shape matches
        # csr_article_array = sparse_article_user.toarray()

        self.train_user()

        self.train_item()

    def train_user(self):
        from implicit.evaluation import train_test_split
        from implicit.als import AlternatingLeastSquares

        # #### Create train & test data
        train, test = train_test_split(self.sparse_user_article, train_percentage=0.8)

        # #### Building the ALS Model
        # Got best params from tuning
        # precision@k = 0.144
        best_user_based_f = 92
        best_user_based_r = 0.115
        best_user_based_i = 40

        self.best_user_based_als = AlternatingLeastSquares(
            factors=best_user_based_f, 
            regularization=best_user_based_r, 
            iterations=best_user_based_i, 
            calculate_training_loss=False
        )
        self.best_user_based_als.fit(train)

    def train_item(self):
        from implicit.evaluation import train_test_split
        from implicit.als import AlternatingLeastSquares
        item_train, item_test = train_test_split(self.sparse_article_user, train_percentage=0.8, random_state=42)

        best_article_als_f = 20
        best_article_als_r = 1.2
        best_article_als_i = 120

        self.best_item_als = AlternatingLeastSquares(
            factors=best_article_als_f, 
            regularization=best_article_als_r, 
            iterations=best_article_als_i, 
            calculate_training_loss=False
        )
        self.best_item_als.fit(item_train)

    def get_articles_for_user_from_als(self, user_id, n=20):
        import pandas as pd
        id, scores = self.best_user_based_als.recommend(user_id, self.sparse_user_article[user_id], N=50, filter_already_liked_items=True)

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

    def get_articles_matching_article_from_als(self, article_id, n=20, all=False):
        import pandas as pd
        ids, scores = self.best_item_als.similar_items(
            article_id, item_users=self.sparse_article_user, N=50 if not all else n_articles)

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

# Check if ALS does better.
# Expose method to get recommendations for a user

# ## Getting articles matching another article

# Consider item-based collaborative filtering and content-based filtering



# Use article_similarity matrix constructed earlier

# ### ALS for Articles
# Use sparse_article_user created earlier

# After hyperparameter tuning

# Define a function to get the article title from the article id
def get_article_title(article_id):
    # If the article id is not in the article dataframe, log that it is missing
    if article_id not in cnt['item_id_adj'].values:
        print('Missing article id: ', article_id)
        return None
    return cnt[cnt['item_id_adj'] == article_id]['title'].values[0]

def get_article_topics(article_id):
    # If the article id is not in the article dataframe, log that it is missing
    if article_id not in cnt['item_id_adj'].values:
        print('Missing article id: ', article_id)
        return None
    return cnt[cnt['item_id_adj'] == article_id]['topics'].values[0]



# Expose method

# ### Content-based filtering
# #### Derive keywords from the article text
# Join cnt.text_description_lemmatized into a single list
words_list = []
for doc in cnt.text_description_lemmatized:
    words_list.append(doc)

# #### Create Dictionary, Bag of Words, tfidf model & Similarity matrix
from gensim.corpora.dictionary import Dictionary

# create a dictionary from words list
dictionary = Dictionary(words_list)

number_words = 0
for word in words_list:
    number_words = number_words + len(word)

# ##### Generating Bag of Words
bow = dictionary.doc2bow(words_list[0])

# Some words are repeated
# ##### Generating a corpus
#create corpus where the corpus is a bag of words for each document
corpus = [dictionary.doc2bow(doc) for doc in words_list] 

# All the articles are in the corpus, and the length of the first matches the count in the Bag of Words above
# ##### Use the TfIdf model on the corpus
from gensim.models.tfidfmodel import TfidfModel

#create tfidf model of the corpus
tfidf = TfidfModel(corpus) 

# Again, the lengths are matched
# ##### Generate Similarity matrix
from gensim.similarities import MatrixSimilarity

# Create the similarity matrix. This is the most important part where we get the similarities between the movies.
sims = MatrixSimilarity(tfidf[corpus], num_features=len(dictionary))

# Flatten words_list into a set of unique words
words_set = set([word for doc in words_list for word in doc])

len(set(words_set))

len(sims[corpus[0]])

len(sims)

# #### Generating recommendations
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


get_articles_matching_article_from_content_based(test_article_id, n=10)


# 
# #### Comparing item-based and content-based filtering

num_articles = len(collab_out)

num_articles

# Assign the first num_articles rows from recs_df to content_out
content_out = recs_df.iloc[:num_articles]

content_out.reset_index(inplace=True)

content_out.head()

cnt[cnt['item_id_adj'] == test_article_id][['title', 'topics']]

# Rename index to article_id
content_out.rename(columns={'item_id_adj': 'article_id'}, inplace=True)
content_out.drop(columns=['index'], inplace=True)

content_out.head()

collab_out.head()

# Left join the content_out and collab_out DataFrames on article_id
out = pd.merge(collab_out, content_out, on='article_id', how='left')

content_out

collab_out

out

content_out.shape


# **************************** HYBRID RECOMMENDATIONS ****************************
# There isn't much overlap between the item-based collaborative, content-based, and ALS results.
# Check if combining with ALS improves the results

# ### Combining item-based filterings

item_als_result = get_articles_matching_article_from_als(test_article_id, n=50, all=True)

item_als_result.shape

item_collab_result = get_articles_matching_article_from_item_based(test_article_id)

item_collab_result.shape

item_content_result = get_articles_matching_article_from_content_based(test_article_id)

item_content_result.shape


# #### Normalizing the similarity scores using Min-Max normalization

# Normalize the scores in item_als_result
item_als_result['normalized_score_als'] = (item_als_result['score'] - min(item_als_result['score'])) / (max(item_als_result['score']) - min(item_als_result['score']))

min(item_als_result['score'])

item_als_result.head()

# Normalize the scores in item_collab_result
item_collab_result['normalized_score_collab'] = (item_collab_result['score'] - min(item_collab_result['score'])) / (max(item_collab_result['score']) - min(item_collab_result['score']))

min(item_collab_result['score'])

item_collab_result.head()

# Normalize the scores in item_content_result
item_content_result['normalized_score_content'] = (item_content_result['score'] - min(item_content_result['score'])) / (max(item_content_result['score']) - min(item_content_result['score']))

min(item_content_result['score'])

item_content_result.head()


# #### Item-based & content-based

item_collab_result.head()

item_content_result.head()

item_content_hybrid = pd.merge(item_content_result, item_collab_result, on='item_id_adj', how='left')

item_content_hybrid.shape

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
