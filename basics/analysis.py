#!/usr/bin/env python
# coding: utf-8

# ### Intro

# Data analysis for the NLP capstone project of the Upgrad Data Science course.

# Code committed to: https://github.com/kavurisrikanth/news-recommender-capstone

class GlobalData:
    txns = None
    cnt = None

    collab = None
    content_based = None
    als = None

    load_done = False

    consumer_id_helper = None
    item_id_helper = None

    def onload(self):
        self.n_users = self.txns.consumer_id_adj.nunique()
        self.n_articles = self.txns.item_id_adj.nunique()

    # ### Final API
    # #### Get top 10 articles for a user at the start of the day
    def get_top_10_articles_for_user(self, user_id):
        return self.get_articles_for_user_from_als(user_id, n=10)


    # #### Get more articles for a user when they read an article
    # We will be using a hybrid of item-based collaborative filtering and ALS, since the results are intuitive and diverse.
    def get_articles_read_by_user(self, user_id):
        return list(self.txns[self.txns['consumer_id_adj'] == user_id]['item_id_adj'].values)

    def get_more_articles_for_user(self, article_id, user_id):
        to_filter = self.get_articles_read_by_user(user_id)

        # Append the article_id to to_filter
        to_filter.append(article_id)

        return self.get_articles_matching_article_from_als_content_hybrid(article_id, n=10, ignore=to_filter)


# ### User-based collaborative filtering
# ### Item-based collaborative filtering
class Collaborative:
    train = None
    test = None
    article_pred_df = None
    other_article_pred_df = None

    def __init__(self, data):
        import sklearn
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import accuracy_score
        from sklearn.metrics.pairwise import pairwise_distances

        import numpy as np
        import pandas as pd

        self.data = data

        # ### Train test split
        self.train, self.test = sklearn.model_selection.train_test_split(data.txns, test_size=0.3, random_state=42)

        # ### User-Article matrix
        # Since this is collaborative filtering, we will consider the transactions matrix. From this, we construct a matrix of the ratings given by users for each product.
        # Populate the training matrix
        # Fill the training matrix with rating values
        data_matrix = create_and_populate_user_article_matrix(self.train, data.n_users, data.n_articles)

        # Populate the testing matrix
        data_matrix_test = create_and_populate_user_article_matrix(self.test, data.n_users, data.n_articles)

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

        merged = pd.merge(recommendation, self.data.cnt, left_on='article_id', right_on='item_id_adj', how='left')

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
        merged = out.merge(self.data.cnt, how='left', on='item_id_adj')

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

# ## Alternating Least Squares method
class ALS:
    best_user_based_als = None
    sparse_user_article = None

    def __init__(self, data):
        self.data = data
        txns = self.data.txns

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
        merged = out.merge(self.data.cnt, how='left', on='item_id_adj')

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
            article_id, item_users=self.sparse_article_user, N=50 if not all else self.data.n_articles)

        out = pd.DataFrame({'item_id_adj': ids, 'score': scores})

        merged = pd.merge(out, self.data.cnt, how='left', on='item_id_adj')

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

class ContentBased:
    def __init__(self, data):
        # ### Content-based filtering
        # #### Derive keywords from the article text
        # Join cnt.text_description_lemmatized into a single list
        self.data = data
        cnt = data.cnt
        words_list = []
        for doc in cnt.text_description_lemmatized:
            words_list.append(doc)

        # #### Create Dictionary, Bag of Words, tfidf model & Similarity matrix
        from gensim.corpora.dictionary import Dictionary

        # create a dictionary from words list
        self.dictionary = Dictionary(words_list)

        number_words = 0
        for word in words_list:
            number_words = number_words + len(word)

        # ##### Generating Bag of Words
        bow = self.dictionary.doc2bow(words_list[0])

        # Some words are repeated
        # ##### Generating a corpus
        #create corpus where the corpus is a bag of words for each document
        corpus = [self.dictionary.doc2bow(doc) for doc in words_list] 

        # All the articles are in the corpus, and the length of the first matches the count in the Bag of Words above
        # ##### Use the TfIdf model on the corpus
        from gensim.models.tfidfmodel import TfidfModel

        #create tfidf model of the corpus
        self.tfidf = TfidfModel(corpus) 

        # Again, the lengths are matched
        # ##### Generate Similarity matrix
        from gensim.similarities import MatrixSimilarity

        # Create the similarity matrix. This is the most important part where we get the similarities between the movies.
        self.sims = MatrixSimilarity(self.tfidf[corpus], num_features=len(self.dictionary))

    # #### Generating recommendations
    def article_recommendation(self, content):
        import pandas as pd
        # get a bag of words from the content
        query_doc_bow = self.dictionary.doc2bow(content) 

        #convert the regular bag of words model to a tf-idf model
        query_doc_tfidf = self.tfidf[query_doc_bow] 

        # get similarity values between input movie and all other movies
        similarity_array = self.sims[query_doc_tfidf] 

        #Convert to a Series
        similarity_series = pd.Series(similarity_array.tolist(), index=self.data.cnt['item_id_adj']) 

        #get the most similar movies 
        # similarity_output = similarity_series.sort_values(ascending=False)
        similarity_output = similarity_series
        
        return similarity_output

    def get_articles_matching_article_from_content_based(self, article_id, n=-1):
        import pandas as pd
        lemmatized_desc = self.data.cnt[self.data.cnt['item_id_adj'] == article_id]['text_description_lemmatized'].values[0]

        recommendations = self.article_recommendation(lemmatized_desc)

        recommendations_df = pd.DataFrame(recommendations, columns=['score'])

        recommendations_df.reset_index(inplace=True)

        recommendations_df = self.data.cnt.merge(recommendations_df, on='item_id_adj', how='left')

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

# Helpers
class IdHelper:
    _map = {}
    _id = 1
    _ids = []

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
        self._ids.append(num)
        return num

    def is_valid_id(self, id):
        return id in self._map

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

# Main methods
def load(data):
    # ### The Basics - Loading data
    import pandas as pd
    import numpy as np

    import plotly.express as px

    data.txns = pd.read_csv('../data/consumer_transanctions.csv')
    data.cnt = pd.read_csv('../data/platform_content.csv')

def prepare(data):
    from gensim.utils import simple_preprocess
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    txns = data.txns
    cnt = data.cnt

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
    
    data.txns = txns
    data.cnt = cnt

def adjust_ids(data):
    # #### Adjust IDs
    # The user and document IDs in the data make no sense. So create new IDs that start from 1.
    consumer_helper = IdHelper()
    item_helper = IdHelper()

    data.consumer_id_helper = consumer_helper
    data.item_id_helper = item_helper

    txns = data.txns
    cnt = data.cnt

    txns['consumer_id_adj'] = txns['consumer_id'].map(lambda x: consumer_helper.translate(x))

    txns['item_id_adj'] = txns['item_id'].map(lambda x: item_helper.translate(x))

    # Drop item_id and consumer_id from txns
    txns.drop(columns=['item_id', 'consumer_id'], inplace=True)

    # Same for content.
    cnt['item_id_adj'] = cnt['item_id'].map(lambda x: item_helper.translate(x))

    # Drop item_id from cnt
    cnt.drop(columns=['item_id'], inplace=True)

def adjust_ratings(data):
    # ### EDA
    # #### Checking for missing values
    # #### Checking for duplicated ratings
    import pandas as pd

    txns = data.txns
    cnt = data.cnt

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

    data.txns = txns
    data.cnt = cnt

def do_topic_modeling(data):
    # ### Topic Modelling

    # Try to create some basic topics under which each article may be categorized

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    import pandas as pd
    import numpy as np

    # #### Feature extraction

    cnt = data.cnt

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

    data.cnt = cnt

def create_and_populate_user_article_matrix(data, n_users, n_articles):
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

def get_data():
    data = GlobalData()

    if not data.load_done:
        print('*** Loading data...')
        print('')
        load(data)

        print('*** Preparing data...')
        print('')
        prepare(data)

        print('*** Adjusting IDs...')
        print('')
        adjust_ids(data)

        data.onload()

        print('*** Adjusting ratings...')
        print('')
        adjust_ratings(data)

        print('*** Performing Topic Modeling...')
        print('')
        do_topic_modeling(data)

        print('*** Building models...')
        print('')
        data.collab = Collaborative(data)

        data.als = ALS(data)

        data.content_based = ContentBased(data)

        data.load_done = True

    return data

# **************************** HYBRID RECOMMENDATIONS ****************************
# There isn't much overlap between the item-based collaborative, content-based, and ALS results.
# Check if combining with ALS improves the results

# ### Combining item-based filterings

class Hybrid:
    def __init__(self, data):
        self.data = data

    def get_articles_matching_article_from_item_content_hybrid(self, article_id, n=-1, ignore=[]):
        import pandas as pd
        item_collab_result = self.data.collab.get_articles_matching_article_from_item_based(article_id)

        # Normalize the scores in item_collab_result
        item_collab_result['normalized_score_collab'] = (item_collab_result['score'] - min(item_collab_result['score'])) / (max(item_collab_result['score']) - min(item_collab_result['score']))

        item_content_result = self.data.content_based.get_articles_matching_article_from_content_based(article_id)

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

    def get_articles_matching_article_from_als_item_hybrid(self, article_id, n=-1):
        import pandas as pd
        item_als_result = self.data.als.get_articles_matching_article_from_als(article_id, all=True)

        # Normalize the scores in item_als_result
        item_als_result['normalized_score_als'] = (item_als_result['score'] - min(item_als_result['score'])) / (max(item_als_result['score']) - min(item_als_result['score']))

        item_collab_result = self.data.collab.get_articles_matching_article_from_item_based(article_id)

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

    def get_articles_matching_article_from_als_content_hybrid(self, article_id, n=-1, ignore=[]):
        import pandas as pd
        item_als_result = self.data.als.get_articles_matching_article_from_als(article_id, all=True)

        # Normalize the scores in item_als_result
        item_als_result['normalized_score_als'] = (item_als_result['score'] - min(item_als_result['score'])) / (max(item_als_result['score']) - min(item_als_result['score']))

        item_content_result = self.data.content_based.get_articles_matching_article_from_content_based(article_id)

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
