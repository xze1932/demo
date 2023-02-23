import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data = pd.read_excel("/Users/zengxuqian/Downloads/Random.xlsx")


def search_people(name, query):
    # Parse the query into separate words and remove stop words
     # Tokenize the query into separate words
    words = query.split()
    
    # Remove stop words from the list of words
    stop_words = ['and', 'the', 'of', 'in', 'to', 'a', 'for', 'with', 'that', 'on']
    words = [word for word in words if word not in stop_words]
    query_words = ''.join(words)
    
    # Create a boolean mask indicating which names contain the query words
    #mask = df['user_name'].str.contains(''.join(query_words), case=False)

    # Filter the DataFrame to only include rows containing the query input in any column
    filtered_df = data[data['user_name'].str.contains(query_words, case=False) | 
                     data['workplace'].str.contains(query_words, case=False) | 
                     data['user_department'].str.contains(query_words, case=False) | 
                     data['Personality_Color'].str.contains(query_words, case=False)]
    
    # add the searcher info  
    searcher = data[data['user_name'] == name].iloc[0]
    filtered_df = filtered_df.append(searcher, ignore_index=True)

    # Create a dictionary to store the similarity scores for each column
    similarity_scores = {'user_name': 0, 'workplace': 0, 'user_department': 0, 'Personality_Color': 0}
    
    # Loop over the columns and calculate the similarity scores
    for col in similarity_scores.keys():
        for query_word in query_words:
            similarity_scores[col] += filtered_df[col].str.contains(query_word, case=False).sum()
    
    # Sort the dictionary by the similarity scores
    sorted_similarity_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Pick the top three columns
    top_col = ''
    top_col_ = [col[0] for col in sorted_similarity_scores[:1]]
    for x in top_col_:
        top_col += x

    # One-hot encode categorical features
    one_hot = pd.get_dummies(filtered_df[['user_department', 'workplace','Personality_Color']])
    
    # Create the feature matrix
    features = np.concatenate([filtered_df[['Sensitive-Secure', 'Challenging-Friendly']].to_numpy(), one_hot.to_numpy()], axis=1)

    # Calculate the cosine similarity matrix
    cosine_sim = cosine_similarity(features)
    
    suffix = data[data['user_name'] == name]['user_department'].astype('string').values[0]
    col_name = 'user_department_' + suffix
  
    # make the department feature the most important by multiplying its cosine similarity values by a factor
    #cosine_sim[:, department_index] = cosine_sim[:, department_index] * 10
    for i in range(cosine_sim.shape[0]):
      cosine_sim[i, i] = cosine_sim[i, i] + (one_hot.iloc[i][col_name] * 2)

    index = filtered_df[filtered_df['user_name'] == name].index[0]
    similarities = cosine_sim[index, :]
    top_10_indices = similarities.argsort()[-11:-1][::-1]
    top_10_users = filtered_df.iloc[top_10_indices][top_col]
    return top_10_users
#<form action="http://127.0.0.1:5000/search_people" method="GET"></form>

#results-container select::-webkit-scrollbar {
  display: none;
}
#隐藏滚动条

#results-container select {
  overflow: -moz-scrollbars-none;
}



        #results-container {
            position: absolute;
            top: 40px; /* 根据搜索框的高度进行调整 */
            left: 200px;
            right: 0;
            background-color: rgb(129, 129, 121);
            border: 1px solid gray;
            z-index: 1; /* 使搜索结果在导航栏的上方 */
        }