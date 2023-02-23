from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler


app = Flask(__name__)
CORS(app)


@app.route('/search_people/', methods=['GET'])
def search_people_route():
   query = request.args.get('query')
   category = request.args.get('category') 

    # Parse the query into separate words and remove stop words
     # Tokenize the query into separate words
   name = 'Mary'
   words = query.split()
   query_words = ''.join(words)
   
   data = pd.read_excel("/Users/zengxuqian/Downloads/Random.xlsx")
   data = data.fillna("Unknown")
    
    # Filter the DataFrame to only include rows containing the query input in any column
   if category == 'workplace':
      filtered_df = data[data['workplace'].str.contains(query_words, case=False)]
   elif category == 'user_department':
      filtered_df = data[data['user_department'].str.contains(query_words, case=False)]
   elif category == 'Personality_Color':
      filtered_df = data[data['Personality_Color'].str.contains(query_words, case=False)]
   else:
      filtered_df = data[data['user_name'].str.contains(query_words, case=False)]
    
    # add the searcher info  
   searcher = data[data['user_name'] == name].iloc[0]
   filtered_df = filtered_df.append(searcher, ignore_index=True)

   ''' # Create a dictionary to store the similarity scores for each column
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
        top_col += x'''

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
   top_10_indices = similarities.argsort()[-8:-1][::-1]
   top_10_users = filtered_df.iloc[top_10_indices][category]
   result = top_10_users.tolist()
   return jsonify(result)

   #return render_template('appc.html', results=results)



if __name__ == '__main__':
    app.run(debug=True)


