from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
eldf_final = pickle.load(open('eldf_final.pkl', 'rb'))
final_ratings_matrix = pickle.load(open('final_ratings_matrix.pkl', 'rb'))

def recommend_products(user_id, purchased_products, final_ratings_matrix):
    purchased_products = eldf_final[eldf_final['userId'] == user_id]['productId'].tolist()
    user_ratings = final_ratings_matrix.loc[user_id]
    filtered_ratings = user_ratings[~user_ratings.index.isin(purchased_products)]
    sorted_ratings = filtered_ratings.sort_values(ascending=False)
    return sorted_ratings.head(5).index.tolist()

@app.route('/')    
def home():
     
     return render_template('index.html')

@app.route('/users', methods=['GET', 'POST'])
def user():
    if request.method == 'POST':
        user_id = request.values.get('UserId')
        purchased_products = eldf_final[eldf_final['userId'] == user_id]['productId'].tolist()
        recommended_products = recommend_products(user_id, purchased_products, final_ratings_matrix)
        return render_template('user.html', values=recommended_products, userid=user_id)
    else:
      
         return render_template('user.html') 
    
    
@app.route('/result',methods=['GET', 'POST'])
def result():
    
    output = model.to_html()
    return render_template('result.html',result=output)

if __name__=="__main__":
    
    app.run(debug=True)