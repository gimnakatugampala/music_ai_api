from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('D:\HNDSE\Enterpunership Final Project\GitHub\music_ai_api-main\music_ai_api-main\Model\lyrics_genre_classifier.pkl')

@app.route('/classify-lyrics', methods=['POST'])
def classify_lyrics():
    data = request.json
    lyrics = data['lyrics']
    
    # Predict the genre or mood
    genre = model.predict([lyrics])[0]
    
    return jsonify({'genre': genre})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
