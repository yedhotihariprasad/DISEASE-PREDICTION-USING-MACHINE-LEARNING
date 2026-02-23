from flask import Flask, request, jsonify, render_template
import os
from model import MangoModel

app = Flask(__name__)  # REMOVE static_folder='../frontend'

model = MangoModel()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'no image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400

    try:
        top_k = int(request.args.get('top_k', 3))
        preds = model.predict_file(file)

        if isinstance(preds, list):
            return jsonify({
                'predictions': [
                    {'label': l, 'confidence': p} for l, p in preds
                ]
            })

        label, prob = preds
        return jsonify({
            'predictions': [
                {'label': label, 'confidence': float(prob)}
            ]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)