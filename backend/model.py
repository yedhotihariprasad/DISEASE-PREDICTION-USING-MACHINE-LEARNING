import os
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE, 'backend', 'model.h5')
LABELS_PATH = os.path.join(BASE, 'backend', 'labels.txt')


class MangoModel:
    def __init__(self):
        self.model = None
        self.labels = []
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model(MODEL_PATH)
            except Exception:
                self.model = None
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                self.labels = [l.strip() for l in f.readlines() if l.strip()]
        if not self.labels:
            self.labels = ['healthy', 'anthracnose', 'powdery_mildew']

    def _prepare(self, pil_img, target_size=(224, 224)):
        img = pil_img.convert('RGB').resize(target_size)
        arr = np.array(img).astype('float32') / 255.0
        return np.expand_dims(arr, 0)

    def predict_file(self, file_storage):
        img = Image.open(BytesIO(file_storage.read()))
        return self.predict_pil(img)

    def predict_pil(self, pil_img, top_k=3):
        """Return top_k predictions as list of (label, prob).

        If model not loaded, returns a single high-confidence 'healthy' prediction.
        """
        if self.model is None:
            return [(self.labels[0], 0.99)]
        x = self._prepare(pil_img)
        preds = self.model.predict(x)[0]
        # get top k indices
        top_k = min(top_k, len(preds))
        inds = list(np.argsort(preds)[::-1][:top_k])
        return [(self.labels[i], float(preds[i])) for i in inds]
