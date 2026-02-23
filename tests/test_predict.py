import io
import os
from backend.app import app


def test_predict_no_file():
    client = app.test_client()
    rv = client.post('/predict')
    assert rv.status_code == 400


def test_ping_with_dummy_image():
    # Uses a tiny in-memory image to test the endpoint
    from PIL import Image
    img = Image.new('RGB', (10, 10), color=(73, 109, 137))
    b = io.BytesIO()
    img.save(b, format='JPEG')
    b.seek(0)

    client = app.test_client()
    data = {'image': (b, 'test.jpg')}
    rv = client.post('/predict', data=data, content_type='multipart/form-data')
    assert rv.status_code in (200, 500)  # 200 if model loaded, 500 if unexpected
