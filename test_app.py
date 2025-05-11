def test_home():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Sentiment Classifier API is up!"}
