from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_say_hello():
    r = client.get("/")
    assert r.json() == {"greeting": "Hello World"}