from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_say_hello():
    r = client.get("/")

    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World"}


def test_low_income():
    r = client.post(
        "/inference",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json"
        },
        json={
            "age": 39,
            "workclass": "Private",
            "fnlgt": 77516,
            "education": "Masters",
            "education_num": 14,
            "marital_status": "Divorced",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
        },
    )

    assert r.status_code == 200
    assert r.json() == {'pred': " <=50K"}


def test_high_income():
    r = client.post(
        "/inference",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json"
        },
        json={
            "age": 71,
            "workclass": " ?",
            "fnlgt": 287372,
            "education": " Doctorate",
            "education_num": 16,
            "marital_status": " Married-civ-spouse",
            "occupation": " ?",
            "relationship": " Husband",
            "race": " White",
            "sex": " Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 10,
            "native_country": " United-States"
        },
    )

    assert r.status_code == 200
    assert r.json() == {'pred': ' >50K'}
