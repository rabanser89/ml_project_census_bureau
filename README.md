# ML PROJECT CENSUS BUREAU

In this project an ML model is trained to predict income of citizens.
The data can be found in ./data/census.csv. The model is stored in ./model/model.sav.

More information about the model you find in the model card ./model_card.md.

The project can be found on GitHub [https://github.com/rabanser89/ml_project_census_bureau](https://github.com/rabanser89/ml_project_census_bureau).

## Requirements

Requirements can be found in requirements.txt and can be installed via
```
pip install -r requirements.txt
```

## API deployment

An API has implementes GET and POST on the root domain. The GET gives a greeting and the POST does model inference. You can use 
```
uvicorn main: app --reload
```
to inspect API documentation. A screenshot for that you can be found in ./screenshot/example.png

## Test

In this project test for the ML model and the api are implemented. To run the tests use
```
pytest
```

## Continuous Integration
For this project GitHub action are used. GitHub action run pytest and flake8 on push to main/master.

## API Deployment

The app is deployed to the cloud application platform [www.render.com](https://www.render.com), where continuous delivery is enabled. See the screenshot continuous_deployment.png and continuous_deployment2.png

The live api in action can be found in the screenshot live_get.png and post_get.png
