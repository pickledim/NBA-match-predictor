# NBA-match-predictor

This package is used to scrap the statistics of the first half of matches, analyze them and use them as training data for 2 AI models(MLP neural networks & xgboost) in order to predict the final result of the match 

![This is an image](https://github.com/pickledim/NBA-match-predictor/blob/main/pipeline.jpeg)

Steps

1. Run Get_Training_Data.ipynb 
2. Run Get_Validation_Data.ipynb  
2. Run Keras.ipynb or ml_models.py to create the models
3. Run Game_Analysis_Season_2022_23.ipynb 
