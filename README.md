# Kaggle_NCFM
Using Keras+TensorFlow to solve NCFM-Leadboard Top 5%

Step1. Download dataset from https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data

Step2. Use ```split_train_val.py``` to split the labed data into training and validation. 
Very often, 80% for training and 20% for validation is a good start. 

Step3. Use ```train.py``` to train a Inception_V3 network. The best model and its weights will be saved as "weights.h5".

Step4. Use ```predict.py``` to predict labels for testing images and generating the submission file "submit.csv".
