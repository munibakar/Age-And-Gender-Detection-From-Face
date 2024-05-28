# Gender and Age Prediction Model


### About the Project
This project contains a Convolutional Neural Network (CNN) model that predicts a person's age and gender using the UTKFace dataset. The model has two separate output layers for predicting age and gender simultaneously.





### Technologies Used
- Python
- Keras
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Seaborn




### Clone this repository:

1. `git clone` https://github.com/munibakar/Age-And-Gender-Detection-From-Face.git

2. `cd Age-And-Gender-Detection-From-Face` 




### Model Training
The CNN model is defined and trained in `model.py`. The model includes four convolutional layers followed by two fully connected layers. It is compiled using `binary_crossentropy` for gender and `mean_squared_error` for age. The model is trained for 125 epochs and saved as `gender_age_prediction_model.h5`.

+ `python model.py`

### Model Testing
In `main.py`, the trained model is loaded to make predictions on a test image. The test image is preprocessed and passed through the model to predict age and gender, which are then displayed.

+ `python main.py`


