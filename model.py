import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from keras.regularizers import l2
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split

BASE_DIR = 'UTKFace/'

#The paths and tags of the image files are taken and converted into DataFrame. Age distribution and number of genders are shown in graphs.
image_paths = []
age_labels = []
gender_labels = []

for filename in tqdm(os.listdir(BASE_DIR)):
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)


df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels


gender_dict = {0:'Male', 1:'Female'}

sns.histplot(df['age'], kde=True)  


sns.countplot(df['gender'])


plt.figure(figsize=(20, 20))
files = df.iloc[0:25]

for index, file, age, gender in files.itertuples():
    plt.subplot(5, 5, index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title(f"Age: {age} Gender: {gender_dict[gender]}")
    plt.axis('off')

#Images are resized to 128x128 and pixel values are normalized to a range of 0-1. Gender and age labels are converted to numpy arrays.
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='rgb')
        img = img.resize((128, 128))
        img = np.array(img)
        features.append(img)
        
    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 3)
    return features

X = extract_features(df['image'])
X.shape

X = X/255.0

y_gender = np.array(df['gender'])
y_age = np.array(df['age'])

input_shape = (128, 128, 3)
inputs = Input((input_shape))

#In this section, the Convolutional Neural Network (CNN) model is defined. The model has four convolution and pooling layers, followed by two fully connected layers.
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
conv_1 = BatchNormalization()(conv_1)
maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxp_1)
conv_2 = BatchNormalization()(conv_2)
maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(maxp_2)
conv_3 = BatchNormalization()(conv_3)
maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(maxp_3)
conv_4 = BatchNormalization()(conv_4)
maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

flatten = Flatten()(maxp_4)

dense_1 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(flatten)
dense_2 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(flatten)

dropout_1 = Dropout(0.4)(dense_1)
dropout_2 = Dropout(0.4)(dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)
output_2 = Dense(1, activation='relu', name='age_out')(dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])


#The data is divided into a 20% test set, the model is compiled using man optimization with binary_crossentropy and mean_squared_error. 
#Training takes 125 epochs, recordings are made with a validation set
model.compile(loss=['binary_crossentropy', 'mean_squared_error'], optimizer='adam', metrics=['accuracy', 'mse'])

X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(X, y_gender, y_age, test_size=0.2, stratify=y_gender)

history = model.fit(x=X_train, 
                    y=[y_gender_train, y_age_train], 
                    batch_size=32, 
                    epochs=125, 
                    validation_split=0.2)


#Visualization processes are performed.
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['gender_out_accuracy'], label='Gender Training Accuracy')
plt.plot(history.history['val_gender_out_accuracy'], label='Gender Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

plt.figure(figsize=(12, 6))

plt.plot(history.history['age_out_mse'], label='Training MSE')  
plt.plot(history.history['val_age_out_mse'], label='Validation MSE')  
plt.title('Age MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

#model saved...
model.save("gender_age_prediction_model.h5")
