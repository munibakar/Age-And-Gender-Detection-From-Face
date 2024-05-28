from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np
import matplotlib.pyplot as plt

#The model we trained is loaded
loaded_model = load_model('gender_age_prediction_model.h5', compile=True)
loaded_model.compile(optimizer='adam', loss=['binary_crossentropy', 'mse'], metrics=['accuracy', 'mse'])

#The necessary operations are performed to prepare the test image for the model
test_image_path = "./test_image/ronaldo.jpg"

test_img = load_img(test_image_path, color_mode='rgb', target_size=(128, 128))
test_img_array = np.array(test_img)
test_img_array = test_img_array / 255.0  

input_test_image = test_img_array.reshape(1, 128, 128, 3)

pred_gender, pred_age = loaded_model.predict(input_test_image)

#Visualization processes are performed.
plt.imshow(test_img)
plt.axis('off')

gender_dict = {0: 'Male', 1: 'Female'}
predicted_gender = gender_dict[round(pred_gender[0][0])]  
predicted_age = round(pred_age[0][0])
plt.title(f"Predicted Gender: {predicted_gender}, Predicted Age: {predicted_age}")
plt.show()



