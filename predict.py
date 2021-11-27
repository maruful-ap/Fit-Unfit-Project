from tensorflow.keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2

# dimensions of our images
img_width, img_height = 400, 400

ima_path = 'pic/input/188.jpg'
image = cv2.imread(ima_path)
# load the model we saved
model = load_model('../checkpoint/model-vgg16/model-003-0.760580-0.861702.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# predicting images
img = load_img(ima_path, target_size=(img_width, img_height))
x = img_to_array(img)
x = preprocess_input(x)
x = np.expand_dims(x, axis=0)

#images = np.vstack([x])
classes = model.predict(x)
print(classes)

(Fitness_vehicle, Fitnessless_vehicle) = model.predict(x)[0]

if (Fitness_vehicle > Fitnessless_vehicle):
    label = "Fitness_vehicle"
    color = (255,255,0)
else:
    label = "Fitnessless_vehicle"
    color = (0, 0, 255)


# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (10, 30) 
  
# fontScale 
fontScale = .70
   
  
# Line thickness of 2 px 
thickness = 2

# Using cv2.putText() method 
image = cv2.putText(image, label, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
   
# Displaying the image 
#cv2.imshow(window_name, image)
# show the output image
print("[info] saving image")
cv2.imwrite("pic/output/7.png", image)
cv2.imshow("Output", image)
cv2.waitKey(0)  


"""
cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)

# predicting multiple images at once
img = image.load_img('test2.jpg', target_size=(img_width, img_height))
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)

# pass the list of multiple images np.vstack()
images = np.vstack([x, y])
classes = model.predict_classes(images, batch_size=10)
"""
# print the classes, the images belong to
#print(classes)
#print(classes[0])
