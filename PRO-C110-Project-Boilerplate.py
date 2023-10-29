# To Capture Frame
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')
# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status, frame = camera.read()


	if status:


		frame = cv2.flip(frame , 1)
		
		img = cv2.resize(frame,(224,224))



		test_image = np.array(img, dtype=np.float32)
		test_image = np.expand_dims(test_image, axis=0)

   
		normalised_image = test_image/255.0

   
		prediction = model.predict(normalised_image)

		print("Prediction : ", prediction)
			
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			print("Closing")
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
