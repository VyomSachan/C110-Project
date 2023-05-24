import cv2
import numpy
import tensorflow

model = tensorflow.keras.models.load_model("keras_RPS_model.h5")
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    image = cv2.resize(frame, (224, 224))
    test_image = numpy.array(image, dtype=numpy.float32)
    test_image = numpy.expand_dims(test_image, axis=0)
    normalised_image = test_image/255.0
    prediction = model.predict(normalised_image)
    prediction = prediction[0]
    print("Prediction : ", prediction)

    RPS_index = numpy.argmax(prediction)
    if RPS_index == 0:
        print("It is a Stone!")
    elif RPS_index == 1:
        print("It is a Paper!")
    elif RPS_index == 2:
        print("It is a Scissor!")

    frame = cv2.flip(frame ,1)
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == 32:
        print("Closing...")
        break

video.release()
cv2.destroyAllWindows()