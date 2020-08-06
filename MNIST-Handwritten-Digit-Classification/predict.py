# make a prediction for a new image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from matplotlib import pyplot

# load and prepare image
def load_image(filename):
    # load
    img = load_img(filename, grayscale=True, target_size=(28,28))
    # convert to array
    img = img_to_array(img)
    # reshape into single sample with 1 channel
    img = img.reshape(1,28,28,1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

# make predictions with loaded image
def predict():
    # load
    img = load_image('sample.png')
    pyplot.imshow(img.squeeze())
    pyplot.show()
    model = load_model('final_model.h5')
    # predict
    digit = model.predict_classes(img)
    print('The digit is: ', digit[0])

# run prediction
predict()