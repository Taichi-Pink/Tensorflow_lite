import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K
import cv2
import numpy as np
import matplotlib.pyplot as plt
patch_size = 256
scene_no = 50
bs = 1
results_p = './result/'
result_path = './log/'

select_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
vgg19 = VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
vgg19.trainable = False
for l in vgg19.layers:
    l.trainable = False
select = [vgg19.get_layer(name).output for name in select_layers]

model_vgg = Model(inputs=vgg19.input, outputs=select)
model_vgg.trainable = False


def vgg_loss(y_true, y_pred):
    out_pred = model_vgg(y_pred)
    out_true = model_vgg(y_true)
    loss_f = 0
    for f_g, f_l in zip(out_pred, out_true):
        loss_f += K.mean(K.abs(f_g - f_l))
    return loss_f + tf.math.reduce_mean(tf.square(y_true - y_pred))


k = 6
def model_test1(w, h, c):
    input_1 = keras.Input((w, h, c))
    input_2 = keras.Input((w, h, c))
    input = layers.Concatenate(axis=3)([input_1,input_2])
    x1 = layers.SeparableConv2D(64, (k, k), strides=(1, 1), activation='relu', padding="same")(input)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(16, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(32, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(16, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(32, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(16, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(32, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(16, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(32, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(16, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(32, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Concatenate(axis=3)([x1, input])
    x1 = layers.SeparableConv2D(16, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(32, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(16, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(32, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.SeparableConv2D(3, (k, k), strides=(1, 1), activation='relu', padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    output_ = layers.Activation('tanh')(x1)
    model = Model(inputs=[input_1, input_2], outputs=output_)
    op = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=op, loss='mse')
    return model


def train(model):

    print(model.summary())
    model.load_weights('./model1_16.h5')
    #model.save('./model.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_dog_model = converter.convert()
    open("lite_model.tflite", "wb").write(tflite_dog_model)

def norm_0_to_1(img):
    img = np.float32(img)
    img_flat = img.flatten()
    max_value = np.max(img_flat)
    min_value = np.min(img_flat)
    new_img = (img - min_value) * 1 / (max_value - min_value)
    return new_img

def test_tflite():
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="lite_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # load data
    p_over = r'F:\Project\Android Studio\app\src\main\assets\7.jpg'
    over_exp = cv2.imread(p_over)
    over_exp = over_exp[:, :, ::-1]

    p_under = r'F:\Project\Android Studio\app\src\main\assets\1.jpg'
    under_exp = cv2.imread(p_under)
    under_exp = under_exp[:, :, ::-1]

    h = 2736
    w = 1824
    over_exp = cv2.resize(over_exp, (h, w))
    under_exp = cv2.resize(under_exp, (h, w))

    over_exp = np.expand_dims(over_exp, axis=0)
    over_exp = np.array(over_exp, dtype=np.float32)
    under_exp = np.expand_dims(under_exp, axis=0)
    under_exp = np.array(under_exp, dtype=np.float32)
    over_exp = norm_0_to_1(over_exp)
    under_exp = norm_0_to_1(under_exp)

    # Test model on random input data.
    # input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], over_exp)
    interpreter.set_tensor(input_details[1]['index'], under_exp)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)

    # output_array = output_data.numpy()
    plt.imshow(np.squeeze(output_data))
    plt.show()

if __name__ == "__main__":
    model = model_test1(1824,2736,3)
    train(model)

