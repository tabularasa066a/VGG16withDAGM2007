## https://qiita.com/T_Tao/items/0e869e440067518b6b58

from __future__ import print_function
import keras
from keras.applications import VGG16
from keras.models import Sequential, load_model, model_from_json
from keras import models, optimizers, layers
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.preprocessing import image as images
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras import backend as K
import os
import numpy as np
import glob
import pandas as pd
import cv2

num_classes = 2
folder = ["Norml/Class1","withDefects/Class1_def"]
image_size = 224
x = []
y = []

for index, name in enumerate(folder):
    dir = "../DAGM2007/" + name
    files = glob.glob(dir + "/*.png")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        x.append(data)
        y.append(index)

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=111)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# ｙ　ラベルをワンホット表現に
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


## modelの読み込み(存在する場合のみ)
if os.path.exists('./grad1_vgg16_weight_DAGM_C1.h5'):
  model.load_weights('./grad1_vgg16_weight_DAGM_C1.h5')


vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
last = vgg_conv.output

mod = Flatten()(last)
mod = Dense(1024, activation='relu')(mod)
mod = Dropout(0.5)(mod)
preds = Dense(2, activation='sigmoid')(mod)

model = models.Model(vgg_conv.input, preds)
model.summary()

epochs = 100
batch_size = 48

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True)


scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

### Plot accuracy & loss
import matplotlib.pyplot as plt

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)

model.save_weights('grad_vgg16_weight_DAGM_C1.h5')


#plot accuracy
plt.plot(epochs, acc, label = "Training acc" )
plt.plot(epochs, val_acc, label = "Validation acc")
plt.title("Training and Validation accuracy")
plt.legend()
plt.show()
plt.close()

#plot loss
plt.plot(epochs, loss,  label = "Training loss" )
plt.plot(epochs, val_loss, label = "Validation loss")
plt.title("Training and Validation loss")
plt.legend()
plt.show()


K.set_learning_phase(1) #set learning phase

def Grad_Cam(input_model, pic_array, layer_name):

    # 前処理
    pic = np.expand_dims(pic_array, axis=0)
    pic = pic.astype('float32')
    preprocessed_input = pic / 255.0

    # 予測クラスの算出
    predictions = input_model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = input_model.output[:, class_idx]

    #  勾配を取得
    conv_output = input_model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
    gradient_function = K.function([input_model.input], [conv_output, grads])  # input_model.inputを入力すると、conv_outputとgradsを出力する関数

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # 重みを平均化して、レイヤーのアウトプットに乗じる
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像化してヒートマップにして合成
    cam = cv2.resize(cam, (224, 224), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    jetcam = (np.float32(jetcam) + pic / 2)   # もとの画像に合成
    return jetcam


pic_array = img_to_array(load_img('../DAGM2007/withDefects/Class1_def/12.png', target_size=(224, 224)))
pic = pic_array.reshape((1,) + pic_array.shape)
array_to_img(pic_array)


picture = Grad_Cam(model, pic_array, 'block5_conv3')
picture = picture[0,:,:,]
array_to_img(picture)
