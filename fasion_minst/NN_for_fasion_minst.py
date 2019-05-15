#!/usr/bin/env python
# coding: utf-8

# In[90]:


# いろいろインポート
# kerasはTFのラッパー関数
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(198)
tf.set_random_seed(198)


# In[91]:


# データセットの準備　minstの中のfassion_minstを利用。
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[92]:


# Labelが数字だとわかりにくいので決めておく
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[29]:


# 28*28ピクセルの写真が6万枚ある。
train_images.shape


# In[94]:


# i番目の画像をチェックする。
i = 20
#print(train_images[i])
plt.figure()
plt.imshow(train_images[i], cmap=plt.cm.binary) # 白黒にする
plt.colorbar()
plt.gca().grid(False)


# In[95]:


# データの前処理、ピクセル数を[0,1]にする
train_images = train_images / 255.0

test_images = test_images / 255.0


# In[96]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])


# In[139]:


# 深層モデルの構築
# 入力はピクセル(28*28)を行ベクトル(1*784)にしたもの
# 入力　→ 20 → 10 → ソフトマックス層(出力層)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[140]:


# Early-stopping 
early_stopping = keras.callbacks.EarlyStopping(patience=2, verbose= 1) 


# In[141]:


# 各バッチの損失リスト
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

history = LossHistory()


# In[142]:


# モデルをコンパイルする
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[156]:


hist = model.fit(train_images, train_labels, batch_size=1024, epochs=10,                  validation_data=(test_images, test_labels),                  callbacks=None, verbose=1)


# In[157]:


# lossをプロット
x = np.linspace(0, len(history.losses)+1, len(history.losses))
plt.figure(figsize=(6, 6))
plt.plot(x, history.losses)
plt.show()


# In[145]:


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


# In[146]:


predictions = model.predict(test_images)


# In[147]:


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# In[150]:


# i番目のテストサンプルの評価
i = 16
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


# In[153]:


# X個のテスト画像、予測されたラベル、正解ラベルを表示します。
# 正しい予測は青で、間違った予測は赤で表示しています。
num_rows = 10
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)

