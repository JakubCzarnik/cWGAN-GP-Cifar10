import pickle
import numpy as np
import tensorflow as tf

def get_data(size:int) -> tuple[np.ndarray, np.ndarray]:
   def load_cifar_batch(file:str):
      with open(file, 'rb') as fo:
         dict = pickle.load(fo, encoding='bytes')
      return dict

   x_train, labels = [], []
   for i in range(1, size+1):
      file_path = f"cifar-10/data_batch_{i}"
      label = load_cifar_batch(file_path)[b'labels']
      imgs = load_cifar_batch(file_path)[b'data']
      x_train.append(imgs)
      labels.append(label)
   
   x_train = np.concatenate(x_train, axis=0)
   x_train = np.reshape(x_train, (len(x_train), 3, 32, 32)).astype('float32')
   x_train = (x_train / 255 - 0.5) * 2
   x_train = np.clip(x_train, -1, 1)
   x_train = np.transpose(x_train, (0, 2, 3, 1))
   labels = np.concatenate(labels, axis=0)
   return x_train, labels


def augment_image(image, label):
   image = tf.image.random_flip_left_right(image)
   image = tf.image.random_hue(image, 0.2)
   return image, label


def apply_augmentation(image, label):
    augmented_image, augmented_label = tf.cond(
        tf.random.uniform(()) < 0.5,
        lambda: augment_image(image, label),
        lambda: (image, label)
    )
    return augmented_image, augmented_label