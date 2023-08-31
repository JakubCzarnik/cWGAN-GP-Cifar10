import pickle
import numpy as np
import tensorflow as tf

def load_data(size:int):
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
   # images
   x_train = np.concatenate(x_train, axis=0)
   x_train = np.reshape(x_train, (len(x_train), 3, 32, 32)).astype('float32')
   x_train = np.transpose(x_train, (0, 2, 3, 1))
   # labels
   labels = np.concatenate(labels, axis=0)
   labels = tf.one_hot(labels, 10)
   return x_train, labels


def augment_image(image, label):
   image = tf.image.random_flip_left_right(image)
   image = tf.image.random_brightness(image, 0.1)
   image = tf.image.random_contrast(image, 0.9, 1.1)
   image = tf.image.random_hue(image, 0.1)
   return image, label


def apply_augmentation(image, label):
   augmented_image, augmented_label = tf.cond(
      tf.random.uniform(()) < 0.6,
      lambda: augment_image(image, label),
      lambda: (image, label)
   )
   augmented_image = (augmented_image / 255 - 0.5) * 2
   augmented_image = tf.clip_by_value(augmented_image, -1, 1)
   return augmented_image, augmented_label


def load_dataset(data_size, batch_size, augmentations=True):
   x_train, y_train = load_data(data_size)
   dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(data_size*10000)
   if augmentations:
      dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)  
   dataset = dataset.batch(batch_size, drop_remainder=True)
   return dataset
