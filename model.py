import os, time, cv2, datetime
import tensorflow as tf
import numpy as np
from tensorflow import summary
from tensorflow.keras.layers import GaussianNoise, Input, Dense, Conv2D, Conv2DTranspose, Reshape, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.losses import BinaryCrossentropy


class ModelGAN(Model):
   def __init__(self, generator, discriminator, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.generator = generator
      self.discriminator = discriminator


   def compile(self, g_opt, d_opt, b_loss:BinaryCrossentropy, lat_dim:int, epoch_img:int, epoch_checkpoint:int, *args, **kwargs) -> None:
      super().compile(*args, **kwargs)
      self.g_opt = g_opt
      self.d_opt = d_opt
      self.b_loss = b_loss
      self.lat_dim = lat_dim
      self.epoch_img = epoch_img
      self.epoch_checkpoint = epoch_checkpoint

      self.checkpoint_dir = './checkpoints'
      self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
      self.checkpoint = tf.train.Checkpoint(g_opt=self.g_opt,
                                            d_opt=self.d_opt,
                                            generator=self.generator,
                                            discriminator=self.discriminator)


   def generator_loss(self, fake_output:tf.Tensor) -> tf.Tensor:
      return self.b_loss(tf.ones_like(fake_output), fake_output)
      


   def discriminator_loss(self, real_output:tf.Tensor, fake_output:tf.Tensor) -> tf.Tensor:
      fake_loss = self.b_loss(tf.zeros_like(fake_output), fake_output)
      real_loss = self.b_loss(tf.ones_like(real_output), real_output)      
      return real_loss + fake_loss


   @tf.function
   def train_step(self, real_images:tf.Tensor, labels:tf.Tensor) -> dict:
      noise = tf.random.normal((real_images.shape[0], self.lat_dim))
      labels = tf.one_hot(labels, 10)

      with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
         fake_images = self.generator([noise, labels] , training=True)

         real_output = self.discriminator([real_images, labels], training=True)
         fake_output = self.discriminator([fake_images, labels], training=True)

         d_loss = self.discriminator_loss(real_output, fake_output)
         g_loss = self.generator_loss(fake_output)

         real_accuracy = tf.reduce_mean(tf.cast(tf.greater_equal(real_output, 0.5), tf.float32))
         fake_accuracy = tf.reduce_mean(tf.cast(tf.less(fake_output, 0.5), tf.float32))

      dgrad = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
      self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

      ggrad = g_tape.gradient(g_loss, self.generator.trainable_variables)
      self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

      return  {"d_loss": d_loss, "g_loss": g_loss, "r_acc": real_accuracy, "f_acc": fake_accuracy}
   

   def train(self, dataset, data_size:int, batch_size:int=32, epochs:int=1) -> dict:
      logdir = f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}/"
      summary_writer = summary.create_file_writer(logdir)
      n_batches = data_size//batch_size

      for epoch in range(1, epochs+1):
         print(f"Epoch: {epoch}/{epochs}")

         if epoch % 2 == 0 and epoch>1:
            self.g_opt.learning_rate.assign(self.g_opt.learning_rate * 0.98)
            self.d_opt.learning_rate.assign(self.d_opt.learning_rate * 0.98)

         batch_times = []
         for i, (batch_images, batch_labels) in enumerate(dataset):
            t1_iter = time.time()
            params_dict = self.train_step(batch_images, batch_labels)

            # create logs dict for batch and epoch
            if i == 0:
               params_names = [key for key in params_dict.keys()]
               train_logs = [[] for _ in range(len(params_names))]
               

            if epoch == 1:
               hist = {key:[] for key in params_dict.keys()}

            # Create dynamic print
            info = ""
            for idx, val in enumerate(params_dict.values()):

               train_logs[idx].append(val)
               info += f" - {params_names[idx]}: {np.mean(train_logs[idx]):.3f}"

            batch_times.append(time.time() - t1_iter)
            time_to_end = np.mean(batch_times) * (n_batches - (i+1))
            m = int((i+1)/(n_batches/33))+1
            print(f"Batch: {i+1}/{n_batches}  [{'='*m}{'-'*(33-m)}]  {time_to_end:.1f}s{info}   ", end='\r')
         
         epoch_time = tf.reduce_sum(batch_times)
         print(f"Batch: {i+1}/{n_batches}  [{'='*m}{'-'*(33-m)}]  {epoch_time:.1f}s{info}   ", end="\n")


         # Save logs, imgs, models ...
         for idx, param in enumerate(params_names):
            hist[param].append(np.mean(train_logs[idx]))

         if (epoch) % self.epoch_checkpoint == 0 and epoch > 1:
               self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            
         if (epoch) % self.epoch_img == 0 or epoch == 1:
            images_grid = self.sample_img(epoch)
         else:
            images_grid = self.sample_img(epoch, save=False)
         
         with summary_writer.as_default():
            summary.image(f"generator_output", images_grid, step=epoch)
            summary.scalar("metrics/d_loss", hist['d_loss'][-1], step=epoch)
            summary.scalar("metrics/g_loss", hist['g_loss'][-1], step=epoch)
            summary.scalar("metrics/r_acc", hist['r_acc'][-1], step=epoch)
            summary.scalar("metrics/f_acc", hist['f_acc'][-1], step=epoch)

      return hist
   

   def sample_img(self, epoch:int, save:bool=True) -> None:
      rows, cols = 10, 10
      noise = tf.random.normal((rows*cols, self.lat_dim))
      labels = np.repeat(np.arange(10), rows)
      labels_onehot = tf.one_hot(labels, 10)

      images = self.generator([noise, labels_onehot], training = False)
      images = (0.5 * images + 0.5) * 255
      grid_size = (10, 10)
      image_size = (32, 32)
      grid_image = np.zeros((grid_size[0]*image_size[0], grid_size[1]*image_size[1], 3), dtype=np.uint8)

      idx = 0
      for i in range(grid_size[0]):
         for j in range(grid_size[1]):
            if idx < len(images):
                  image = images[idx]
                  grid_image[i*image_size[0]:(i+1)*image_size[0], j*image_size[1]:(j+1)*image_size[1], :] = image
                  idx += 1
      if save:
         grid_bgr = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
         cv2.imwrite(f"images/epo_{epoch}.png", grid_bgr)
      return tf.expand_dims(grid_image, axis=0)
      

   def restore_checkpoint(self) -> None:
      self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))



def conv_block(x, filters, kernel=5, strides=2, padding='same'):
   x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding=padding)(x)
   x = BatchNormalization(momentum=0.9)(x)
   x = LeakyReLU(alpha=0.2)(x)
   return x


def build_discriminator():
   labels = Input(shape=(10,))
   input = Input(shape=(32,32,3))

   x = GaussianNoise(0.2)(input)

   x = conv_block(x, 32)
   x = conv_block(x, 64)
   x = conv_block(x, 128)
   x = conv_block(x, 256)

   x = Flatten()(x)

   x = Concatenate()([x, labels])
   x = Dense(64, activation='relu')(x)
      
   x = Dense(1, activation='sigmoid')(x)

   model = Model(inputs=[input, labels], outputs=x)
   return model


def convT_block(x, filters, kernel_size=5, strides=2, padding='same'):
   x = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
   x = BatchNormalization(momentum=0.9)(x)
   x = LeakyReLU(alpha=0.2)(x)
   return x


def build_generator(latent_dim):
   labels = Input(shape=(10,))
   z = Input(shape=(latent_dim,))
   x = Concatenate()([z, labels])

   x = Dense(2*2*512, activation='relu')(x)
   x = BatchNormalization(momentum=0.9)(x)
   x = LeakyReLU(alpha=0.2)(x)

   x = Reshape((2, 2, 512))(x)

   x = convT_block(x, 256)
   x = convT_block(x, 128)
   x = convT_block(x, 64)

   x = Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh')(x)

   model = Model([z, labels], x)
   return model


