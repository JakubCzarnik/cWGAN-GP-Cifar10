import os, time, cv2, datetime
import tensorflow as tf
import numpy as np
from tensorflow import summary
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Reshape, BatchNormalization, LeakyReLU, Flatten, Concatenate
from tensorflow.keras.models import Model


class WGAN_GP(Model):
   def __init__(self, generator, critic, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.generator = generator
      self.critic = critic


   def compile(self, 
               g_opt,
               c_opt,
               lat_dim:int,
               lambda_gp:int,
               critic_steps:int,
               epoch_img:int,
               epoch_checkpoint:int,
               *args, **kwargs):
      super().compile(*args, **kwargs)
      self.g_opt = g_opt
      self.c_opt = c_opt
      self.lat_dim = lat_dim
      self.lambda_gp = lambda_gp
      self.critic_steps = critic_steps
      self.epoch_img = epoch_img
      self.epoch_checkpoint = epoch_checkpoint
      # checkpoint settings
      self.checkpoint_dir = './checkpoints'
      self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
      self.checkpoint = tf.train.Checkpoint(g_opt=self.g_opt,
                                            c_opt=self.c_opt,
                                            generator=self.generator,
                                            critic=self.critic)
   

   def critic_loss(self, real_images, fake_images, fake_output, real_output, labels):
      alpha = tf.random.uniform([real_images.shape[0], 1, 1, 1], minval=0.0, maxval=1.0)
      interpolated_images = alpha * real_images + (1 - alpha) * fake_images

      with tf.GradientTape() as gp_tape:
         gp_tape.watch(interpolated_images)
         interpolated_output = self.critic([interpolated_images, labels], training=True)

      gradients = gp_tape.gradient(interpolated_output, interpolated_images)
      gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
      gradient_penalty = tf.reduce_mean(tf.square(gradients_norm - 1))
      
      gp = self.lambda_gp * gradient_penalty
      loss = -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output) + gp
      return loss, gp
   

   def generator_loss(self, fake_output):
      return -tf.reduce_mean(fake_output)


   @tf.function
   def train_step(self, real_images:tf.Tensor, labels:tf.Tensor) -> dict:
      noise = tf.random.normal((real_images.shape[0], self.lat_dim))
      with tf.GradientTape() as c_tape, tf.GradientTape() as g_tape:
         fake_images = self.generator([noise, labels], training=True)

         real_output = self.critic([real_images, labels], training=True)
         fake_output = self.critic([fake_images, labels], training=True)

         c_loss, gp = self.critic_loss(real_images, fake_images, fake_output, real_output, labels)
         g_loss = self.generator_loss(fake_output)
        
      # calculate grads
      cgrad = c_tape.gradient(c_loss, self.critic.trainable_variables)
      ggrad = g_tape.gradient(g_loss, self.generator.trainable_variables)
      # update models
      self.c_opt.apply_gradients(zip(cgrad, self.critic.trainable_variables))
      self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

      real_out = tf.reduce_mean(real_output)
      return  {"c_loss": c_loss, "g_loss": g_loss, "real_out": real_out, "GP": gp}


   @tf.function
   def train_step_discriminator(self, real_images:tf.Tensor, labels:tf.Tensor) -> dict:
      noise = tf.random.normal((real_images.shape[0], self.lat_dim))
      with tf.GradientTape() as c_tape:
         fake_images = self.generator([noise, labels], training=True)

         real_output = self.critic([real_images, labels], training=True)
         fake_output = self.critic([fake_images, labels], training=True)

         c_loss, gp = self.critic_loss(real_images, fake_images, fake_output, real_output, labels)
         g_loss = self.generator_loss(fake_output)

      # update critic
      cgrad = c_tape.gradient(c_loss, self.critic.trainable_variables)
      self.c_opt.apply_gradients(zip(cgrad, self.critic.trainable_variables))

      real_out = tf.reduce_mean(real_output)
      return  {"c_loss": c_loss, "g_loss": g_loss, "real_out": real_out, "GP": gp}
   

   def train(self, dataset, data_size:int, batch_size=32, epochs=10) -> dict:
      logdir = f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}/"
      summary_writer = summary.create_file_writer(logdir)
      n_batches = data_size//batch_size-1
      k = 33 # lenght of loading bar
      for epoch in range(1, epochs+1):
         print(f"Epoch: {epoch}/{epochs}")
         mean_batch_time = 0
         ds = iter(dataset)

         for step in range(n_batches):
            time_step = time.time()
            batch_images, batch_labels = next(ds)

            # Train Step
            if step%self.critic_steps==0: 
               params_dict = self.train_step(batch_images, batch_labels)
            else:
               params_dict = self.train_step_discriminator(batch_images, batch_labels)

            # Create logs dict for batch and epoch
            if step == 0:
               batch_logs = {key:0 for key in params_dict.keys()}
            if epoch == 1:
               hist = {key:[] for key in params_dict.keys()}

            # Dynamic print
            alpha = 1/(step+1) if step>0 else 0
            info = ""
            for key, val in params_dict.items():
               batch_logs[key] = (1-alpha)*batch_logs[key] + alpha*val
               info += f" - {key}: {batch_logs[key]:.3f}"

            # Calculate time, loading bar lenght and display info
            mean_batch_time = (1-alpha)*mean_batch_time + alpha*(time.time() - time_step)
            time_to_end = mean_batch_time * (n_batches - step)
            m = int((step+1)/(n_batches/k))+1
            bar = f"[{'='*m}{'-'*(33-m)}]"
            print(f"Batch: {step+1}/{n_batches}  {bar}  {time_to_end:.1f}s{info}{' '*6}", end='\r')
         
         epoch_time = mean_batch_time * (step+1)
         print(f"Batch: {n_batches}/{n_batches}  {bar}  {epoch_time:.1f}s{info}{' '*6}")
         
         # Learning rate decrease
         if epoch%2==0:
            # generator
            lr = self.g_opt.learning_rate.numpy() * 0.995
            self.g_opt.learning_rate.assign(lr)
            # discriminator
            lr = self.c_opt.learning_rate.numpy() * 0.995
            self.c_opt.learning_rate.assign(lr)

         # Save Checkpoint
         if (epoch) % self.epoch_checkpoint == 0 and epoch > 1:
               self.checkpoint.save(file_prefix=self.checkpoint_prefix)
         # Save Images
         if (epoch) % self.epoch_img == 0 or epoch == 1:
            images_grid = self.sample_img(epoch, save=True)
         else:
            images_grid = self.sample_img(epoch)
         # Save logs
         for key, val in batch_logs.items():
            hist[key].append(val)

         with summary_writer.as_default():
            summary.image(f"generator_output", images_grid, step=epoch)
            summary.scalar(f"learning_rate/generator", self.g_opt.learning_rate.numpy(), step=epoch)
            summary.scalar(f"learning_rate/critic", self.c_opt.learning_rate.numpy(), step=epoch)
            for key in hist:
               summary.scalar(f"metrics/{key}", hist[key][-1], step=epoch)

      return hist
   

   def sample_img(self, epoch:int, save=False) -> None:
      rows, cols = 10, 10
      if not hasattr(self, 'sample_noise'):
         self.sample_noise = tf.random.normal((rows*cols, self.lat_dim))
         labels = np.repeat(np.arange(10), rows)
         self.labels_onehot = tf.one_hot(labels, 10)
      images = self.generator([self.sample_noise, self.labels_onehot], training=False)
      images = (0.5 * images + 0.5) * 255
      image_size = (32, 32)
      grid_image = np.zeros((cols*image_size[0], rows*image_size[1], 3), dtype=np.uint8)

      idx = 0
      for i in range(cols):
         for j in range(rows):
            if idx < len(images):
                  image = images[idx]
                  grid_image[i*image_size[0]:(i+1)*image_size[0], j*image_size[1]:(j+1)*image_size[1], :] = image
                  idx += 1
      if save:
         grid_bgr = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
         cv2.imwrite(f"images/epo_{epoch}.png", grid_bgr)
      return tf.expand_dims(grid_image, axis=0)
      

   def restore_checkpoint(self):
      self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))


### Critic ###
def conv2D_block(x, filters, kernel_size=5, strides=2, padding='same'):
   x = Conv2D(filters=filters, 
              kernel_size=kernel_size, 
              strides=strides, 
              padding=padding,
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
   x = LeakyReLU(alpha=0.2)(x)
   return x


def build_critic(image_size=(32,32,3), num_classes=10):
   images = Input(shape=image_size)
   labels = Input(shape=(num_classes,))

   x = conv2D_block(images, 64)
   x = conv2D_block(x, 128)
   x = conv2D_block(x, 196)
   x = conv2D_block(x, 324)

   x = Flatten()(x)
   x = Concatenate()([x, labels])
   x = Dense(256)(x)
   x = LeakyReLU(alpha=0.2)(x)
   
   validity = Dense(1)(x)

   model = Model(inputs=[images, labels], outputs=validity, name='critic')
   return model


### Generator ###
def conv2DT_block(x, filters, kernel_size=5, strides=2, padding='same'):
   x = Conv2DTranspose(filters, 
                       kernel_size=kernel_size, 
                       strides=strides, 
                       padding=padding,
                       use_bias=False,
                       kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
   x = BatchNormalization()(x)
   x = LeakyReLU(alpha=0.2)(x)
   return x


def build_generator(latent_dim, num_classes=10):
   z = Input(shape=(latent_dim,))
   labels = Input(shape=(num_classes,))

   x = Concatenate()([z, labels])

   x = Dense(2*2*512, use_bias=False)(x)
   x = BatchNormalization()(x)
   x = LeakyReLU(alpha=0.2)(x)

   x = Reshape((2, 2, 512))(x)
   x = conv2DT_block(x, 256)
   x = conv2DT_block(x, 128)
   x = conv2DT_block(x, 64)

   image = Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh')(x)

   model = Model(inputs=[z, labels], outputs=image, name='generator')
   return model


