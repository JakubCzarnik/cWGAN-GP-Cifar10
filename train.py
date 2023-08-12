import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from model import build_discriminator, build_generator, ModelGAN
from data_tools import get_data, apply_augmentation

for gpu in tf.config.list_physical_devices('GPU'):
   print(gpu) 
   tf.config.experimental.set_memory_growth(gpu, True)

###  settings  ###
data_size = 6 #  <-- times 10k, max 6 - 60k

generator_lr = 0.0003
discriminator_lr = 0.0002
beta_1 = 0.5

batch_size = 100
lat_dim = 100
epochs = 150

image_on_epoch = 1
checkpoint_on_epoch = 10
restore_last_checkpoint = False
### end-settings  ###

generator = build_generator(lat_dim)
discriminator = build_discriminator()
discriminator.summary()
generator.summary()

b_loss = BinaryCrossentropy()
g_opt = Adam(generator_lr, beta_1=beta_1)
d_opt = Adam(discriminator_lr, beta_1=beta_1)

gan = ModelGAN(generator, discriminator)
gan.compile(g_opt, d_opt, b_loss, lat_dim, image_on_epoch, checkpoint_on_epoch)

if restore_last_checkpoint:
   gan.restore_checkpoint()


x_train, y_train = get_data(data_size)
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(data_size*10000)
dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch_size, drop_remainder=True)


gan.train(dataset, data_size=data_size*10000, epochs=epochs, batch_size=batch_size)

