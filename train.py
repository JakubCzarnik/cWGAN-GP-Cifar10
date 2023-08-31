import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import build_critic, build_generator, WGAN_GP
from data_tools import load_dataset

for gpu in tf.config.list_physical_devices('GPU'):
   print(gpu) 
   tf.config.experimental.set_memory_growth(gpu, True)

###  settings  ###
data_size = 6 # times 10k, max 6 - 60k

generator_lr = 5e-4
critic_lr = 2e-4
beta_1 = 0.5
beta_2 = 0.999

batch_size = 128
lat_dim = 128
lambda_gp = 10 # gradient penalty coefficient
disc_steps = 5 # critic train steps every 1 generator train step

epochs = 1000

image_on_epoch = 1  # Save sample images every {epoch_img} epochs
checkpoint_on_epoch = 25 # Save checkpoint every {epoch_checkpoint} epochs
restore_last_checkpoint = False
### end-settings  ###


# Create cWGAN_GP
generator = build_generator(lat_dim)
critic = build_critic()
generator.summary()
critic.summary()


g_opt = Adam(generator_lr, beta_1=beta_1, beta_2=beta_2)
c_opt = Adam(critic_lr, beta_1=beta_1, beta_2=beta_2)

gan = WGAN_GP(generator, critic)
gan.compile(g_opt=g_opt, 
            c_opt=c_opt,  
            lat_dim=lat_dim,
            lambda_gp=lambda_gp,
            critic_steps=disc_steps,
            epoch_img=image_on_epoch,
            epoch_checkpoint=checkpoint_on_epoch)

if restore_last_checkpoint:
   gan.restore_checkpoint()

# load dataset
dataset = load_dataset(data_size, batch_size)

# start training
gan.train(dataset, data_size=data_size*10000, epochs=epochs, batch_size=batch_size)

