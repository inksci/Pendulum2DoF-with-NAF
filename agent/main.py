from keras import backend
import tensorflow as tf

print( "tf.__version__: ", tf.__version__ )

TCP_PORT = 15001
from ink_v1.env_tcp import env_client
env = env_client( tcp_port = TCP_PORT )






import argparse
import keras
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=102)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--batch_norm', action="store_true", default=False)
parser.add_argument('--no_batch_norm', action="store_false", dest="batch_norm")
parser.add_argument('--min_train', type=int, default=10)
parser.add_argument('--train_repeat', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--episodes', type=int, default=20000)
parser.add_argument('--max_timesteps', type=int, default=200)
parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
parser.add_argument('--optimizer_lr', type=float, default=0.001)
parser.add_argument('--noise_decay', choices=['linear', 'exp', 'fixed'], default='linear')
parser.add_argument('--fixed_noise', type=float, default=0.1)
parser.add_argument('--display', action='store_true', default=False)
parser.add_argument('--no_display', dest='display', action='store_false')

args = parser.parse_args()

state_dim = env.state_dim
action_dim = env.action_dim

def P(x):
    print( "x.shape", x.shape )  
    return tf.matmul(x, tf.transpose(x, (0, 2, 1))) 

def Q(t):
    v, a = t
    print( "a.shape", a.shape )
    print( "v.shape", v.shape )
    print( "(v+a).shape", (v+a).shape )
    return v + a

def L(x):
    l = x

    pivot = 0
    rows = []
    for idx in range(action_dim):
        count = action_dim - idx

        diag_elem = tf.exp(tf.slice(l, (0, pivot), (-1, 1)))
        non_diag_elems = tf.slice(l, (0, pivot+1), (-1, count-1))
        # row = tf.pad(tf.concat(1, (diag_elem, non_diag_elems)), ((0, 0), (idx, 0)))
        row = tf.pad(tf.concat(values = (diag_elem, non_diag_elems), axis = 1), ((0, 0), (idx, 0)))
        rows.append(row)

        pivot += count

    return tf.transpose(tf.stack(rows, axis=1), (0, 2, 1))

def A(t):
    m, p, u = t
    u_mu = tf.expand_dims(u - m, -1)
    A0 = -tf.matmul(tf.transpose(u_mu, [0, 2, 1]), tf.matmul(p, u_mu))*0.5
    A0 = tf.reshape(A0, [-1, 1])

    return A0


def createLayers():
    x = Input(shape=[state_dim], name='x')
    u = Input(shape=[action_dim], name='u')
    if args.batch_norm:
        h = BatchNormalization()(x)
    else:
        h = x
    for i in range(args.layers):
        h = Dense(args.hidden_size, activation=args.activation, name='h'+str(i+1))(h)
        if args.batch_norm and i != args.layers - 1:
            h = BatchNormalization()(h)
    v = Dense(1, name='v')(h)
    m = Dense(action_dim, activation="tanh", name="m", kernel_initializer="uniform")(h)
    l = Dense( int( (action_dim * (action_dim + 1))/2 ) , name='l0')(h)
    # l = Reshape((action_dim, action_dim))(l)
    l = Lambda(L, output_shape=(action_dim, action_dim), name='l')(l)
    p = Lambda(P, output_shape=(action_dim, action_dim), name='p')(l)
    a = merge([m, p, u], mode=A, output_shape=(1,), name="a")
    q = merge([v, a], mode=Q, output_shape=(1,), name="q")
    return x, u, m, v, q

x, u, m, v, q = createLayers()

model = Model(input=[x,u], output=q)
model.summary()

if args.optimizer == 'adam':
    optimizer = Adam(args.optimizer_lr)
elif args.optimizer == 'rmsprop':
    optimizer = RMSprop(args.optimizer_lr)
else:
    assert False
model.compile(optimizer=optimizer, loss='mse')

_mu = K.function([K.learning_phase(), x], [m])
mu = lambda x: _mu([0] + [x])
# The learning phase flag is a bool tensor (0 = test, 1 = train) 
# to be passed as input to any Keras function 
# that uses a different behavior at train time and test time.

x, u, m, v, q = createLayers()

target_model = Model(input=[x,u], output=q)
target_model.set_weights(model.get_weights())

_V = K.function([K.learning_phase(), x], [v])
V = lambda x: _V([0] + [x])


is_render = 0

if args.display:
    is_render = 1


pointer = 0
MEMORY_CAPACITY = 10000

prestates = np.zeros((MEMORY_CAPACITY, state_dim), dtype=np.float32)
actions = np.zeros((MEMORY_CAPACITY, action_dim), dtype=np.float32)
poststates = np.zeros((MEMORY_CAPACITY, state_dim), dtype=np.float32)
rewards = np.zeros((MEMORY_CAPACITY, 1), dtype=np.float32)

i_episode = 0
r_save = []

MAX_EPISODES = 100

noise = 0.9

for _ in range(MAX_EPISODES):
    i_episode += 1
    observation = env.reset()
    dis_save = []
    episode_reward = 0
    
    if pointer > MEMORY_CAPACITY:
        noise *= 0.95
    
    
    for t in range( env.max_steps ):

        index = pointer % MEMORY_CAPACITY
    
        x = np.array([observation])
        u = mu(x)
        
        if pointer > MEMORY_CAPACITY:
            action = u[0][0] + np.random.randn(action_dim) * max( noise, 0.05 )
            
        else:
            action = np.random.randn(action_dim)* 1.0

        prestates[index, :] = observation
        actions[index, :] = action
        # print( "prestate:", observation)

        observation, reward, done, info = env.step(action)
        episode_reward += reward
        # print( "poststate:", observation)

        rewards[index, :] = reward
        poststates[index, :] = observation         
        
        pointer += 1
        
        
        dis_save.append( info )

        if pointer > MEMORY_CAPACITY:
            for k in range(args.train_repeat):
                assert( len(prestates) > args.batch_size )
                indexes = np.random.choice( MEMORY_CAPACITY, size=args.batch_size)

                v = V(np.array(poststates)[indexes])
                y = np.squeeze( np.array(rewards)[indexes] ) + args.gamma * np.squeeze(v)
                model.train_on_batch([np.array(prestates)[indexes], np.array(actions)[indexes]], y)

                weights = model.get_weights()
                target_weights = target_model.get_weights()
                for i in range(len(weights)):
                    target_weights[i] = args.tau * weights[i] + (1 - args.tau) * target_weights[i]
                target_model.set_weights(target_weights)

        if done:
            break

    r_save.append(episode_reward)
    print("Episode {} finished after {} timesteps, average reward {}".format(i_episode + 1, t + 1, episode_reward))
    if episode_reward > -0.1:
        is_render = 1



env.close_tcp()
