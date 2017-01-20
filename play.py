import tensorflow as tf
import numpy as np
import game.wrapped_flappy_bird as game
from model import QFuncModel
from config import *
from utils import *

model = QFuncModel(args)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('save')
with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    game_state = game.GameState()
    do_nothing = np.zeros(args.actions)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = rgb2gray(resize(x_t))
    s_t = np.stack([x_t for i in range(args.frames)], axis=2)
    while True:
        a_t = np.zeros([args.actions])
        readout_t = sess.run(model.readout, feed_dict={model.s: [s_t]})
        a_t[np.argmax(readout_t)] = 1

        x_t_next, r_t, terminal = game_state.frame_step(a_t)
        x_t_next = rgb2gray(resize(x_t_next))
        s_t = np.append(x_t_next[:, :, np.newaxis], s_t[:, :, 0:3], axis=2)
        print '%d, %f, %d' % (a_t[1], r_t, terminal)