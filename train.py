import tensorflow as tf
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from collections import deque
import game.wrapped_flappy_bird as game
from model import QFuncModel
from config import *

def rgb2gray(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    return res

def resize(image):
    return cv2.resize(image, (args.resize_width, args.resize_height))

model = QFuncModel(args)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    game_state = game.GameState()
    do_nothing = np.zeros(args.actions)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = rgb2gray(resize(x_t))
    s_t = np.stack([x_t for i in range(args.frames)], axis=2)
    epsilon = args.initial_epsilon
    D = deque(maxlen=args.replay_memory)
    t = 0
    while True:
        a_t = np.zeros([args.actions])
        if random.random() < epsilon:
            a_t[random.randrange(args.actions)] = 1
        else:
            readout_t = sess.run(model.readout, feed_dict={model.s: [s_t]})
            a_t[np.argmax(readout_t)] = 1

        x_t_next, r_t, terminal = game_state.frame_step(a_t)
        x_t_next = rgb2gray(resize(x_t_next))
        s_t_next = np.append(x_t_next[:, :, np.newaxis], s_t[:, :, 0:3], axis=2)
        D.append((s_t, a_t, r_t, s_t_next, terminal))

        if epsilon > args.final_epsilon and t > args.observe:
            epsilon -= (args.initial_epsilon - args.final_epsilon) / args.explore

        if t > args.observe:
            minibatch = random.sample(D, args.batch_size)
            s_batch, a_batch, r_batch, s_next_batch, _ = zip(*minibatch)
            readout_batch = sess.run(model.readout, feed_dict={model.s: s_next_batch})
            y_batch = []
            for i, (s, a, r, s_next, terminal) in enumerate(minibatch):
                if terminal:
                    y_batch.append(r)
                else:
                    y_batch.append(r + args.gamma * np.max(readout_batch[i]))
            sess.run(model.train_op, feed_dict={
                model.s: s_batch,
                model.y: y_batch,
                model.a: a_batch
            })

        print '%d, %d, %f, %d' % (t, a_t[1], r_t, terminal)
        t += 1
        s_t = s_t_next
        if t % 10000 == 0:
            saver.save(sess, 'save/model.tfmodel', global_step=t)



