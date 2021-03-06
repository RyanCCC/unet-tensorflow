import os
from random import shuffle

import cv2
from tensorflow import keras
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from PIL import Image


def dice_loss_with_CE(beta=1, smooth = 1e-5):
    def _dice_loss_with_CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))

        tp = K.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true[...,:-1], axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, CE_loss])
        return CE_loss + dice_loss
    return _dice_loss_with_CE

def CE():
    def _CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))
        # dice_loss = tf.Print(CE_loss, [CE_loss])
        return CE_loss
    return _CE

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def Focal_Loss(alpha=0.5, gamma=2):
    '''
    y_pred：用one-hot的方式表示，如两类别，则输出的维度为[batch_size, width, height, class_num]
    '''
    def Focal_Loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
        logpt = - y_true[...,:-1] * K.log(y_pred)
        logpt = - K.sum(logpt, axis = -1)
        pt = tf.exp(logpt)
        if alpha is not None:
            logpt*=alpha
        CE_loss = -((1 - pt) ** gamma) * logpt
        CE_loss = K.mean(CE_loss)
        return CE_loss
    return Focal_Loss_fn

def dice_loss_with_Focal_Loss(beta=1, smooth = 1e-5, alpha=0.5, gamma=2):
    def _dice_loss_with_Focal_Loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        logpt = - y_true[...,:-1] * K.log(y_pred) 
        logpt = - K.sum(logpt, axis = -1)
        pt = tf.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        CE_loss = -((1 - pt) ** gamma) * logpt
        CE_loss = K.mean(CE_loss)
        tp = K.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true[...,:-1], axis=[0,1,2]) - tp
        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, CE_loss])
        return CE_loss + dice_loss
    return _dice_loss_with_Focal_Loss




