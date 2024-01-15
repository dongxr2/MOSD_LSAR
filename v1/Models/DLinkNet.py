# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 18:13:46 2023

@author: Lenovo
"""

from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Add, Activation
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.models import Model



INPUT_SHAPE = (256, 256, 3)


def residual_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    input_tensor = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    res_tensor = Add()([input_tensor, x])
    res_tensor = Activation('relu')(res_tensor)
    return res_tensor


def dilated_center_block(input_tensor, num_filters):

    dilation_1 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(1, 1), padding='same')(input_tensor)
    dilation_1 = Activation('relu')(dilation_1)

    dilation_2 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(2, 2), padding='same')(dilation_1)
    dilation_2 = Activation('relu')(dilation_2)

    dilation_4 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(4, 4), padding='same')(dilation_2)
    dilation_4 = Activation('relu')(dilation_4)

    dilation_8 = Conv2D(num_filters, kernel_size=(3, 3), dilation_rate=(8, 8), padding='same')(dilation_4)
    dilation_8 = Activation('relu')(dilation_8)

    final_diliation = Add()([input_tensor, dilation_1, dilation_2, dilation_4, dilation_8])

    return final_diliation


def decoder_block(input_tensor, num_filters):
    decoder_tensor = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
    decoder_tensor = BatchNormalization()(decoder_tensor)
    decoder_tensor = Activation('relu')(decoder_tensor)

    decoder_tensor = Conv2DTranspose(num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(decoder_tensor)
    decoder_tensor = BatchNormalization()(decoder_tensor)
    decoder_tensor = Activation('relu')(decoder_tensor)

    decoder_tensor = Conv2D(num_filters, (1, 1), padding='same')(decoder_tensor)
    decoder_tensor = BatchNormalization()(decoder_tensor)
    decoder_tensor = Activation('relu')(decoder_tensor)
    return decoder_tensor


def encoder_block(input_tensor, num_filters, num_res_blocks):
    encoded = residual_block(input_tensor, num_filters)
    while num_res_blocks > 1:
        encoded = residual_block(encoded, num_filters)
        num_res_blocks -= 1
    encoded_pool = MaxPooling2D((2, 2), strides=(2, 2))(encoded)
    return encoded, encoded_pool


def dlinknet():
    inputs = Input(shape=INPUT_SHAPE)
    inputs_ = Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
    inputs_ = BatchNormalization()(inputs_)
    inputs_ = Activation('relu')(inputs_)
    max_pool_inputs = MaxPooling2D((2, 2), strides=(2, 2))(inputs_)

    encoded_1, encoded_pool_1 = encoder_block(max_pool_inputs, num_filters=64, num_res_blocks=3)
    encoded_2, encoded_pool_2 = encoder_block(encoded_pool_1, num_filters=128, num_res_blocks=4)
    encoded_3, encoded_pool_3 = encoder_block(encoded_pool_2, num_filters=256, num_res_blocks=6)
    encoded_4, encoded_pool_4 = encoder_block(encoded_pool_3, num_filters=512, num_res_blocks=3)

    center = dilated_center_block(encoded_4, 512)

    decoded_1 = Add()([decoder_block(center, 256), encoded_3])
    decoded_2 = Add()([decoder_block(decoded_1, 128), encoded_2])
    decoded_3 = Add()([decoder_block(decoded_2, 64), encoded_1])
    decoded_4 = decoder_block(decoded_3, 64)

    final = Conv2DTranspose(32, kernel_size=(3, 3), padding='same')(decoded_4)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(final)
    model_i = Model(inputs=[inputs], outputs=[outputs])
    #model_i.compile(optimizer='adam', loss=combined_loss, metrics=[dice_coeff])
    model_i.summary()
    # model_i.load_weights(save_model_path)
    return model_i