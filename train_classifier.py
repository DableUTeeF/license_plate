from model import model, preprocess_input, smodel
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from generator import Generator
import json

if __name__ == '__main__':
    model = smodel()
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    print(model.summary())
    model.load_weights('weights/classifier.h5')

    listsss = json.load(open('list_withbndbx.json', 'r'))
    train_gen = Generator(listsss[:7211], '/home/palm/PycharmProjects/DATA/Tanisorn/imgCarResize/',
                          preprocess_function=preprocess_input)
    test_gen = Generator(listsss[7211:], '/home/palm/PycharmProjects/DATA/Tanisorn/imgCarResize/',
                         preprocess_function=preprocess_input)
    reduce_lr_01 = ReduceLROnPlateau(monitor='val_1st_acc', factor=0.2,
                                     patience=5, min_lr=0, mode='max')
    reduce_lr_02 = ReduceLROnPlateau(monitor='val_2nd_acc', factor=0.2,
                                     patience=5, min_lr=0., mode='max')
    reduce_lr_03 = ReduceLROnPlateau(monitor='val_3rd_acc', factor=0.2,
                                     patience=5, min_lr=0, mode='max')
    reduce_lr_04 = ReduceLROnPlateau(monitor='val_4th_acc', factor=0.2,
                                     patience=5, min_lr=0., mode='max')
    for i in range(30):
        f = model.fit_generator(train_gen, steps_per_epoch=7210 / 32, epochs=1, validation_data=test_gen,
                                validation_steps=1000 / 32,
                                callbacks=[reduce_lr_01, reduce_lr_02, reduce_lr_03, reduce_lr_04])
        model.save_weights('weights/classifier.h5')
