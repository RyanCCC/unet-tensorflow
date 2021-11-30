import os
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from tensorflow.keras.optimizers import Adam
from nets.unet import Unet
from nets.loss import CE, dice_loss_with_CE, Focal_Loss, dice_loss_with_Focal_Loss
from utils.metrics import Iou_score, f_score
from utils.dataloader import UnetDatasetGenerator
import tensorflow as tf
import configparser

config = configparser.ConfigParser()
config.read('unet_config.cfg')


os.environ['CUDA_VISIBLE_DEVICES'] = config.get('default', 'CUDA_VISIBLE_DEVICES')



if __name__ == "__main__":    
    log_dir = config.get('default', 'log_dir')
    img_size = config.getint('default', 'imgsize')
    inputs_size = [img_size,img_size,3]
    #分类个数
    num_classes = config.getint('default','num_class')
    dice_loss = config.getboolean('train', 'dice_loss')
    dataset_path = config.get('default', 'dataset_path')
    # 获取model
    model = Unet(inputs_size,num_classes)
    model_path = config.get('train', 'pretrain_model')
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    with open(os.path.join(dataset_path, "train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(dataset_path, "val.txt"),"r") as f:
        val_lines = f.readlines()

    checkpointPath = log_dir + 'tmp.h5'
    checkpoint_period = ModelCheckpoint(checkpointPath,
                                    monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir)

    freeze_layers = 17
    for i in range(freeze_layers): model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))
    if True:
        lr              = config.getfloat('train', 'freeze_learning_rate')
        Init_Epoch      = config.getint('train', 'Init_Epoch')
        Freeze_Epoch    = config.getint('train', 'Freeze_Epoch')
        Batch_size      = config.getint('train', 'Batch_size')
        # 可以使用focal loss的损失
        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])

        gen             = UnetDatasetGenerator(Batch_size, train_lines, inputs_size, num_classes, dataset_path).generate()
        gen_val         = UnetDatasetGenerator(Batch_size, val_lines, inputs_size, num_classes, dataset_path).generate()

        epoch_size      = len(train_lines) // Batch_size
        epoch_size_val  = len(val_lines) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))
        # linux 系统上不需要loss_history
        model.fit_generator(gen,
                steps_per_epoch=epoch_size,
                validation_data=gen_val,
                validation_steps=epoch_size_val,
                epochs=Freeze_Epoch,
                initial_epoch=Init_Epoch,
                callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard])
    
    
    for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        lr              = config.getfloat('train', 'learning_rate')
        Freeze_Epoch    = config.getint('train', 'Freeze_Epoch')
        Unfreeze_Epoch  = config.getint('train', 'Unfreeze_Epoch')
        Batch_size      = config.getint('train', 'Batch_size')
        # 可以使用focal loss的损失
        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])

        gen             = UnetDatasetGenerator(Batch_size, train_lines, inputs_size, num_classes, dataset_path).generate()
        gen_val         = UnetDatasetGenerator(Batch_size, val_lines, inputs_size, num_classes, dataset_path).generate(False)
        
        epoch_size      = len(train_lines) // Batch_size
        epoch_size_val  = len(val_lines) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))
        # linux 系统上不需要loss_history
        model.fit_generator(gen,
                steps_per_epoch=epoch_size,
                validation_data=gen_val,
                validation_steps=epoch_size_val,
                epochs=Unfreeze_Epoch,
                initial_epoch=Freeze_Epoch,
                callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard])

                
