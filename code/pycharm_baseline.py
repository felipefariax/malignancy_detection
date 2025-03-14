import torch

# Download an example image
import urllib
url, filename = ("https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png", "TCGA_CS_4944.png")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)

import numpy as np
from PIL import Image
from torchvision import transforms

input_image = Image.open(filename)
m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=m, std=s),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model = model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

print(torch.round(output[0]))

# #####################################################
# from glob import glob
# import os
#
# import cv2
# import numpy as np  # linear algebra
# import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# from tensorflow.keras.applications.nasnet import preprocess_input
# from torchvision import models
#
# from utils import get_id_from_file_path, data_gen, chunker
#
# from utils import get_seq, save_as_images
#
# datasets = {}
# id_label_map = {}
# for subset in ['train' , 'valid', 'test']:
#     datasets[subset] = glob(f'/data/pcam/{subset}/**/*.tif')
#     id_label_map.update({os.path.basename(file).replace('.tif', ''): 0 for file in glob(f'/data/pcam/{subset}/0/*.tif')})
#     id_label_map.update({os.path.basename(file).replace('.tif', ''): 1 for file in glob(f'/data/pcam/{subset}/1/*.tif')})
#
# model = models.resnext50_32x4d(pretrained=True)
#
# batch_size = 32
# h5_path = "model.h5"
# # checkpoint = ModelCheckpoint(h5_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#
# from keras.utils import HDF5Matrix
# from keras.preprocessing.image import ImageDataGenerator
# dir = '/home/fcf/data/pcam/train/'
# dir = '/data/pcam/'
# # x_train = HDF5Matrix(os.path.join(dir, 'train/camelyonpatch_level_2_split_train_x.h5'), 'x')
# # x_train_mask = HDF5Matrix(os.path.join(dir, 'train/camelyonpatch_level_2_split_train_mask.h5'), 'mask')
# # y_train = HDF5Matrix(os.path.join(dir, 'train/camelyonpatch_level_2_split_train_y.h5'), 'y')
# # save_as_images('/data/pcam/train', x_train, y_train, x_train_mask)
# # x_valid = HDF5Matrix(os.path.join(dir, 'valid/camelyonpatch_level_2_split_valid_x.h5'), 'x')
# # y_valid = HDF5Matrix(os.path.join(dir, 'valid/camelyonpatch_level_2_split_valid_y.h5'), 'y')
# # save_as_images('/data/pcam/valid', x_valid, y_valid, None)
# # x_test = HDF5Matrix(os.path.join(dir, 'test/camelyonpatch_level_2_split_test_x.h5'), 'x')
# # y_test = HDF5Matrix(os.path.join(dir, 'test/camelyonpatch_level_2_split_test_y.h5'), 'y')
# # save_as_images('/data/pcam/test', x_test, y_test, None)
# # exit(0)
#
# # seq = get_seq()
# # datagen = ImageDataGenerator(
# #     # preprocessing_function=lambda x: preprocess_input(seq.augment_images(x)),
# #     width_shift_range=4,  # randomly shift images horizontally
# #     height_shift_range=4,  # randomly shift images vertically
# #     horizontal_flip=True,  # randomly flip images
# #     vertical_flip=True)  # randomly flip images
# #
# #
# # model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), validation_data=datagen.flow(x_valid, y_valid, batch_size=batch_size), callbacks=[checkpoint], steps_per_epoch=len(x_train) // batch_size, epochs=2)
#
# _ = model.fit_generator(
#     data_gen(datasets['train'], id_label_map, batch_size, augment=False),
#     # validation_data=data_gen(datasets['valid'], id_label_map, batch_size),
#     epochs=2, verbose=2,
#     # callbacks=[checkpoint],
#     steps_per_epoch=len(datasets['train']) // batch_size,
#     # validation_steps=len(datasets['valid']) // batch_size)
#     # batch_size = 64
#     # _ = model.fit_generator(
#     #     data_gen(train, id_label_map, batch_size, augment=True),
#     #     validation_data=data_gen(val, id_label_map, batch_size),
#     #     epochs=6, verbose=1,
#     #     callbacks=[checkpoint],
#     #     steps_per_epoch=len(train) // batch_size,
#     #     validation_steps=len(val) // batch_size)
#
#     print("Finished!")
# model.load_weights(h5_path)
#
# preds = []
# ids = []
#
# for batch in chunker(datasets['test'], batch_size):
#     X = [preprocess_input(cv2.resize(cv2.imread(x), (224, 224))) for x in batch]
# ids_batch = [get_id_from_file_path(x) for x in batch]
# X = np.array(X)
# preds_batch = ((model.predict(X).ravel() * model.predict(X[:, ::-1, :, :]).ravel() * model.predict(
#     X[:, ::-1, ::-1, :]).ravel() * model.predict(X[:, :, ::-1, :]).ravel()) ** 0.25).tolist()
# preds += preds_batch
# ids += ids_batch
#
# df = pd.DataFrame({'id': ids, 'label': preds})
# df.to_csv("baseline_nasnet.csv", index=False)
# df.head()
