# # Writes samples
# keys = ['images_train', 'labels_train', 'images_valid', 'labels_valid', 'images_test', 'labels_test']
# for directory in [paths[key] for key in keys]:
#     initialise_directory(directory=directory, remove=True)

# def filename(key:str, i:int) -> str:
#     directory = paths[key.replace('_', 's_')]
#     filename  = path.join(directory, '{}_{:05d}.jpeg'.format(key, i))
#     return filename

# filename('label_test', 0)
# for i in range(images_train.shape[0]):
#     pyplot.imsave(fname=filename('image_train', i), arr=images_train[i])
#     pyplot.imsave(fname=filename('label_train', i), arr=labels_train[i])

# for i in range(images_valid.shape[0]):
#     pyplot.imsave(fname=filename('image_valid', i), arr=images_valid[i])
#     pyplot.imsave(fname=filename('label_valid', i), arr=labels_valid[i])

# for i in range(images_test.shape[0]):
#     pyplot.imsave(fname=filename('image_test', i), arr=images_test[i])
#     pyplot.imsave(fname=filename('label_test', i), arr=labels_test[i])

# del(images_train, labels_valid, images_valid, labels_valid, images_test, labels_test)
