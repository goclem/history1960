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


def standardise_image(image:np.ndarray) -> np.ndarray:
    bandmeans   = np.mean(image, axis=(0, 1), keepdims=True)
    bandstds    = np.std(image,  axis=(0, 1), keepdims=True)
    standarised = (image - bandmeans) / bandstds
    return standarised


def display_statistics(image_test:np.ndarray, label_test:np.ndarray, proba_pred:np.ndarray, label_pred:np.ndarray, masks:np.ndarray) -> None:
        # Augmented images
        colour  = (255, 255, 0)
        images  = [np.where(np.tile(mask, (1, 1, 3)), colour, image_test) for mask in masks]
        # Figure
        images = [image_test, label_test, proba_pred, label_pred] + images
        titles = ['Test image', 'Test label', 'Predicted probability', 'Predicted label', 'True positive', 'True negative', 'False positive', 'False negative']
        fig, axs = pyplot.subplots(2, 4, figsize=(20, 10))
        for image, title, ax in zip(images, titles, axs.ravel()):
            ax.imshow(image)
            ax.set_title(title, fontsize=20)
            ax.axis('off')
        pyplot.tight_layout(pad=2.0)
        pyplot.show()
