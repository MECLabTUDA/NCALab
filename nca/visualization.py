import matplotlib.pyplot as plt # type: ignore[import-untyped]
import numpy as np


def show_image_row(
    ax,
    images,
    vmin=None,
    vmax=None,
    cmap=None,
    overlays=None,
    overlay_vmin=None,
    overlay_vmax=None,
    overlay_cmap=None,
):
    for j in range(len(images)):
        image = images[j]
        ax[j].imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
        if overlays is not None:
            ax[j].imshow(
                overlays[j],
                vmin=overlay_vmin,
                vmax=overlay_vmax,
                cmap=overlay_cmap,
                alpha=0.5,
                colormap="jet",
            )
        ax[j].axis("off")


def show_batch_binary_image_classification(x_seed, x_pred, y_true, nca):
    batch_size = x_pred.shape[0]
    image_width = x_pred.shape[1]
    image_height = x_pred.shape[2]

    figure, ax = plt.subplots(
        2, batch_size, figsize=[batch_size * 2, 5], tight_layout=True
    )

    # 1st row: input image
    images = (x_seed[:, :, :, 0] > 0).astype(np.float32)
    for i in range(batch_size):
        images[i, :, :] *= y_true[i] + 1
    images -= 1
    show_image_row(ax[0], images, vmin=-1, vmax=nca.num_classes, cmap="Set3")

    # 2nd row: prediction
    images = np.ones((batch_size, image_width, image_height))
    class_channels = x_pred[..., nca.num_image_channels + nca.num_hidden_channels :]
    y_pred = np.argmax(class_channels, axis=-1)
    images = (x_seed[:, :, :, 0] > 0).astype(np.float32)
    for i in range(batch_size):
        images[i, :, :] *= y_pred[i] + 1
    images -= 1
    show_image_row(ax[1], images, vmin=-1, vmax=nca.num_classes, cmap="Set3")
    return figure


def show_batch_classification(x_seed, x_pred, y_true, nca):
    batch_size = x_pred.shape[0]
    image_width = x_pred.shape[1]
    image_height = x_pred.shape[2]

    print(batch_size, image_width, image_height)

    figure, ax = plt.subplots(
        3, batch_size, figsize=[batch_size * 2, 5], tight_layout=True
    )

    # 1st row: input image
    images = np.ones((batch_size, image_width, image_height))
    hidden_channels = x_pred[..., nca.num_image_channels : -nca.num_output_channels]
    class_channels = x_pred[..., nca.num_image_channels + nca.num_hidden_channels :]
    images = x_seed[:, :, :, : nca.num_image_channels].astype(np.float32)
    show_image_row(ax[0], np.clip(images, 0, 1))

    for i in range(batch_size):
        mask = np.max(hidden_channels[i]) > 0.1
        class_channels[i] *= mask

    y_pred = np.mean(class_channels, 1)
    y_pred = np.mean(y_pred, 1)
    y_pred = np.argmax(y_pred, axis=-1)

    # 2nd row: prediction vs. true
    for j in range(batch_size):
        ax[1, j].text(0, 1, f"true: {y_true[j][0]}")
        ax[1, j].text(0, 0.9, f"pred: {y_pred[j]}")
        ax[1, j].axis("off")

    # 3rd row: predicted classes per pixel
    images = np.ones((batch_size, image_width, image_height))
    class_channels = x_pred[..., nca.num_image_channels + nca.num_hidden_channels :]
    y_pred = np.argmax(class_channels, axis=-1)
    images = (x_seed[:, :, :, 0] > 0).astype(np.float32)
    for i in range(batch_size):
        images[i, :, :] *= y_pred[i] + 1
    images -= 1
    show_image_row(ax[2], images, vmin=-1, vmax=nca.num_classes, cmap="Set3")

    return figure


def show_batch_binary_segmentation(x_seed, x_pred, y_true, nca):
    batch_size = x_pred.shape[0]

    figure, ax = plt.subplots(
        3, batch_size, figsize=[batch_size * 3, 5], tight_layout=True
    )

    # 1st row: true image
    images = x_seed[..., : nca.num_image_channels]
    show_image_row(ax[0], np.clip(images, 0.0, 1.0))

    # 2nd row: prediction
    images = y_true
    show_image_row(ax[1], np.clip(images, 0.0, 1.0), cmap="gray")

    # 3rd row: prediction
    images = x_pred[..., -nca.num_classes:]
    show_image_row(ax[2], np.clip(images, 0.0, 1.0), cmap="gray")

    plt.tight_layout()

    return figure


def show_batch_growing(x_seed, x_pred, y_true, nca):
    batch_size = x_pred.shape[0]

    figure, ax = plt.subplots(
        2, batch_size, figsize=[batch_size * 2, 5], tight_layout=True
    )

    # 1st row: true image
    images = y_true[..., : nca.num_image_channels]
    show_image_row(ax[0], images)

    # 2nd row: prediction
    images = x_pred[..., : nca.num_image_channels]
    show_image_row(ax[1], np.clip(images, 0.0, 1.0))

    return figure
