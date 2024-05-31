import matplotlib.pyplot as plt
import numpy as np

row_titles = {
    "pred_class": "Predicted Class",
    "true_class": "True Class",
    "pred_class_overlay": "Predicted Class",
    "true_class_overlay": "True Class",
    "input_image": "Input Image",
    "pred_image": "Predicted Image"
}


def show_batch(x_seed, x_pred, y_true, nca):
    batch_size = 8
    image_width = x_pred.shape[1]
    image_height = x_pred.shape[2]

    rows = nca.visualization_rows
    figure, ax = plt.subplots(
        len(rows), batch_size, figsize=[batch_size * 2, 5], tight_layout=True
    )
    for i, row in enumerate(rows):
        if row not in row_titles:
            valid_row_types = ", ".join(row_titles)
            raise Exception(
                f"Unknown row type: {row}. Must be one of {valid_row_types}."
            )

        cmap = None
        vmin = None
        vmax = None
        overlay = None
        overlay_cmap = None
        overlay_vmin = None
        overlay_vmax = None

        if row == "pred_class":
            cmap = "Set3"
            vmin = -1
            vmax = nca.num_output_channels
            images = np.zeros((batch_size, image_width, image_height, 1))
            for k in range(batch_size):
                for m in range(image_width):
                    for n in range(image_height):
                        images[k, m, n] = np.argmax(x_pred[k, m, n, :-nca.num_output_channels]) + 1
            for c in range(len(y_true)):
                mask = ((np.max(x_seed[c, :, :, :nca.num_image_channels], axis=-1) > 0) * 2 - 1)
                images[c, :, :, -1] *= mask
                images[c] = np.clip(images[c], -1, nca.num_output_channels)
        elif row == "true_class":
            cmap = "Set3"
            vmin = -1
            vmax = nca.num_output_channels
            images = np.ones((batch_size, image_width, image_height))
            for c in range(len(y_true)):
                images[c] *= y_true[c]
                mask = ((np.max(x_seed[c, :, :, :nca.num_image_channels], axis=-1) > 0) * 2 - 1)
                images[c] *= mask
                images[c] = np.clip(images[c], -1, nca.num_output_channels)
        elif row == "pred_class_overlay":
            cmap_overlay = "Set3"
            images = x_seed[..., :nca.num_image_channels]
            overlay = x_pred[..., :-nca.num_output_channels]
        elif row == "true_class_overlay":
            images = x_seed[..., :nca.num_image_channels]
            overlay = y_true[..., :-nca.num_output_channels]
        elif row == "input_image":
            images = x_seed[..., :nca.num_image_channels]
        elif row == "pred_image":
            images = x_pred[..., :nca.num_image_channels]

        title_idx = batch_size // 2 - 1
        if title_idx < 0:
            title_idx = 0
        ax[i, title_idx].set_title(row_titles[row])
            
        for j in range(batch_size):
            image = images[j]
            ax[i, j].imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
            if overlay is not None:
                ax[i, j].imshow(overlay, vmin=overlay_vmin, vmax=overlay_vmax, cmap=overlay_cmap)
            ax[i, j].axis("off")
    return figure
