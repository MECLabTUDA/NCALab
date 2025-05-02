#!/usr/bin/env python3
import sys
import os
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import DepthNCAModel, WEIGHTS_PATH, get_compute_device

import click

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt  # type: ignore[import-untyped]

import torch
import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]
from transformers import pipeline

import cv2

import open3d as o3d


plt.rcParams["savefig.bbox"] = "tight"

COLOR = "#202020"
plt.rcParams["text.color"] = COLOR
plt.rcParams["axes.labelcolor"] = COLOR
plt.rcParams["xtick.color"] = COLOR
plt.rcParams["ytick.color"] = COLOR


def render3D(rgb, d, create_mesh=True):
    f = 156
    fx = f
    fy = f
    cx = 168
    cy = 168

    rgb = (255 * rgb).astype(np.uint8)
    # rgb = cv2.fisheye.undistortImage(rgb, K=intrinsics, D=np.array([k1, k2, k1, k2]))
    # d = cv2.fisheye.undistortImage(d, K=intrinsics, D=np.array([k1, k2, k1, k2]))

    color = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(d)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False, depth_scale=1
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(336, 336, fx, fy, cx, cy)

    pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pointcloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    if create_mesh:
        pointcloud = pointcloud.voxel_down_sample(voxel_size=0.01)
        pointcloud.estimate_normals()

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pointcloud, depth=10
        )[0]
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()

        # vis.add_geometry(mesh)

        # vis.run()
        # vis.destroy_window()
        # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True,
        #    zoom=0.69,
        #    front=[-0.046321999451160951, 0.77562734499013297, 0.62948907382924757],
        #    lookat=[-0.048960089683532715, -0.06241309642791748, 0.8695310652256012],
        #    up=[-0.010939072833973022, -0.63052160343289188, 0.77609461039872452],)
    else:

        o3d.visualization.draw_geometries(
            [pointcloud],
            zoom=0.69,
            front=[-0.046321999451160951, 0.77562734499013297, 0.62948907382924757],
            lookat=[-0.048960089683532715, -0.06241309642791748, 0.8695310652256012],
            up=[-0.010939072833973022, -0.63052160343289188, 0.77609461039872452],
        )


def make_vignette_kvasir(corner_size=(55, 55), image_size=(336, 336), offset=(2, 2)):
    vignette = np.ones(image_size)
    ur = np.triu(np.ones(corner_size)).astype(bool)
    ul = ~np.flip(ur, axis=1).astype(bool)
    bl = np.tril(np.ones(corner_size)).astype(bool)
    br = ~np.flip(bl, axis=1).astype(bool)
    vignette[image_size[0] - corner_size[0] :, 0 : corner_size[1]] *= ur
    vignette[0 : corner_size[0], 0 : corner_size[1]] *= ul
    vignette[0 : corner_size[0], image_size[1] - corner_size[1] :] *= bl
    vignette[image_size[0] - corner_size[0] :, image_size[1] - corner_size[1] :] *= br
    vignette[0 : offset[0], :] = 0
    vignette[-offset[0] :, :] = 0
    vignette[:, 0 : offset[1]] = 0
    vignette[:, -offset[1] :] = 0
    return vignette


vignette = make_vignette_kvasir()
vignette_rgb = np.stack((vignette, vignette, vignette), axis=-1).astype(bool)
vignette_rgb.shape


@click.command()
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
@click.option("--hidden-channels", "-H", default=20, type=int)
def eval_depth(gpu: bool, gpu_index: int, hidden_channels: int):
    device = "cpu"  # get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    # eNCApsulateD
    nca = DepthNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=1,
        pad_noise=True,
    ).to(device)
    nca.load_state_dict(
        torch.load(WEIGHTS_PATH / "depth_kvasircapsule.best.pth", weights_only=True)
    )
    nca.eval()
    INPUT_SIZE = 64
    T = A.Compose(
        [
            A.CenterCrop(320, 320),
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            ToTensorV2(),
        ]
    )

    # Depth Anything V2
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device="cuda:0",
    )

    # MiDaS
    midas_model_type = (
        "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    )
    midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if midas_model_type == "DPT_Large" or midas_model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    fig, ax = plt.subplots(ncols=5, nrows=4, squeeze=True)

    benchmarks_path = Path("tasks/depth/benchmark")
    benchmark_name = ["COMPLEX\nFOLDS", "BUBBLE", "FOREIGN\nBODY", "DEBRIS", "BLOOD"]

    for i, benchmark_path in enumerate(sorted(benchmarks_path.glob("*"))):
        ax[0, i].set_title(benchmark_name[i], fontname="Calibri", weight="bold")
        ax[0, 0].set_ylabel("Image", fontname="Calibri", weight="bold")
        for image_path in benchmark_path.glob("*.jpg"):
            image = Image.open(image_path)
            image = np.asarray(image, dtype=np.float32) / 255.0
            image = T(image=image)["image"].to(device).unsqueeze(0)
            image = image[0].permute(1, 2, 0)
            image = image.cpu().numpy()
            ax[0, i].imshow(image, aspect="auto")
            ax[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            break

    ### DAv2
    for i, benchmark_path in enumerate(sorted(benchmarks_path.glob("*"))):
        for j, image_path in enumerate(benchmark_path.glob("*.jpg")):
            image = Image.open(image_path)
            depth = pipe(image)["depth"]
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            depth = np.asarray(depth)
            depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
            depth = np.uint8(255 * depth)
            depth[vignette == 0] = 0
            if j == 0:
                # render3D(image, depth)
                ax[1, i].imshow(depth, cmap="magma", aspect="auto")
                ax[1, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                ax[1, 0].set_ylabel(
                    "DAv2\n(Pseudo GT)", fontname="Calibri", weight="bold"
                )
                break

    ### MiDaS
    for i, benchmark_path in enumerate(sorted(benchmarks_path.glob("*"))):
        for j, image_path in enumerate(benchmark_path.glob("*.jpg")):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

            input_batch = transform(image).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                depth = prediction.cpu().numpy()
            depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
            depth = np.uint8(255 * depth)
            depth[vignette == 0] = 0

            if j == 0:
                # render3D(image, depth)
                ax[2, i].imshow(depth, cmap="magma", aspect="auto")
                ax[2, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                ax[2, 0].set_ylabel("MiDaS", fontname="Calibri", weight="bold")
                break

    ### eNCApsulateD
    for i, benchmark_path in enumerate(sorted(benchmarks_path.glob("*"))):
        for j, image_path in enumerate(benchmark_path.glob("*.jpg")):
            image = Image.open(image_path)
            image = np.asarray(image, dtype=np.float32) / 255.0
            image_T = T(image=image)["image"].to(device).unsqueeze(0)
            with torch.no_grad():
                depth = nca.estimate_depth(image_T, steps=80)
                depth = depth[0, :, :, 0]
            depth = depth.cpu().numpy()
            depth = cv2.resize(depth, dsize=(336, 336), interpolation=cv2.INTER_CUBIC)
            depth = (depth - np.min(depth[vignette != 0])) / (
                np.max(depth[vignette != 0]) - np.min(depth[vignette != 0])
            )
            depth = np.uint8(255 * depth)
            depth[vignette == 0] = 0

            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

            # if i == 4 and j == 0:
            #    print(i, j)
            #    render3D(image, depth)

            if j == 0:
                ax[3, i].imshow(depth, cmap="magma", aspect="auto")
                ax[3, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                ax[3, 0].set_ylabel(
                    "eNCApsulate\n(ours)", fontname="Calibri", weight="bold"
                )
                break
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig("depth_comparison.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    eval_depth()
