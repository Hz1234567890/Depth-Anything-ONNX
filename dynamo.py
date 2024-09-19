from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Optional
import os
import re
import numpy as np
import cv2
from MySobel import compute_normal_vectors,visualize_normals 
import onnxruntime as ort
import torch
import typer

from depth_anything_v2.config import Encoder, Metric
from depth_anything_v2.dpt import DepthAnythingV2

def find_nxn_region_from_center(depth_matrix, threshold, n):
    h, w = depth_matrix.shape
    center_i, center_j = h // 2, w // 2

    # 以中心为起点，逐步向外扩展搜索
    for radius in range(min(h, w) // 2):
        for i in range(max(0, center_i - radius), min(h - (n-1), center_i + radius + 1)):
            for j in range(max(0, center_j - radius), min(w - (n-1), center_j + radius + 1)):
                region = depth_matrix[i:i+n, j:j+n]
                region_range = region.max() - region.min()
                if region_range <= threshold:
                    return (i, j), region
    return None, None

class ExportFormat(StrEnum):
    onnx = auto()
    pt2 = auto()


class InferenceDevice(StrEnum):
    cpu = auto()
    cuda = auto()


app = typer.Typer()


@app.callback()
def callback():
    """Depth-Anything Dynamo CLI"""


def multiple_of_14(value: int) -> int:
    if value % 14 != 0:
        raise typer.BadParameter("Value must be a multiple of 14.")
    return value


@app.command()
def export(
    encoder: Annotated[Encoder, typer.Option()] = Encoder.vitb,
    metric: Annotated[
        Optional[Metric], typer.Option(help="Export metric depth models.")
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            dir_okay=False,
            writable=True,
            help="Path to save exported model.",
        ),
    ] = None,
    format: Annotated[
        ExportFormat, typer.Option("-f", "--format", help="Export format.")
    ] = ExportFormat.onnx,
    batch_size: Annotated[
        int,
        typer.Option(
            "-b",
            "--batch-size",
            min=0,
            help="Batch size of exported ONNX model. Set to 0 to mark as dynamic (opset <= 17).",
        ),
    ] = 1,
    height: Annotated[
        int,
        typer.Option(
            "-h",
            "--height",
            min=0,
            help="Height of input image. Set to 0 to mark as dynamic (opset <= 17).",
            callback=multiple_of_14,
        ),
    ] = 518,
    width: Annotated[
        int,
        typer.Option(
            "-w",
            "--width",
            min=0,
            help="Width of input image. Set to 0 to mark as dynamic (opset <= 17).",
            callback=multiple_of_14,
        ),
    ] = 518,
    opset: Annotated[
        int,
        typer.Option(
            max=18,
            help="ONNX opset version of exported model. Defaults to 18 (export via TorchDynamo).",
        ),
    ] = 18,
):
    """Export Depth-Anything V2 using TorchDynamo."""
    if encoder == Encoder.vitg:
        raise NotImplementedError("Depth-Anything-V2-Giant is coming soon.")

    if torch.__version__ < "2.3":
        typer.echo(
            "Warning: torch version is lower than 2.3, export may not work properly."
        )

    if output is None:
        output = Path(f"weights/depth_anything_v2_{encoder}_{opset}.{format}")

    config = encoder.get_config(metric)
    model = DepthAnythingV2(
        encoder=encoder.value,
        features=config.features,
        out_channels=config.out_channels,
        max_depth=20
        if metric == Metric.indoor
        else 80
        if metric == Metric.outdoor
        else None,
    )
    model.load_state_dict(torch.hub.load_state_dict_from_url(config.url))

    if format == ExportFormat.onnx:
        if opset == 18:
            onnx_program = torch.onnx.dynamo_export(
                model, torch.randn(batch_size, 3, 518, 518)
            )
            onnx_program.save(str(output))
        else:  # <= 17
            typer.echo("Exporting to ONNX using legacy JIT tracer.")
            dynamic_axes = {}
            if batch_size == 0:
                dynamic_axes[0] = "batch_size"
            if height == 0:
                dynamic_axes[2] = "height"
            if width == 0:
                dynamic_axes[3] = "width"
            torch.onnx.export(
                model,
                torch.randn(batch_size or 1, 3, height or 140, width or 140),
                str(output),
                input_names=["image"],
                output_names=["depth"],
                opset_version=opset,
                dynamic_axes={"image": dynamic_axes, "depth": dynamic_axes},
            )
    elif format == ExportFormat.pt2:
        batch_dim = torch.export.Dim("batch_size")
        export_program = torch.export.export(
            model.eval(),
            (torch.randn(2, 3, 518, 518),),
            dynamic_shapes={
                "x": {0: batch_dim},
            },
        )
        torch.export.save(export_program, output)


@app.command()
def infer(
    model_path: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Path to ONNX model."
        ),
    ],
    folder_path: Annotated[
        Path,
        typer.Option(
            "-f",
            "--folder",
            exists=True,
            dir_okay=True,
            readable=True,
            help="Path to input folder_path.",
        ),
    ],
    height: Annotated[
        int,
        typer.Option(
            "-h",
            "--height",
            min=14,
            help="Height at which to perform inference. The input image will be resized to this.",
            callback=multiple_of_14,
        ),
    ] = 518,
    width: Annotated[
        int,
        typer.Option(
            "-w",
            "--width",
            min=14,
            help="Width at which to perform inference. The input image will be resized to this.",
            callback=multiple_of_14,
        ),
    ] = 518,
    device: Annotated[
        InferenceDevice, typer.Option("-d", "--device", help="Inference device.")
    ] = InferenceDevice.cuda,
    output_folder_path: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            dir_okay=True,
            writable=True,
            help="Path to save output depth map. If not given, show visualization.",
        ),
    ] = None,
):
    """Depth-Anything V2 inference using ONNXRuntime. No dependency on PyTorch."""
    # Preprocessing, implement this part in your chosen language:
    # 获取文件夹中的所有文件名
    files = os.listdir(folder_path)
    jpg_files = [file for file in files if file.endswith('.jpg')]

    n=50 #差异性区域
    threshold = 0.2  # 设置差异性阈值

    # 按顺序重命名文件
    for index, file in enumerate(jpg_files):
        print(index,file)
        image_path=os.path.join(folder_path, file)

        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        original_image = image.copy()  # 保留原始图像用于标注
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = image.transpose(2, 0, 1)[None].astype("float32")

        # Inference
        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = False
        # For inspecting applied ORT-optimizations:
        # sess_options.optimized_model_filepath = "weights/optimized.onnx"
        providers = ["CPUExecutionProvider"]
        if device == InferenceDevice.cuda:
            providers.insert(0, "CUDAExecutionProvider")

        session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        binding = session.io_binding()
        ort_input = session.get_inputs()[0].name
        binding.bind_cpu_input(ort_input, image)
        ort_output = session.get_outputs()[0].name
        binding.bind_output(ort_output, device.value)

        session.run_with_iobinding(binding)  # Actual inference happens here.

        depth = binding.get_outputs()[0].numpy()

        # 后处理
        depth = depth.squeeze()  # 移除多余的批量维度
        print("depth:")
        print(depth)
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype("uint8")
        depth_resized = cv2.resize(depth_normalized, (w, h), interpolation=cv2.INTER_CUBIC)

        # # 查找满足条件的5x5区域
        # region_position, region = find_nxn_region_from_center(depth, threshold, n)
        # if region is not None:
        #     print(f"Found {n}x{n} region at position {region_position} with depth range within {threshold}")
        #     print("Region:")
        #     print(region)

        #     # 标注原始RGB图像
        #     i, j = region_position
        #     top_left = (j, i)
        #     bottom_right = (j + n, i + n)
        #     cv2.rectangle(original_image, top_left, bottom_right, (0, 255, 0), 2)

            
        # else:
        #     print(f"No {n}x{n} region found in {file} with depth range within {threshold}")

        # # 保存标注后的RGB图像
        # annotated_image_filename = f"annotated_{file}"
        # annotated_image_path = os.path.join(output_folder_path, annotated_image_filename)
        # cv2.imwrite(annotated_image_path, original_image)

        # # 保存或显示深度图
        # if output_folder_path is None:
        #     cv2.imshow("depth", depth_resized)
        #     cv2.waitKey(0)
        # else:
        #     numbers = re.findall(r'\d+', file)
        #     new_image_filename = f"depth_{numbers[0]}.png"
        #     image_output_path = os.path.join(output_folder_path, new_image_filename)
        #     cv2.imwrite(str(image_output_path), depth_resized)

        #     new_matrix_filename = f"depth_matrix_{numbers[0]}.npy"
        #     matrix_output_path = os.path.join(output_folder_path, new_matrix_filename)
        #     np.save(matrix_output_path, depth)  # 保存深度矩阵
        # 假设depth_matrix是我们给定的深度矩阵

        #获取normal_vectors的文件名
        normals_filename = f"annotated_{file}"
        normals_path = os.path.join(output_folder_path, normals_filename)

        normal_x, normal_y, normal_z = compute_normal_vectors(depth)
        visualize_normals(normal_x, normal_y, normal_z,normals_path)

if __name__ == "__main__":
    app()
