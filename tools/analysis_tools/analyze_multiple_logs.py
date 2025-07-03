import subprocess
from pathlib import Path


def run_test(log_path, out_file):
    """"""
    cmd = [
        "python", "analyze_logs.py",
        log_path,
        "--out", out_file,
    ]
    # print(cmd)
    subprocess.run(cmd, check=True)


def main():
    log_files = [Path("/home/bhamscher/Masterthesis/mmseg_cityscapes/work_dirs/deeplabv3plus_stylized_voronoi8/20241104_175132/vis_data/20241104_175132.json"),
                 Path("/home/bhamscher/Masterthesis/mmseg_cityscapes/work_dirs/deeplabv3plus_stylized_voronoi8/20241029_233546/vis_data/20241029_233546.json"),
                 Path("/home/bhamscher/Masterthesis/mmseg_cityscapes/work_dirs/deeplabv3plus_stylized_voronoi16/20241028_182831/vis_data/20241028_182831.json"),
                 Path("/home/bhamscher/Masterthesis/mmseg_cityscapes/work_dirs/deeplabv3plus_stylized_voronoi16/20241105_151515/vis_data/20241105_151515.json"),
                 Path("/home/bhamscher/Masterthesis/mmseg_cityscapes/work_dirs/deeplabv3plus_stylized_voronoi16_v1/20241112_025317/vis_data/20241112_025317.json"),
                 Path("/home/bhamscher/Masterthesis/mmseg_cityscapes/work_dirs/cs_baseline/20240911_152901/vis_data/20240911_152901.json"),
                 Path("/home/bhamscher/Masterthesis/mmseg_cityscapes/work_dirs/deeplabv3plus_stylized_config/20241006_013045/vis_data/20241006_013045.json"),
                 Path("/home/bhamscher/Masterthesis/mmseg_cityscapes/work_dirs/deeplabv3plus_stylized_panoptic_50percent_config/20241010_220538/vis_data/20241010_220538.json"),
                 Path("/home/bhamscher/Masterthesis/mmseg_cityscapes/work_dirs/deeplabv3plus_stylized_panoptic_config/20240925_174458/vis_data/20240925_174458.json")]

    for log_file in log_files:
        out_file = log_file.parent / Path("loss_plot.png")
        # print(log_file)
        # print(out_file)
        run_test(str(log_file), str(out_file))

if __name__ == "__main__":
    main()

