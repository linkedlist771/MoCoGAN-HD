"""
For this calculation, we make it simple.
"""

import os
import torch
from torchvision.io import write_video
from torchvision.utils import save_image
from pathlib import Path
from options.test_options import TestOptions
from models.models import create_model
import tqdm


def sort_by_file_name_int(file_paths: list):
    return sorted(file_paths, key=lambda x: int(x.stem))


class InferenceOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)
        self.parser.add_argument(
            "--real_images_dir",
            type=str,
            required=True,
            help="input directory",
        )
        self.parser.add_argument(
            "--split_ratio",
            type=float,
            default=0.9,
            help="Ratio of training data to total data",
        )

def test():

    opt = InferenceOptions().parse(save=False)

    ### initialize models
    modelG = create_model(opt)

    z = torch.FloatTensor(1, opt.latent_dimension)
    z = z.cuda()

    os.makedirs(opt.results_dir, exist_ok=True)

    real_images_dir = Path(opt.real_images_dir)

    with torch.no_grad():
        # Add tqdm for the main directory iteration
        sub_dirs = [d for d in real_images_dir.iterdir() if d.is_dir()]
        val_sub_dirs = sub_dirs[:int(len(sub_dirs) * opt.split_ratio)]
        for sub_dir in tqdm.tqdm(val_sub_dirs, desc="Processing directories"):
            if sub_dir.is_dir():
                images_paths = list(sub_dir.glob("*.png"))  # 只获取PNG文件
                images_number = len(images_paths)
                
                z.data.normal_()
                x_fake, _, _ = modelG(
                    styles=[z],
                    n_frame=images_number,
                    use_noise=False,
                    interpolation=opt.interpolation,
                )
                x_fake = x_fake.view(1, -1, 3, opt.style_gan_size, opt.style_gan_size).data
                x_fake = x_fake.clamp(-1, 1)

                save_sub_dir = os.path.join(opt.results_dir, sub_dir.stem)
                # 创建保存图片的目录
                os.makedirs(save_sub_dir, exist_ok=True)

                # Add tqdm for the frame processing
                for i, (frame, image_name) in enumerate(tqdm.tqdm(zip(x_fake[0], images_paths), 
                                                                total=len(images_paths),
                                                                desc=f"Processing frames in {sub_dir.stem}")):
                    frame = (frame + 1) / 2
                    save_image(frame, os.path.join(save_sub_dir, f"{image_name.stem}.png"))



if __name__ == "__main__":
    test()
