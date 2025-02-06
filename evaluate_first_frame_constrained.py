import os

import torch
from torchvision.io import write_video
from torchvision.utils import save_image
from pathlib import Path
from options.test_options import TestOptions
from models.models import create_model


raise NotImplementedError("This infer")
"""
需要传入一个这样的文件结构，然后根据对应的frame做constrain 然后作为输入，重新生成后面的subsequences。

input_constrained_frames_dir/
├── sub_dir_1/
│   ├── 0.png
│   ├── 10.png
│   └── ...
├── sub_dir_2/
│   ├── 0.png
│   ├── 10.png
│   └── ...
└── sub_dir_3/
    ├── 0.png
    ├── 10.png
    └── ...

改变思路:
1. 遍历每个sub_dir 然后以sub_dir 里面的每个constrain frame 来生成后续的frame.
2. PS: 这里可以asset一下，这里面的sub_directory 的sep 是和传入的args是一样的。    
"""




def sort_by_file_name_int(file_paths: list):
    return sorted(file_paths, key=lambda x: int(x.stem))


class FrameConstrainedOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)
        self.parser.add_argument(
            "--input_constrained_frames_dir",
            type=str,
            required=True,
            help="input directory",
        )


def test():

    opt = FrameConstrainedOptions().parse(save=False)

    ### initialize models
    modelG = create_model(opt)

    z = torch.FloatTensor(1, opt.latent_dimension)
    z = z.cuda()

    def create_and_save(z, modelG, opt, use_noise, prefix):
        x_fake, _, _ = modelG(
            styles=[z],
            n_frame=opt.n_frames_G,
            use_noise=use_noise,
            interpolation=opt.interpolation,
        )
        x_fake = x_fake.view(1, -1, 3, opt.style_gan_size, opt.style_gan_size).data
        x_fake = x_fake.clamp(-1, 1)

        # 创建保存图片的目录
        save_dir = os.path.join(opt.results_dir, prefix)
        os.makedirs(save_dir, exist_ok=True)

        # 保存每一帧为单独的图片
        for i, frame in enumerate(x_fake[0]):
            # 将像素值从[-1, 1]转换到[0, 1]
            frame = (frame + 1) / 2
            save_image(frame, os.path.join(save_dir, f"frame_{i:04d}.png"))

        print(f"Saved {opt.n_frames_G} frames to {save_dir}")

    os.makedirs(opt.results_dir, exist_ok=True)

    constrained_frames_dir = Path(opt.input_constrained_frames_dir)

    with torch.no_grad():

        for sub_dir in constrained_frames_dir.iterdir():
            if sub_dir.is_dir():
                constrained_image_paths = list(sub_dir.glob("*.png"))  # 只获取PNG文件
                sorted_constrained_image_paths = sort_by_file_name_int(
                    constrained_image_paths
                )
                for constrained_image_path in sorted_constrained_image_paths:
                    ## Infer here!!!
                    raise NotImplementedError

        for j in range(opt.num_test_videos):
            z.data.normal_()
            prefix = (
                opt.name + "_" + str(opt.load_pretrain_epoch) + "_" + str(j) + "_noise"
            )
            create_and_save(z, modelG, opt, True, prefix)

            prefix = opt.name + "_" + str(opt.load_pretrain_epoch) + "_" + str(j)
            create_and_save(z, modelG, opt, False, prefix)

        print(opt.name + " Finished!")


if __name__ == "__main__":
    test()
