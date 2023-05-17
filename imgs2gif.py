import glob
import contextlib
from PIL import Image

# filepaths
# fp_in = "outputs/big_hotdog/*.png" 
# fp_out = "outputs/big_hotdog/img.gif"
# fp_in = glob.glob("data/nerf_synthetic/hotdog/test/*.png")
fp_in = ["data/nerf_synthetic/hotdog/test/r_" + str(i) + ".png" for i in range(0, 200)]
fp_in = [file for file in fp_in if "depth" not in file]
fp_out = "outputs/hotdog_gt.gif" 
# print(fp_in)

# use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:

    # lazily load images
    imgs = (stack.enter_context(Image.open(f)) for f in fp_in)

    # extract  first image from iterator
    img = next(imgs)

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=100, loop=0)