from angio_gen.utils.anonimizer import make_anonimizer, anonimize_from_PIL
from angio_gen.utils.config import load_weight_config
from angio_gen.models.cgan import CGANModule
from pathlib import Path
from PIL import Image
from angio_gen.data.transforms import image_transform as trafo
import matplotlib.pyplot as plt
import matplotlib
import torch
from angio_gen.utils.metrics import compute_ssim

matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.port"] = 2137
matplotlib.rcParams['webagg.open_in_browser'] = False

ANZ = make_anonimizer(CGANModule, load_weight_config()["weights"]["pix2pix"])

def test_sample(img_path):
    mask_path = img_path.with_suffix(".mask.png")
    img_pil = Image.open(img_path).convert("L")

    gt = trafo(img_pil).unsqueeze(0)
    mask_pil = Image.open(mask_path).convert("L")
    x_anon = anonimize_from_PIL(ANZ, mask_pil)
    x_anon = x_anon.detach().cpu().unsqueeze(0).unsqueeze(0)
    print(x_anon.shape, gt.shape)
    ssim, _ = compute_ssim(x_anon, gt)

    print(ssim)

input_path = Path("aia_dataset") / "test" / "_0.png"
test_sample(input_path)


