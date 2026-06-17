from angio_gen.utils.anonimizer import make_anonimizer, make_anon_dataset_from_masks
from angio_gen.utils.config import load_weight_config
from angio_gen.models.diffusion import DiffusionModule


if __name__ == "__main__":
    anz = make_anonimizer(DiffusionModule, load_weight_config()["weights"]["diffusion"])
    device = anz.device
    anz.to(device)
    anz.eval()
    make_anon_dataset_from_masks(anz.sample, output_dir="aia_dataset_anon_diffusion")
