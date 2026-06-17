from angio_gen.utils.anonimizer import make_anonimizer, make_anon_dataset_from_masks
from angio_gen.utils.config import load_weight_config
from angio_gen.models.cgan import CGANModule


if __name__ == "__main__":
    anz = make_anonimizer(CGANModule, load_weight_config()["weights"]["pix2pix"])
    device = anz.device
    anz.to(device)
    anz.eval()
    make_anon_dataset_from_masks(anz, output_dir="aia_dataset_anon_pix2pix")
