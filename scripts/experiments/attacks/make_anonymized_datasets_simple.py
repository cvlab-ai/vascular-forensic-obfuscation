from angio_gen.utils.anonimizer import make_anon_dataset_from_images
from angio_gen.utils.simple_anonimizers import *



if __name__ == "__main__":
    make_anon_dataset_from_images(LocalContrastNormAnonimizer(), output_dir="aia_dataset_anon_cn")
    make_anon_dataset_from_images(BorderCropAnonimizer(), output_dir="aia_dataset_anon_bc")
    make_anon_dataset_from_images(LaplaceNoiseAnonimizer(), output_dir="aia_dataset_anon_ln")
    make_anon_dataset_from_images(MedianFilterAnonimizer(), output_dir="aia_dataset_anon_mf")
    make_anon_dataset_from_images(QuantizationAnonimizer(), output_dir="aia_dataset_anon_qu")
    make_anon_dataset_from_images(JPEGCompressionAnonimizer(), output_dir="aia_dataset_anon_jpeg")
