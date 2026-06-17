'''
Data preparation script

Assumes:
- CARDIAG dataset is unpacked under `data/CARDIAG`
- XCAD dataset is unpacked under `data/XCAD`
'''
import re
import shutil
import subprocess
import random
import json
import requests
import zipfile
import io
import numpy as np

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO


DATA_DIR = Path("data")
DATASET_DIR = Path("aia_dataset")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

DATASET_DIR.mkdir(exist_ok=True)
TRAIN_DIR.mkdir(exist_ok=True)
VAL_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)


METADATA = []

sample_index = 0
def next_index():
    global sample_index
    sample_index += 1    
    return sample_index - 1



def prepare_CARDIAG(base_dir):
    base_path = Path(base_dir)
    json_files = list(base_path.glob("**/*/dcm.json"))
    
    def get_highest_numbered_file(files):
        """
        Given a list of Path objects, returns the one with the highest integer in its name.
        Example: ['0.png', '1.png'] -> returns '1.png'
        """
        if not files:
            return None
        
        def extract_number(f):
            # Finds all digits in the filename and takes the last group found
            numbers = re.findall(r'\d+', f.name)
            return int(numbers[-1]) if numbers else -1

        return max(files, key=extract_number)


    hospital_mapping = defaultdict(list)
    
    print(f"Scanning {len(json_files)} exams...")
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                h_id = data.get("InstitutionName") or \
                       data.get("(0008,0080)") or \
                       data.get("StationName") or \
                       "Unknown"
                h_id = h_id["value"]
                hospital_mapping[h_id].append(json_path.parent)
        except:
            continue

    selected_exams = []
    print("\n" + "="*40)
    print(f"{'Hospital ID':<25} | {'Found':<6} | {'Status'}")
    print("-" * 40)

    for h_name, folders in hospital_mapping.items():
        count = len(folders)
        
        if count >= 150:
            random.shuffle(folders)
            sample_size = min(count, 200)
            chosen_folders = folders[:sample_size]
            
            for folder in chosen_folders:
                selected_exams.append((folder, h_name))
                
            print(f"{h_name[:25]:<25} | {count:<6} | ACCEPTED (Taking {sample_size})")
        else:
            print(f"{h_name[:25]:<25} | {count:<6} | REJECTED (Too few samples)")

    print(f"\nExtracting {len(selected_exams)} total pairs...")
    for exam_dir, h_id in tqdm(selected_exams, desc="Finalizing Dataset"):
        img_dir = exam_dir / "img"
        mask_dir = exam_dir / "masks" / "vessels"

        img_files = list(img_dir.glob("*.png"))
        mask_files = list(mask_dir.glob("*.png"))

        if not img_files or not mask_files:
            continue

        best_mask = get_highest_numbered_file(mask_files)
        best_img = img_dir / best_mask.name

        if best_img.exists():
            ix = next_index()
            
            shutil.copy(best_img, DATASET_DIR / f"_{ix}.png")
            shutil.copy(best_mask, DATASET_DIR / f"_{ix}.mask.png")

            METADATA.append({
                "index": ix,
                "hospital_id": h_id,
                "source_path": str(exam_dir),
                "is_synthetic": False
            })

def prepare_ICANJ(icanj_dir):
    IMAGES_DIR = icanj_dir / "V1.0" / "ICA_PNG"
    MASK_DIR = icanj_dir / "V1.0" / "label"

    if not icanj_dir.exists():
        try:
            subprocess.run([
                "git", "clone",
                "https://github.com/laudominik/ICA_NJ_BinarySeg.git",
                str(icanj_dir)
            ], check=True)
        except Exception as e:
            print(f"Clone failed: {e}")
            pass

    all_valid_images = []
    for img_path in IMAGES_DIR.glob("*.png"):
        if (MASK_DIR / img_path.name).exists():
            all_valid_images.append(img_path)

    random.seed(42)
    random.shuffle(all_valid_images)
    selected_images = all_valid_images[:200]

    print(f"ICA_NJ: Found {len(all_valid_images)} valid pairs. Sampling {len(selected_images)}.")

    for img_path in tqdm(selected_images, desc="Processing ICA_NJ"):
        mask_path = MASK_DIR / img_path.name
        
        ix = next_index()
        shutil.copy(img_path, DATASET_DIR / f"_{ix}.png")
        shutil.copy(mask_path, DATASET_DIR / f"_{ix}.mask.png")
        
        METADATA.append({
            "index": ix,
            "hospital_id": "Jiangsu Province People’s Hospital",
            "original_filename": img_path.name,
        })


def prepare_DCA1(cimat_dir):
    URL = "http://personal.cimat.mx:8181/~ivan.cruz/Databases/DB_Angiograms_134.zip"
    IMAGES_DIR = cimat_dir / "Database_134_Angiograms"
    MASK_DIR = IMAGES_DIR
    
    if not cimat_dir.exists():
        print("Downloading DCA1...")
        response = requests.get(URL)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(cimat_dir)
    all_valid_pairs = []
    for img_path in IMAGES_DIR.glob("*.pgm"):
            mask_name = f"{img_path.stem}_gt.pgm"
            mask_path = MASK_DIR / mask_name
                
            if mask_path.exists():
                all_valid_pairs.append((img_path, mask_path))

    random.seed(42)
    random.shuffle(all_valid_pairs)
    selected_pairs = all_valid_pairs[:200]

    print(f"DCA1: Found {len(all_valid_pairs)} pairs. Sampling {len(selected_pairs)}.")

    for img_path, mask_path in tqdm(selected_pairs, desc="Processing CIMAT"):
        ix = next_index()
      
        with Image.open(img_path) as img:
            img.save(DATASET_DIR / f"_{ix}.png", "PNG")
        with Image.open(mask_path) as mask:
            mask.save(DATASET_DIR / f"_{ix}.mask.png", "PNG")

        METADATA.append({
            "index": ix,
            "hospital_id": "The Cardiology Department of the Mexican Social Security Institute",
            "original_filename": img_path.name,
        })


def make_splits(train_ratio=0.7, val_ratio=0.15):
    patient_to_data = defaultdict(list)
    patient_to_hospital = {}

    for entry in METADATA:
        if "source_path" in entry:
            patient_id = entry["source_path"]
        else:
            patient_id = f"{entry['hospital_id']}_{entry['original_filename'].split('.')[0]}"
        
        patient_to_data[patient_id].append(entry["index"])
        patient_to_hospital[patient_id] = entry["hospital_id"]

    patient_ids = list(patient_to_data.keys())
    hospital_labels = [patient_to_hospital[pid] for pid in patient_ids]

    train_pids, temp_pids = train_test_split(
        patient_ids, 
        train_size=train_ratio, 
        stratify=hospital_labels, 
        random_state=42
    )

    temp_labels = [patient_to_hospital[pid] for pid in temp_pids]
    relative_val_ratio = val_ratio / (1 - train_ratio)
    
    val_pids, test_pids = train_test_split(
        temp_pids, 
        train_size=relative_val_ratio, 
        stratify=temp_labels, 
        random_state=42
    )

    split_map = {
        'train': [idx for pid in train_pids for idx in patient_to_data[pid]],
        'val': [idx for pid in val_pids for idx in patient_to_data[pid]],
        'test': [idx for pid in test_pids for idx in patient_to_data[pid]]
    }

    path_map = {'train': TRAIN_DIR, 'val': VAL_DIR, 'test': TEST_DIR}

    for split_name, indices in split_map.items():
        target_path = path_map[split_name]
        print(f"Moving {len(indices)} images to {split_name}...")
        for ix in indices:
            img_name = f"_{ix}.png"
            mask_name = f"_{ix}.mask.png"
            if (DATASET_DIR / img_name).exists():
                shutil.move(DATASET_DIR / img_name, target_path / img_name)
            if (DATASET_DIR / mask_name).exists():
                shutil.move(DATASET_DIR / mask_name, target_path / mask_name)


def prepare_XCAD(xcad_dir):
    IMAGES_DIR = xcad_dir / "test" / "images"
    MASK_DIR = xcad_dir / "test" / "masks"

    image_files = list(IMAGES_DIR.glob("*.png"))
    
    all_valid_pairs = []
    for img_path in image_files:
        mask_path = MASK_DIR / img_path.name
        if mask_path.exists():
            all_valid_pairs.append((img_path, mask_path))

    random.seed(42)
    random.shuffle(all_valid_pairs)
    selected_pairs = all_valid_pairs[:200]

    for img_path, mask_path in tqdm(selected_pairs, desc="Processing XCAD Test"):
        ix = next_index()
        
        shutil.copy(img_path, DATASET_DIR / f"_{ix}.png")
        shutil.copy(mask_path, DATASET_DIR / f"_{ix}.mask.png")
        
        METADATA.append({
            "index": ix,
            "hospital_id": "XCAD",
            "original_filename": img_path.name,
        })


def prepare_ARCADE(split_map, input_root, sample_limit=200):
    all_available_samples = []

    for split_name, (image_subpath, json_subpath) in split_map.items():
        json_path = input_root / json_subpath
        if not json_path.exists():
            print(f"Warning: Annotation not found: {json_path}")
            continue
            
        coco = COCO(str(json_path))
        img_ids = coco.getImgIds()
        
        for img_id in img_ids:
            all_available_samples.append({
                'coco': coco,
                'img_id': img_id,
                'img_dir': input_root / image_subpath
            })

    random.seed(42)
    random.shuffle(all_available_samples)
    selected_samples = all_available_samples[:sample_limit]

    print(f"ARCADE: Found {len(all_available_samples)} total frames across splits. Sampling {len(selected_samples)}.")

    for sample in tqdm(selected_samples, desc="Processing ARCADE"):
        coco = sample['coco']
        img_id = sample['img_id']
        img_dir = sample['img_dir']
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        h, w = img_info['height'], img_info['width']
        
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            instance_mask = coco.annToMask(ann)
            binary_mask[instance_mask > 0] = 255

        src_img_path = img_dir / file_name
        if src_img_path.exists():
            ix = next_index()
            
            shutil.copy(src_img_path, DATASET_DIR / f"_{ix}.png")
            Image.fromarray(binary_mask).save(DATASET_DIR / f"_{ix}.mask.png")

            METADATA.append({
                "index": ix,
                "hospital_id": "ARCADE_MultiCenter", 
                "original_filename": file_name,
            })
        else:
            print(f"Warning: Image {file_name} not found in {img_dir}")

arcade_splits = {
    "training": (
        Path("dataset_phase_1") / "segmentation_dataset" / "seg_train" / "images",
        Path("dataset_phase_1") / "segmentation_dataset" / "seg_train" / "annotations" / "seg_train.json"
    ),
    "validation": (
        Path("dataset_phase_1") / "segmentation_dataset" / "seg_val" / "images",
        Path("dataset_phase_1") / "segmentation_dataset" / "seg_val" / "annotations" / "seg_val.json"
    ),
    "test": (
        Path("dataset_final_phase") / "test_case_segmentation" / "images",
        Path("dataset_final_phase") / "test_case_segmentation" / "annotations" / "instances_default.json"
    )
}

prepare_CARDIAG(DATA_DIR / "CARDIAG")
prepare_DCA1(DATA_DIR / "DCA1")
prepare_XCAD(DATA_DIR / "XCAD")
prepare_ARCADE(arcade_splits, DATA_DIR / "ARCADE")
make_splits()

with open(DATASET_DIR / "metadata.json", "w") as f:
        json.dump(METADATA, f, indent=4)

