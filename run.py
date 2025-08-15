import SimpleITK as sitk
import numpy as np
import logging
from pathlib import Path
from evalutils.io import FileLoader, ImageLoader, SimpleITKLoader
from pandas import DataFrame
from evalutils.exceptions import FileLoaderError
import json
from konfai.utils.dataset import Dataset
import os
from typing import Optional


logger = logging.getLogger(__name__)

TASK_TYPE = "mri"
INPUT_FOLDER = "/input"
OUTPUT_FOLDER = "/output"

DEFAULT_IMAGE_PATH = Path("{}/images/{}".format(INPUT_FOLDER, TASK_TYPE))
DEFAULT_REGION_PATH = Path("{}/region.json".format(INPUT_FOLDER))
DEFAULT_MASK_PATH = Path("{}/images/body".format(INPUT_FOLDER))
DEFAULT_OUTPUT_PATH = Path("{}/images/synthetic-ct".format(OUTPUT_FOLDER))
DEFAULT_OUTPUT_FILE = Path("{}/results.json".format(OUTPUT_FOLDER))
    

def _preprocess(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    data = sitk.GetArrayFromImage(image).astype(np.float32)
    data_mask = sitk.GetArrayFromImage(mask)
    data = (data - np.mean(data[data_mask == 1]))/np.std(data[data_mask == 1])
    standardized_image = sitk.GetImageFromArray(data)
    standardized_image.CopyInformation(image)
    standardized_image_masked = sitk.Mask(standardized_image, mask, -2)
    return sitk.Cast(standardized_image_masked, sitk.sitkFloat32)

def _load_cases(folder: Path, file_loader: ImageLoader) -> DataFrame:
    cases = []

    for fp in sorted(folder.glob("*")):
        try:
            new_cases = file_loader.load(fname=fp)
        except FileLoaderError:
            logger.warning(f"Could not load {fp.name} using {file_loader}.")
        else:
            cases.extend(new_cases)

    if len(cases) == 0:
        raise FileLoaderError(
            f"Could not load any files in {folder} with " f"{file_loader}."
        )

    return cases

def _load_input_image(image, file_loader) -> tuple[sitk.Image, Path]:
    input_image_file_path = image["path"]
    input_image_file_loader = file_loader

    if not isinstance(input_image_file_loader, ImageLoader):
        raise RuntimeError("The used FileLoader was not of subclass ImageLoader")

    # Load the image
    input_image = input_image_file_loader.load_image(input_image_file_path)

    # Check that it is the expected image
    if input_image_file_loader.hash_image(input_image) != image["hash"]:
        raise RuntimeError("Image hashes do not match")
    return input_image, input_image_file_path

class SynthradAlgorithm():

    def __init__(self, input_path: Path = DEFAULT_IMAGE_PATH,
        mask_path: Path = DEFAULT_MASK_PATH,
        region_path: Path = DEFAULT_REGION_PATH,
        output_path: Path = DEFAULT_OUTPUT_PATH,
        output_file: Path = DEFAULT_OUTPUT_FILE,
        validators: Optional[dict[str, callable]] = None,
        file_loader: FileLoader = SimpleITKLoader()):

        self.input_path = input_path
        self.mask_path = mask_path
        self.region_path = region_path
        self.output_path = output_path
        self.output_file = output_file
        self.file_loader = file_loader

        self.images_file_paths: dict[str, dict[str, Path]] = {}

    def prepareData(self):
        images = _load_cases(folder=self.input_path, file_loader=self.file_loader)
        masks = _load_cases(folder=self.mask_path, file_loader=self.file_loader)
        
        dataset = Dataset("./Dataset/", "mha")
        for image_path, mask_pah in zip(images, masks):
            image, image_name = _load_input_image(image_path, file_loader=self.file_loader)
            region = image_name.name[1:3]
            self.images_file_paths[image_name.name] = {}
            self.images_file_paths[image_name.name]["image"] = image_name
            mask, self.images_file_paths[image_name.name]["mask"] = _load_input_image(mask_pah, file_loader=self.file_loader)
            mask = sitk.Cast(mask, sitk.sitkUInt8)

            dataset.write(f"{region}/MR", image_name.name.split(".")[0], _preprocess(image, mask))
            dataset.write(f"{region}/MASK", image_name.name.split(".")[0], mask)

    def save(self):
        _case_results = []
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True, exist_ok=True)
        
        for name in self.images_file_paths:
            out_path = self.output_path / name
            sitk.WriteImage(sitk.ReadImage("./Predictions/Out/Dataset/{}/sCT.mha".format(name.split(".")[0])), out_path)
            _case_results.append({
                "outputs": [dict(type="metaio_image", filename=str(out_path))],
                "inputs": [
                    dict(type="metaio_image", filename=str(fn))
                    for fn in self.images_file_paths[name].values()
                ] + [dict(type="String", filename=str(self.region_path))],
                "error_messages": [],
            })
        
        with open(str(self.output_file), "w") as f:
            json.dump(_case_results, f)

if __name__ == "__main__":
    algorithm = SynthradAlgorithm()
    algorithm.prepareData()
    os.system("konfai PREDICTION -y --gpu 0 --config Checkpoints/AB-TH/Prediction.yml --MODEL Checkpoints/AB-TH/CV_0.pt:Checkpoints/AB-TH/CV_1.pt:Checkpoints/AB-TH/CV_2.pt:Checkpoints/AB-TH/CV_3.pt:Checkpoints/AB-TH/CV_4.pt")
    os.system("konfai PREDICTION -y --gpu 0 --config Checkpoints/HN/Prediction.yml --MODEL Checkpoints/HN/CV_0.pt:Checkpoints/HN/CV_1.pt:Checkpoints/HN/CV_2.pt:Checkpoints/HN/CV_3.pt:Checkpoints/HN/CV_4.pt")
    algorithm.save()
    