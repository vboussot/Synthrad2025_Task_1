import os
import SimpleITK as sitk
from konfai.utils.dataset import Dataset, Attribute
from sklearn.model_selection import KFold

import numpy as np
import random
from scipy import ndimage
from totalsegmentator.python_api import totalsegmentator

def clipAndNormalizeAndMask(image: sitk.Image, mask: sitk.Image, v_min: float, v_max: float) -> sitk.Image:
    data = sitk.GetArrayFromImage(image).astype(np.float32)
    data = np.clip(data, v_min, v_max)

    data = 2.0 * (data - v_min) / (v_max - v_min) - 1.0

    normalized_image = sitk.GetImageFromArray(data)
    normalized_image.CopyInformation(image)
    normalized_image_masked = sitk.Mask(normalized_image, mask, -1)
    return normalized_image_masked
    
def clipAndNormalizeAndMask_CT(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    return clipAndNormalizeAndMask(image, mask, -1024, 3071)

def clipAndStandardizeAndMask_MR(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    data = sitk.GetArrayFromImage(image).astype(np.float32)
    data_mask = sitk.GetArrayFromImage(mask)
    data = (data - np.mean(data[data_mask == 1]))/np.std(data[data_mask == 1])
    standardized_image = sitk.GetImageFromArray(data)
    standardized_image.CopyInformation(image)
    standardized_image_masked = sitk.Mask(standardized_image, mask, -2)
    return sitk.Cast(standardized_image_masked, sitk.sitkFloat32)

def getMask_MR(image: sitk.Image, mask: sitk.Image):
    sitk.WriteImage(image, "./tmp/image.nii.gz")

    seg_img = totalsegmentator("./tmp/image.nii.gz", "./tmp", task="body_mr", skip_saving=False)
    data = np.transpose(seg_img.get_fdata(), (2, 1, 0))
    data = np.where(data > 0, 1, 0)
    
    mask_data = sitk.GetArrayFromImage(mask)

    result = sitk.Cast(sitk.GetImageFromArray(data | mask_data), sitk.sitkUInt8)
    result.CopyInformation(image)
    return result
 
def prepare_train_task_1(dataset: Dataset, region : str):
    patients = [name for name in sorted(os.listdir("./raw_data/Train/Task_1/{}/".format(region))) if name != "overviews"]

    for patient in patients:
        CT = sitk.ReadImage("./raw_data/Train/Task_1/{}/{}/ct.mha".format(region, patient))
        MR = sitk.ReadImage("./raw_data/Train/Task_1/{}/{}/mr.mha".format(region, patient))
        MASK = sitk.Cast(sitk.ReadImage("./raw_data/Train/Task_1/{}/{}/mask.mha".format(region, patient)), sitk.sitkUInt8)
        impact = sitk.ReadTransform("./raw_data/Train/Task_1/{}/{}/IMPACT.itk.txt".format(region, patient))
        elastix = sitk.ReadTransform("./raw_data/Train/Task_1/{}/{}/elastic.itk.txt".format(region, patient))
        
        MASK_corrected = getMask_MR(MR, MASK)

        CT_elastix = sitk.Resample(CT, elastix)
        MR_IMPACT = sitk.Resample(MR, impact)

        CT = clipAndNormalizeAndMask_CT(CT, MASK_corrected)
        CT_elastix = clipAndNormalizeAndMask_CT(CT_elastix, MASK_corrected)
        MR = clipAndStandardizeAndMask_MR(MR, MASK_corrected)
        MR_IMPACT = clipAndStandardizeAndMask_MR(MR_IMPACT, MASK_corrected)

        dataset.write("{}/CT".format(region), patient, CT)
        dataset.write("{}/CT_ELASTIX".format(region), patient, CT_elastix)
        dataset.write("{}/MR".format(region), patient, MR)
        dataset.write("{}/MR_IMPACT".format(region), patient, MR_IMPACT)
        dataset.write("{}/MASK".format(region), patient, MASK)

def prepare_validation_task_1(dataset: Dataset, region : str):
    patients = [name for name in sorted(os.listdir("./raw_data/Validation/Task_1/{}/".format(region))) if name != "overviews"]

    for patient in patients:
        MR = sitk.ReadImage("./raw_data/Validation/Task_1/{}/{}/mr.mha".format(region, patient))
        MASK = sitk.Cast(sitk.ReadImage("./raw_data/Validation/Task_1/{}/{}/mask.mha".format(region, patient)), sitk.sitkUInt8)
        
        MASK_corrected = getMask_MR(MR, MASK)

        MR = clipAndStandardizeAndMask_MR(MR, MASK_corrected)
        
        dataset.write("{}/MR".format(region), patient, MR)
        dataset.write("{}/MASK".format(region), patient, MASK)

def validation_task_1(dataset: Dataset):
    n_folds = 5
    regions_centers = {"AB": ["A", "B", "C"], "HN" : ["A", "C", "D"], "TH" : ["A", "B"]}

    indices_regions_centers = []
    for region, centers in regions_centers.items():
        names = dataset.get_names(f"{region}/CT")
        for center in centers:
            indices_regions_centers.append(list(np.random.permutation([n for n in names if n.startswith(f"1{region}{center}")])))
    
    for i in range(n_folds):
        if os.path.exists(f"./Validation/Task_1/CrossValidation_{i}.txt")
            os.remove(f"./Validation/Task_1/CrossValidation_{i}.txt")
    
    for indices in indices_regions_centers:
        for i, a in enumerate(np.array_split([indice for indice in indices], n_folds)):
            with open(f"./Validation/Task_1/CrossValidation_{i}.txt", "a") as f:    
                for l in a:
                    f.write(f"{l}\n")

if __name__ == "__main__":
    dataset = Dataset("./Dataset/Train/Task_1", "mha")
    prepare_train_task_1(dataset, "AB")
    prepare_train_task_1(dataset, "HN")
    prepare_train_task_1(dataset, "TH")

    dataset = Dataset("./Dataset/Validation/Task_1", "mha")
    prepare_validation_task_1(dataset, "AB")
    prepare_validation_task_1(dataset, "HN")
    prepare_validation_task_1(dataset, "TH")