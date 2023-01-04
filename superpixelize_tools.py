import numpy as np
from scipy.stats import iqr
from skimage.segmentation import slic

def summarize_segment(segment, segment_gt):

    segment_means = np.mean(segment, axis=0)
    segment_medians = np.median(segment, axis=0)
    segment_iqrs = iqr(segment, axis=0)
    segment_maxs = np.max(segment, axis = 0)
    segment_mins = np.min(segment, axis = 0)
    segment_stds = np.std(segment, axis = 0)

    segment_gt_avg = np.mean(segment_gt)

    feature_names = [
    'R_med', 'G_med', 'B_med', 'Nir_med',
    'R_avg', 'G_avg', 'B_avg', 'Nir_avg',
    'R_iqr', 'G_iqr', 'B_iqr', 'Nir_iqr',
    'R_max', 'G_max', 'B_max', 'Nir_max',
    'R_min', 'G_min', 'B_min', 'Nir_min',
    'R_std', 'G_std', 'B_std', 'Nir_std',
    'gt_avg'
    ]
    segment_features = np.array([segment_medians, segment_means, segment_iqrs, segment_maxs, segment_mins, segment_stds]).flatten()
    segment_features_gt = np.append(segment_features, segment_gt_avg)

    return(segment_features_gt, feature_names)

def superpixelize_patch(image_nir, image_gt, segments):
    patch_superpixels = np.empty((0,25))

    for segment_id in range(1,len(np.unique(segments))+1):
        segment_mask = (segments == segment_id)#Create segment mask

        segment_unfiltered = image_nir[segment_mask]#Create segment from the patch image
        segment_gt_unfiltered = image_gt[segment_mask]#Create gt segment from patch gt

        segment = segment_unfiltered[np.sum(segment_unfiltered, axis=1) > 0]#Filter out the margins in segment
        segment_gt = segment_gt_unfiltered[np.sum(segment_unfiltered, axis=1) > 0]#Filter out the margins in segment gt

        if(segment.shape[0]==0):#Skip for segments consisting purely of margins
            continue
        
        superpixel, feature_names= summarize_segment(segment=segment, segment_gt=segment_gt)#Create superpixel from the current segment
        patch_superpixels = np.vstack([patch_superpixels, superpixel])#Add superpixel to superpixels array

    return(patch_superpixels)

def superpixelize_scene(scene_data, numSegments=200, sigma=5):
    superpixelized_scene = np.empty((0,25))
    for patch_id in range(scene_data.n_patches): 
        image = scene_data.open_as_array(idx=patch_id)
        image_nir = scene_data.open_as_array(idx=patch_id, include_nir=True)
        image_gt = scene_data.open_mask(idx=patch_id)

        segments = slic(image, n_segments = numSegments, sigma = sigma, convert2lab=True, channel_axis=2)
        superpixelized_scene = np.vstack([superpixelized_scene ,superpixelize_patch(image_nir=image_nir, image_gt=image_gt, segments=segments)])
    return(superpixelized_scene)