# %%
!pip install scikit-image matplotlib opencv-python

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# %%
# List of image paths
image_paths = [
    "/content/butterfly.jpg",
    "/content/coins.jpg",
    "/content/desk.jpg"

]

# Dictionaries to store images
images_rgb = {}    # store RGB images
images_gray = {}   # store grayscale images

# Loop through each image
for path in image_paths:
    # Read image
    image = cv2.imread(path)

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images_rgb[path] = image_rgb  # store in dict

    # Convert to Grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images_gray[path] = image_gray  # store in dict

    # Display RGB
    plt.figure(figsize=(6,6))
    plt.imshow(image_rgb)
    plt.title(f"{path.split('/')[-1]} (RGB)")
    plt.axis("off")
    plt.show()

    # Display Grayscale
    plt.figure(figsize=(6,6))
    plt.imshow(image_gray, cmap='gray')
    plt.title(f"{path.split('/')[-1]} (Grayscale)")
    plt.axis("off")
    plt.show()


# %%
# Dictionaries to store noisy images
gaussian_noisy_images = {}
sp_noisy_images = {}

# Loop through each RGB image
for path, image_rgb in images_rgb.items():
    # Apply Gaussian noise
    gaussian_noisy = random_noise(image_rgb, mode='gaussian', var=0.02)
    gaussian_noisy_uint8 = (255 * gaussian_noisy).astype(np.uint8)
    gaussian_noisy_images[path] = gaussian_noisy_uint8

    # Apply Salt & Pepper noise
    sp_noisy = random_noise(image_rgb, mode='s&p', amount=0.2)
    sp_noisy_uint8 = (255 * sp_noisy).astype(np.uint8)
    sp_noisy_images[path] = sp_noisy_uint8

    # Display the noisy images
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(gaussian_noisy_uint8)
    plt.title(f"{path.split('/')[-1]} - Gaussian Noise")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(sp_noisy_uint8)
    plt.title(f"{path.split('/')[-1]} - Salt & Pepper Noise")
    plt.axis("off")

    plt.show()


# %%
from skimage.exposure import rescale_intensity
# Parameters
scale = 0.5

# Dictionaries to store processed images
gauss_gray_dict = {}
gauss_hist_eq_dict = {}
gauss_contrast_dict = {}

sp_gray_dict = {}
sp_hist_eq_dict = {}
sp_contrast_dict = {}

# ---- Process Gaussian noisy images ----
for path, gauss_img in gaussian_noisy_images.items():
    # Convert to grayscale
    gray_gauss = cv2.cvtColor(gauss_img, cv2.COLOR_RGB2GRAY)
    # Resize
    resize_dim = (int(gray_gauss.shape[1]*scale), int(gray_gauss.shape[0]*scale))
    gray_gauss = cv2.resize(gray_gauss, resize_dim, interpolation=cv2.INTER_AREA)
    gauss_gray_dict[path] = gray_gauss

    # Histogram Equalization
    gauss_hist_eq = cv2.equalizeHist(gray_gauss)
    gauss_hist_eq_dict[path] = gauss_hist_eq

    # Contrast Stretching
    p2, p98 = np.percentile(gray_gauss, (2, 98))
    gauss_contrast = rescale_intensity(gray_gauss, in_range=(p2, p98))
    gauss_contrast_dict[path] = gauss_contrast

    # Display
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(gray_gauss, cmap='gray')
    plt.title(f"{path.split('/')[-1]} - Gaussian Grayscale")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(gauss_hist_eq, cmap='gray')
    plt.title(f"{path.split('/')[-1]} - Gaussian Hist Eq")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(gauss_contrast, cmap='gray')
    plt.title(f"{path.split('/')[-1]} - Gaussian Contrast Stretch")
    plt.axis("off")

    plt.show()

# ---- Process Salt & Pepper noisy images ----
for path, sp_img in sp_noisy_images.items():
    # Convert to grayscale
    gray_sp = cv2.cvtColor(sp_img, cv2.COLOR_RGB2GRAY)
    # Resize
    resize_dim = (int(gray_sp.shape[1]*scale), int(gray_sp.shape[0]*scale))
    gray_sp = cv2.resize(gray_sp, resize_dim, interpolation=cv2.INTER_AREA)
    sp_gray_dict[path] = gray_sp

    # Histogram Equalization
    sp_hist_eq = cv2.equalizeHist(gray_sp)
    sp_hist_eq_dict[path] = sp_hist_eq

    # Contrast Stretching
    p2, p98 = np.percentile(gray_sp, (2, 98))
    sp_contrast = rescale_intensity(gray_sp, in_range=(p2, p98))
    sp_contrast_dict[path] = sp_contrast

    # Display
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(gray_sp, cmap='gray')
    plt.title(f"{path.split('/')[-1]} - S&P Grayscale")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(sp_hist_eq, cmap='gray')
    plt.title(f"{path.split('/')[-1]} - S&P Hist Eq")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(sp_contrast, cmap='gray')
    plt.title(f"{path.split('/')[-1]} - S&P Contrast Stretch")
    plt.axis("off")

    plt.show()


# %%
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt

# Parameters
scale = 0.5

# Dictionaries to store filtered results
filtered_results_dict = {}

# Loop through all images
for path in images_gray.keys():
    # Original grayscale resized
    image_gray_resized = cv2.resize(images_gray[path],
                                    (int(images_gray[path].shape[1]*scale), int(images_gray[path].shape[0]*scale)),
                                    interpolation=cv2.INTER_AREA)

    # Get processed images for this path
    gauss_hist = gauss_hist_eq_dict[path]
    gauss_contrast_img = gauss_contrast_dict[path]
    sp_hist = sp_hist_eq_dict[path]
    sp_contrast_img = sp_contrast_dict[path]

    # Dictionary to store filtered images for this particular image
    results = {}

    # --- Gaussian Noise + HistEq ---
    results["GaussianHist_GaussianFilter"] = cv2.GaussianBlur(gauss_hist, (5,5), 0)
    results["GaussianHist_MedianFilter"] = cv2.medianBlur(gauss_hist, 5)

    # --- Gaussian Noise + Contrast Stretching ---
    results["GaussianCS_GaussianFilter"] = cv2.GaussianBlur(gauss_contrast_img, (5,5), 0)
    results["GaussianCS_MedianFilter"] = cv2.medianBlur(gauss_contrast_img, 5)

    # --- Salt & Pepper Noise + HistEq ---
    results["SPHist_GaussianFilter"] = cv2.GaussianBlur(sp_hist, (5,5), 0)
    results["SPHist_MedianFilter"] = cv2.medianBlur(sp_hist, 5)

    # --- Salt & Pepper Noise + Contrast Stretching ---
    results["SPCS_GaussianFilter"] = cv2.GaussianBlur(sp_contrast_img, (5,5), 0)
    results["SPCS_MedianFilter"] = cv2.medianBlur(sp_contrast_img, 5)

    # Store filtered results
    filtered_results_dict[path] = results

    # Display filtered images with PSNR and SSIM
    for title, img in results.items():
        psnr_val = cv2.PSNR(image_gray_resized, img)
        ssim_val = ssim(image_gray_resized, img, data_range=255)

        plt.figure(figsize=(5,5))
        plt.imshow(img, cmap='gray')
        plt.title(f"{path.split('/')[-1]}\n{title}\nPSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
        plt.axis("off")
        plt.show()


# %%
# ---------- Imports ----------
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color, morphology
from skimage.filters import threshold_local
from skimage.measure import regionprops
from skimage.morphology import closing, square, remove_small_objects
from scipy import ndimage as ndi

# ---------- Helper functions ----------
def to_display_rgb(img):
    """Ensure an image is uint8 RGB for plotting with plt.imshow."""
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = (255 * (img - img.min()) / (img.max() - img.min() + 1e-9)).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def compute_color_histogram(rgb_image, bbox, bins=16):
    """Compute normalized color histogram for a bounding box region.
       rgb_image: uint8 RGB image
       bbox: (min_row, min_col, max_row, max_col)
    """
    minr, minc, maxr, maxc = bbox
    patch = rgb_image[minr:maxr, minc:maxc]
    if patch.size == 0:
        return np.zeros((3*bins,))
    hist = []
    for ch in range(3):
        h = np.histogram(patch[:,:,ch], bins=bins, range=(0,255))[0].astype(float)
        if h.sum() > 0:
            h = h / h.sum()
        hist.append(h)
    return np.concatenate(hist)

def dice_coefficient(mask1, mask2):
    """Compute Dice coefficient between two boolean masks."""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    inter = np.logical_and(mask1, mask2).sum()
    s = mask1.sum() + mask2.sum()
    if s == 0:
        return 1.0
    return 2.0 * inter / s

def annotate_regions_on_image(rgb_img, regions, color_box=(0,255,0), color_centroid=(255,0,0)):
    """Draw bounding boxes and centroids on a copy of rgb_img (uint8 RGB)."""
    out = rgb_img.copy()
    for r in regions:
        minr, minc, maxr, maxc = r.bbox
        # draw rectangle
        cv2.rectangle(out, (minc, minr), (maxc, maxr), color_box, 2)
        # draw centroid
        cy, cx = r.centroid
        cv2.circle(out, (int(cx), int(cy)), 3, color_centroid, -1)
        # put area text
        cv2.putText(out, f"A:{int(r.area)}", (minc, max(15, minr-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_box, 1)
    return out


# %%
def adaptive_threshold_segmentation(gray, block_size=51, offset=10, min_size=100):
    """
    Adaptive threshold segmentation using skimage threshold_local and morphological ops.
    Returns a boolean mask (True = foreground).
    """
    # Ensure gray in uint8
    if gray.dtype != np.uint8:
        gray_u8 = (255 * (gray - gray.min()) / (gray.max() - gray.min() + 1e-9)).astype(np.uint8)
    else:
        gray_u8 = gray
    # local threshold
    t = threshold_local(gray_u8, block_size, offset=offset, method='mean')
    mask = gray_u8 > t
    # morphological cleanup: close and remove small objects
    mask = closing(mask, square(3))
    mask = remove_small_objects(mask, min_size=min_size)
    mask = ndi.binary_fill_holes(mask)
    return mask

def edge_based_segmentation(rgb_or_gray, canny_sigma=1.0, closing_size=5, min_size=100):
    """
    Edge-based segmentation:
     - convert to grayscale if color passed
     - apply Canny -> dilate/close -> fill holes -> remove small objects
    returns boolean mask
    """
    if rgb_or_gray.ndim == 3:
        gray = cv2.cvtColor(rgb_or_gray, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb_or_gray
    # normalize for Canny
    if gray.dtype != np.uint8:
        gray_u8 = (255 * (gray - gray.min()) / (gray.max() - gray.min() + 1e-9)).astype(np.uint8)
    else:
        gray_u8 = gray
    edges = cv2.Canny(gray_u8, threshold1=50, threshold2=150, apertureSize=3)
    # Dilate / close to form borders
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # fill holes by finding connected components on inverted edges
    closed_bool = closed.astype(bool)
    # invert to get regions
    mask = ndi.binary_fill_holes(closed_bool)
    # we likely want regions where gradient exists -> refine by using thresholded intensity
    # Use Otsu or simple threshold to define candidate interiors
    _, thr = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask2 = thr.astype(bool)
    # combine structural mask and intensity mask
    combined = np.logical_or(mask, mask2)
    combined = closing(combined, square(3))
    combined = remove_small_objects(combined, min_size=min_size)
    combined = ndi.binary_fill_holes(combined)
    return combined


# %%
def extract_features_and_annotate(rgb_image, mask, min_area_filter=50):
    """
    Input:
     - rgb_image: uint8 RGB image
     - mask: boolean mask (True = object)
    Returns:
     - regions (list of skimage regionprops)
     - annotated_rgb (uint8) with boxes + centroids
     - features_list: list of dicts {area, centroid, bbox, color_hist}
    """
    labeled = measure.label(mask)
    props = regionprops(labeled)
    # filter small regions
    props = [p for p in props if p.area >= min_area_filter]
    features = []
    for p in props:
        bbox = p.bbox  # (minr, minc, maxr, maxc)
        hist = compute_color_histogram(rgb_image, bbox, bins=16)
        features.append({
            'area': p.area,
            'centroid': p.centroid,
            'bbox': bbox,
            'eccentricity': p.eccentricity,
            'extent': p.extent,
            'color_hist': hist
        })
    annotated = annotate_regions_on_image(rgb_image, props)
    return props, annotated, features


# %%
# ---------- Collate and print a pipeline-level summary ----------
print("\n\n=== PIPELINE SUMMARY ===")
for path, out in segmentation_outputs.items():
    name = path.split('/')[-1]
    print(f"\n{name}:")
    print(f"  Adaptive -> objects: {out['num_adapt']}, mean area: {np.mean(out['areas_adapt']):.1f}")
    print(f"  Edge-based -> objects: {out['num_edge']}, mean area: {np.mean(out['areas_edge']):.1f}")
    print(f"  Dice (adapt vs edge): {out['dice']:.3f}")

# ---------- Qualitative reflection template (printable) ----------
reflection = """
Reflection / What worked:
 - Adaptive thresholding (local) handles uneven illumination better because it uses neighborhood statistics.
 - Edge-based (Canny + closing) highlights fine texture and boundaries; good when object edges are strong.

Limitations / What to improve:
 - Parameter sensitivity: block_size/offset (adaptive) and Canny thresholds/closing_size strongly change outputs.
 - Over-segmentation or merged objects can happen (use watershed / marker-based segmentation to separate touching objects).
 - Color-based segmentation (e.g., KMeans in Lab color space or color thresholding) can help when objects are distinguished by color.
 - Consider morphological opening/closing tuning and connected component area filtering thresholds adapted per-image.
 - For production: use a learning-based segmentation (U-Net / DeepLab) when shapes are complex or variable.

Possible next steps:
 - Use watershed with markers to split touching objects.
 - Use GrabCut for foreground extraction when a rough bounding box is known.
 - Use color-space clustering (k-means) in Lab space, followed by morphological cleanup for color-separable objects.
 - If ground truth masks available, compute IoU/mean average precision per object.
"""
print(reflection)


# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

for i, (path, gray_image) in enumerate(images_gray.items(), start=1):
    # --- Step 1: Edge Detection ---
    edges = cv2.Canny(gray_image, 100, 200)   # You can tune thresholds (100, 200)

    # --- Step 2: Dilate + Close small gaps ---
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- Step 3: Find contours from edges ---
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Step 4: Copy RGB image for drawing ---
    output = images_rgb[path].copy()

    # --- Step 5: Draw bounding boxes and centroids ---
    for c in contours:
        area = cv2.contourArea(c)
        if area < 150:   # ignore small noise
            continue

        # Bounding box (green)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Centroid (red with white border)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(output, (cx, cy), 10, (255, 255, 255), 3)  # white border
            cv2.circle(output, (cx, cy), 6, (255, 0, 0), -1)      # inner red
            cv2.putText(output, f"({cx},{cy})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # --- Step 6: Display results ---
    plt.figure(figsize=(15,6))

    plt.subplot(1,2,1)
    plt.imshow(edges, cmap='gray')
    plt.title(f"Fig. {i+5}. Edge-based Mask (Canny)")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(output)
    plt.title(f"Fig. {i+6}. Annotated Image (Bounding Boxes & Centroids)")
    plt.axis("off")

    plt.show()



