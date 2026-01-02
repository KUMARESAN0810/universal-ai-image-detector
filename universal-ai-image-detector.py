import cv2
import numpy as np
import json
from scipy.fftpack import fft2, fftshift
from scipy.ndimage import gaussian_filter
from skimage.feature import greycomatrix
from skimage.restoration import denoise_nl_means
from tkinter import Tk, filedialog

# -------------------------------
# IMAGE LOADING (UPLOAD)
# -------------------------------
def upload_image():
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not path:
        raise ValueError("No image selected")
    return path

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Unable to read image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# -------------------------------
# 1. HIGH-FREQUENCY RESIDUAL
# -------------------------------
def high_frequency_energy(img):
    g = to_gray(img)
    blur = gaussian_filter(g, sigma=1.5)
    residual = g - blur
    return float(np.sum(residual ** 2))

# -------------------------------
# 2. FOURIER SPECTRAL TEST
# -------------------------------
def spectral_zmax(img):
    g = to_gray(img)
    F = fftshift(fft2(g))
    P = np.abs(F) ** 2
    radial = np.mean(P, axis=0)
    z = (radial - np.mean(radial)) / (np.std(radial) + 1e-8)
    return float(np.max(z))

# -------------------------------
# 3. TEXTURE CO-OCCURRENCE
# -------------------------------
def texture_entropy(img):
    g = to_gray(img)
    g = (g / 16).astype(np.uint8)
    glcm = greycomatrix(g, [1], [0], levels=16, symmetric=True, normed=True)
    entropy = -np.sum(glcm * np.log(glcm + 1e-8))
    return float(entropy * 100)

# -------------------------------
# 4. JPEG BLOCK CONSISTENCY
# -------------------------------
def jpeg_block_difference(img):
    g = to_gray(img)
    h, w = g.shape
    boundary, interior = [], []

    for i in range(8, h, 8):
        boundary.append(np.mean(np.abs(g[i] - g[i - 1])))

    for i in range(1, h):
        if i % 8 != 0:
            interior.append(np.mean(np.abs(g[i] - g[i - 1])))

    return float(abs(np.mean(boundary) - np.mean(interior)))

# -------------------------------
# 5. COLOR COVARIANCE
# -------------------------------
def color_covariance(img):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    cov = np.cov([R.flatten(), G.flatten(), B.flatten()])
    return float(np.linalg.det(cov))

# -------------------------------
# 6. NOISE ISOTROPY (DIFFUSION)
# -------------------------------
def noise_isotropy(img):
    g = to_gray(img)
    denoised = denoise_nl_means(g, fast_mode=True)
    noise = g - denoised
    F = np.abs(fftshift(fft2(noise)))
    return float(np.std(F) / (np.mean(F) + 1e-8))

# -------------------------------
# FINAL ANALYSIS
# -------------------------------
def analyze_image(img):
    metrics = {
        "HighFrequencyEnergy": high_frequency_energy(img),
        "SpectralZmax": spectral_zmax(img),
        "TextureEntropy": texture_entropy(img),
        "JPEGBlockDifference": jpeg_block_difference(img),
        "ColorCovarianceDet": color_covariance(img),
        "NoiseIsotropy": noise_isotropy(img)
    }

    evidence = []

    if metrics["NoiseIsotropy"] < 0.9:
        evidence.append("Isotropic noise field (diffusion signature)")

    if metrics["SpectralZmax"] > 4.5:
        evidence.append("Periodic frequency artifacts (GAN signature)")

    if metrics["TextureEntropy"] < 200:
        evidence.append("Over-regular texture statistics")

    if metrics["JPEGBlockDifference"] > 0.01:
        evidence.append("Inconsistent JPEG compression")

    verdict = "AI-GENERATED" if len(evidence) >= 2 else "REAL"

    return {
        "Verdict": verdict,
        "Evidence": evidence,
        "Metrics": metrics
    }

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    try:
        path = upload_image()
        image = load_image(path)
        report = analyze_image(image)

        print("\n===== FORENSIC ANALYSIS REPORT =====\n")
        print(json.dumps(report, indent=4))

    except Exception as e:
        print("Error:", e)
