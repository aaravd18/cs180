import os
import time
import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from PIL import Image

def four_loop_convolution(image, kernel):
    height, width = image.shape
    kh, kw = kernel.shape

    # flip kernel
    kernel = np.flip(kernel)

    # pad so output is the same size as the input
    ph = kh // 2
    pw = kw // 2
    pad_img = np.pad(image, ((ph, ph), (pw, pw)))

    output = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            for ki in range(kh):
                for kj in range(kw):
                    output[i, j] += pad_img[i + ki, j + kj] * kernel[ki, kj]

    return output


def two_loop_convolution(image, kernel):
    height, width = image.shape
    kh, kw = kernel.shape

    kernel = np.flip(kernel)

    ph = kh // 2
    pw = kw // 2
    pad_img = np.pad(image, ((ph, ph), (pw, pw)))

    output = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            patch = pad_img[i:i + kh, j:j + kw]
            output[i, j] = np.sum(patch * kernel)

    return output

def test_conv_implementation():
    image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)

    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ], dtype=float)

    out_four = four_loop_convolution(image, kernel)
    out_two = two_loop_convolution(image, kernel)
    out_scipy = convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)

    print("Four loop matches scipy:", np.allclose(out_four, out_scipy))
    print("Two loop matches scipy:", np.allclose(out_two, out_scipy))


def compare_runtimes(image_path="aarav.jpg"):
    image = np.array(Image.open(image_path).convert("L"), dtype=float) / 255.0
    kernel = np.ones((5, 5)) / 25

    start = time.time()
    four_loop_convolution(image, kernel)
    four_loop_time = time.time() - start

    start = time.time()
    two_loop_convolution(image, kernel)
    two_loop_time = time.time() - start

    start = time.time()
    convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)
    scipy_time = time.time() - start

    print(f"Four loop convolution: {four_loop_time:.6f} sec")
    print(f"Two loop convolution: {two_loop_time:.6f} sec")
    print(f"Scipy convolution: {scipy_time:.6f} sec")


def part_1_2_partials(image_path="cameraman.png"):
    image = np.array(Image.open(image_path).convert("L"), dtype=float) / 255.0
    Dx = np.array([[1, -1]])
    Dy = np.array([[1], [-1]])

    dx = convolve2d(image, Dx, mode="same", boundary="fill", fillvalue=0)
    dy = convolve2d(image, Dy, mode="same", boundary="fill", fillvalue=0)

    plt.imsave("cameraman_dx.png", dx, cmap="gray")
    plt.imsave("cameraman_dy.png", dy, cmap="gray")


def part_1_2_magnitude(threshold, image_path="cameraman.png"):
    image = np.array(Image.open(image_path).convert("L"), dtype=float) / 255.0
    Dx = np.array([[1, -1]])
    Dy = np.array([[1], [-1]])

    dx = convolve2d(image, Dx, mode="same", boundary="fill", fillvalue=0)
    dy = convolve2d(image, Dy, mode="same", boundary="fill", fillvalue=0)
    magnitude = np.sqrt(dx**2 + dy**2)
    edges = magnitude > threshold

    print(f"threshold={threshold}, magnitude max={magnitude.max():.4f}, edge fraction={edges.mean():.4f}")

    plt.imsave("cameraman_magnitude.png", magnitude, cmap="gray")
    plt.imsave(f"cameraman_edges_{threshold}.png", edges, cmap="gray")


def part_1_2_panel(thresholds=(0.1, 0.2, 0.3), image_path="cameraman.png"):
    image = np.array(Image.open(image_path).convert("L"), dtype=float) / 255.0
    Dx = np.array([[1, -1]])
    Dy = np.array([[1], [-1]])

    dx = convolve2d(image, Dx, mode="same", boundary="fill", fillvalue=0)
    dy = convolve2d(image, Dy, mode="same", boundary="fill", fillvalue=0)
    magnitude = np.sqrt(dx**2 + dy**2)

    fig, axes = plt.subplots(1, 1 + len(thresholds), figsize=(4 * (1 + len(thresholds)), 4))
    axes[0].imshow(magnitude, cmap="gray")
    axes[0].set_title("Gradient magnitude")
    for ax, t in zip(axes[1:], thresholds):
        ax.imshow(magnitude > t, cmap="gray")
        ax.set_title(f"threshold = {t}")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("cameraman_panel.png", dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def gaussian_kernel_2d(ksize=9, sigma=1.5):
    g1 = cv2.getGaussianKernel(ksize, sigma)
    return g1 @ g1.T


def part_1_3_blurred_gradient(threshold=0.08, image_path="cameraman.png"):
    image = np.array(Image.open(image_path).convert("L"), dtype=float) / 255.0
    G = gaussian_kernel_2d()
    blurred = convolve2d(image, G, mode="same", boundary="fill", fillvalue=0)

    dx = convolve2d(blurred, np.array([[1, -1]]), mode="same", boundary="fill", fillvalue=0)
    dy = convolve2d(blurred, np.array([[1], [-1]]), mode="same", boundary="fill", fillvalue=0)
    magnitude = np.sqrt(dx**2 + dy**2)

    plt.imsave("cameraman_blur_dx.png", dx, cmap="gray")
    plt.imsave("cameraman_blur_dy.png", dy, cmap="gray")
    plt.imsave("cameraman_blur_magnitude.png", magnitude, cmap="gray")
    plt.imsave("cameraman_blur_edges.png", magnitude > threshold, cmap="gray")


def part_1_3_dog(threshold=0.08, image_path="cameraman.png"):
    image = np.array(Image.open(image_path).convert("L"), dtype=float) / 255.0
    G = gaussian_kernel_2d()

    dog_x = convolve2d(G, np.array([[1, -1]]), mode="full")
    dog_y = convolve2d(G, np.array([[1], [-1]]), mode="full")

    plt.imsave("cameraman_dog_x.png", dog_x, cmap="gray")
    plt.imsave("cameraman_dog_y.png", dog_y, cmap="gray")

    dx = convolve2d(image, dog_x, mode="same", boundary="fill", fillvalue=0)
    dy = convolve2d(image, dog_y, mode="same", boundary="fill", fillvalue=0)
    magnitude = np.sqrt(dx**2 + dy**2)

    plt.imsave("cameraman_dog_dx.png", dx, cmap="gray")
    plt.imsave("cameraman_dog_dy.png", dy, cmap="gray")
    plt.imsave("cameraman_dog_magnitude.png", magnitude, cmap="gray")
    plt.imsave("cameraman_dog_edges.png", magnitude > threshold, cmap="gray")


def part_2_1_sharpen(image_path="taj.jpg", alphas=(0.5, 1.0, 2.0)):
    prefix = os.path.splitext(image_path)[0]
    image = np.array(Image.open(image_path).convert("RGB"), dtype=float) / 255.0
    G = gaussian_kernel_2d()
    delta = np.zeros_like(G)
    delta[G.shape[0] // 2, G.shape[1] // 2] = 1.0

    def filter_rgb(kernel):
        return np.stack(
            [convolve2d(image[..., c], kernel, mode="same", boundary="fill", fillvalue=0) for c in range(3)],
            axis=-1,
        )

    blurred = filter_rgb(G)
    high_freq = image - blurred
    plt.imsave(f"{prefix}_blurred.png", np.clip(blurred, 0, 1))
    plt.imsave(f"{prefix}_high_freq.png", np.clip(high_freq + 0.5, 0, 1))

    for alpha in alphas:
        unsharp_kernel = (1 + alpha) * delta - alpha * G
        sharpened = filter_rgb(unsharp_kernel)
        plt.imsave(f"{prefix}_sharpened_{alpha}.png", np.clip(sharpened, 0, 1))


def apply_filters(image_path):
    # load image as grayscale
    image = np.array(Image.open(image_path).convert("L"), dtype=float) / 255.0

    # filters
    box_filter = np.ones((9, 9)) / 81
    Dx = np.array([[1, 0, -1]])
    Dy = np.array([[1],
                   [0],
                   [-1]])

    # convolve
    blurred = two_loop_convolution(image, box_filter)
    dx = two_loop_convolution(image, Dx)
    dy = two_loop_convolution(image, Dy)

    # display
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")

    axes[1].imshow(blurred, cmap="gray")
    axes[1].set_title("9x9 Box Filter")

    axes[2].imshow(np.abs(dx), cmap="gray")
    axes[2].set_title("Dx")

    axes[3].imshow(np.abs(dy), cmap="gray")
    axes[3].set_title("Dy")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_runtimes()
    apply_filters("aarav.jpg")


