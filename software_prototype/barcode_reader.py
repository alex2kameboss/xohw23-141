# -> barcode -> value

from pyzbar.pyzbar import decode
import cv2
import numpy as np


def gaussian_filter(kernel_size, img, sigma=1, muu=0):
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x ** 2 + y ** 2)
    normal = 1 / (((2 * np.pi) ** 0.5) * sigma)
    gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2))) * normal
    gauss = np.pad(gauss, [(0, img.shape[0] - gauss.shape[0]), (0, img.shape[1] - gauss.shape[1])], 'constant')
    return gauss


def fft_deblur(img, kernel_size, kernel_sigma=5, factor='wiener', const=0.002):
    gauss = gaussian_filter(kernel_size, img, kernel_sigma)
    img_fft = np.fft.fft2(img)
    gauss_fft = np.fft.fft2(gauss)
    weiner_factor = 1 / (1 + (const / np.abs(gauss_fft) ** 2))
    if factor != 'wiener':
        weiner_factor = factor
    recon = img_fft / gauss_fft
    recon *= weiner_factor
    recon = np.abs(np.fft.ifft2(recon))
    return recon


def get_barcode(img, coords):
    h, w, c = img.shape

    x_min = int(min(max(0.0, coords[0] - 50), coords[2]))
    y_min = int(min(max(0.0, coords[1] - 50), coords[3]))
    x_max = int(max(min(w, coords[2] + 50), coords[0]))
    y_max = int(max(min(h, coords[3] + 50), coords[1]))

    ret_image = img[y_min:y_max, x_min:x_max]

    # image = cv2.cvtColor(ret_image, cv2.COLOR_BGR2GRAY)
    # se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    # bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    # out_gray = cv2.divide(image, bg, scale=255)
    # out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]

    # print("barcode gray:", barcode_reader(out_gray))
    # print("barcode binary:", barcode_reader(out_binary))

    # cv2.imshow(f"barcode {i} gray", out_gray)
    # cv2.imshow(f"barcode {i} binary", out_binary)

    return ret_image


def barcode_reader(img):
    detectedBarcodes = decode(img)

    if detectedBarcodes:
        return detectedBarcodes[0].data.decode('utf-8')

    new_image = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
    detectedBarcodes = decode(new_image)

    if detectedBarcodes:
        return detectedBarcodes[0].data.decode('utf-8')

    new_image = fft_deblur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7, 5, factor='wiener', const=0.5)
    detectedBarcodes = decode(new_image)

    if detectedBarcodes:
        return detectedBarcodes[0].data.decode('utf-8')

    return ''


if __name__ == '__main__':
    img = cv2.imread('./0_0.jpg')
    print(barcode_reader(img))
