from PIL import Image
import numpy as np
import random

def kmeans(rgb, k):
    print(rgb)

def closest_centroid_index(pixel, centroids):
    distances = [euclidean_distance(pixel, centroid) for centroid in centroids]
    return distances.index(min(distances))

def euclidean_distance(pixel1, pixel2):
    return sum((a - b) ** 2 for a, b in zip(pixel1, pixel2)) ** 0.5

def kmeans_helper(original_image, k):
    w, h = original_image.size
    pixels = list(original_image.getdata())
    kmeans(pixels, k)
    kmeans_image = Image.new(original_image.mode, original_image.size)
    kmeans_image.putdata(pixels)
    return kmeans_image

def main(input_image_path, k, output_image_path):
    original_image = Image.open(input_image_path)
    kmeans_image = kmeans_helper(original_image, k)
    kmeans_image.save(output_image_path)


if __name__ == "__main__":
    ks = [2, 5, 10, 15, 20]
    for k in ks:
        main("Koala.jpg", k, "Koala_with_{}.jpg".format(k))
