from PIL import Image
import os

data_path = "/home/nthom/Documents/fracture/data"
save_path = "/home/nthom/Documents/fracture/data_160x120"
directories = os.listdir(data_path)
images = []

for image_directory in directories:
    if not os.path.isdir(save_path + "/" + image_directory):
        os.mkdir(save_path + "/" + image_directory)
    for image_name in os.listdir(data_path + "/" + image_directory):
        if os.path.isfile(data_path+ "/" + image_directory+ "/" + image_name):
            im = Image.open(data_path+ "/" + image_directory+ "/" + image_name)
            width, height = im.size
            imResize = im.resize((int(width * 0.25), int(height * 0.25)), Image.ANTIALIAS)
            imResize.save(save_path + "/" + image_directory+ "/" + image_name, "png")