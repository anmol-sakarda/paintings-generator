# import necessary libraries
import os
from PIL import Image, ImageEnhance


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

path = "dataset/Images/"
images_path = "scraped_images/"
dirs = os.listdir(images_path)


# resizing images to all be the same size. Sharpening each image and then flipping them
def resize_flip():
    for item in dirs:
        if os.path.isfile(images_path + item):
            im = Image.open(images_path + item)
            item_name = item[:-4]
            # resizing image
            imResize = im.resize((256, 256), Image.ANTIALIAS)
            # sharpening the base image
            im_enhancer = ImageEnhance.Sharpness(imResize)
            imResized = im_enhancer.enhance(2)
            normal_sharpen = im_enhancer.enhance(4)
            # flipping the image
            flipped = imResize.transpose(Image.FLIP_LEFT_RIGHT)
            im_flip_enhancer = ImageEnhance.Sharpness(flipped)
            flipped_image = im_flip_enhancer.enhance(2)
            flip_sharpen = im_flip_enhancer.enhance(4)
            # saving all 4 images
            imResized.save("dataset/Images/" + item_name + "_resized.jpg", 'JPEG', quality=90)
            flipped_image.save("dataset/Images/" + item_name + "_resized_flipped.jpg", 'JPEG', quality=90)
            normal_sharpen.save("dataset/Images/" + item_name + "_resized_sharpen.jpg", 'JPEG', quality=90)
            flip_sharpen.save("dataset/Images/" + item_name + "_resized_flipped_sharpen.jpg", 'JPEG', quality=90)


resize_flip()
