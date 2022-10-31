from predict import Predictor
import os


# Currently this takes in a folder of .jpg/.png
# it can also take individual img file paths.
# TBD: can it take straight PILLOW images?

# Returns: `output` a list of strings. Between 0 and NUM_REL.
NUM_REL = 10
IMG__PATH = '/home/kastan/thesis/data/simple_test_data/'

my_pred = Predictor()
my_pred.setup()

valid_images = [".jpg",".png"]
for f in os.listdir(IMG__PATH):
    ext = os.path.splitext(f)[1]

    if ext.lower() not in valid_images:
        continue
    else:
        if f!= 'spacex_goup_photo.png':
            print(f)
            output = my_pred.predict(os.path.join(IMG__PATH,f), num_rel = NUM_REL)
            print(output)
