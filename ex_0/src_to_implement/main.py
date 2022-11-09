# Python imports

# Self imports
from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

# Other imports

# Codes Here
# checker = Checker(250, 25)
# checker.draw()
# checker.show()

# circle = Circle(1024, 200, (512, 256))
# circle.draw()
# circle.show()

# spectrum = Spectrum(10)
# spectrum.draw()
# spectrum.show()

img_gen = ImageGenerator(
    file_path="/Users/pranto/Desktop/dl-exercises-fau/ex_0/src_to_implement/exercise_data",
    label_path="/Users/pranto/Desktop/dl-exercises-fau/ex_0/src_to_implement/Labels.json",
    shuffle=True,
    rotation=False,
    mirroring=False,
    batch_size=12,
    image_size=[100, 100, 3],
)

img_gen.show()
# img_gen.next()
# img_gen.next()
