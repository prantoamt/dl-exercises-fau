from pattern import *
from generator import *


if __name__ == '__main__':
    # checker = Checker(250, 25)
    # checker.show()

    # circle = Circle(1024, 200, (512, 256))
    # circle.show()

    # spectrum = Spectrum(255)
    # spectrum.show()

    file_path   = os.path.join(os.getcwd(), "exercise_data")
    label_path  = os.path.join(os.getcwd(), "Labels.json")
    img_gen     = ImageGenerator(file_path, label_path, 15, [32, 32, 3], rotation=True, mirroring=False, shuffle=True)
    img_gen.show()
