from typing import Union

import numpy
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import yaml


class Object:

    def __init__(self, value: str, mask: np.ndarray):
        self.value = value
        self.mask = mask
        self.bounding_box = []
        self.calculate_bbox()

    def calculate_bbox(self):
        min_x = self.mask[1].min
        max_x = self.mask[1].max
        min_y = self.mask[0].min
        max_y = self.mask[0].max
        self.bounding_box = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]


class Text:

    def __init__(self, text: str, font: PIL.ImageFont.FreeTypeFont, color: tuple = (0, 0, 0)):
        self.text = text
        self.font = font
        self.size = self.font.size
        self.color = color
        self.objects = []
        self.words = []
        self.lines = []


class Box:

    def __init__(self, size: tuple, name: str = ''):
        self.size = size
        self.name = name
        self.text = []
        self.image = []
        self.text_background = []

    def add_text(self, text: Text, background_color: tuple = (255, 255, 255), indentation: int = (10, 10)) -> None:
        self.text_background = Image.new('RGB', self.size, color=background_color)
        draw = ImageDraw.Draw(self.text_background)

        offset_x = 0
        offset_y = 0

        # generate text char by char
        for char in text.text:
            char_size = draw.textsize(char, text.font)
            char_image = Image.new('RGB', char_size, color=background_color)
            draw_tool = ImageDraw.Draw(char_image)
            draw_tool.text((0, 0), char, (0, 0, 0))
            coords = (np.asarray(char_image.convert('L')) < 255).nonzero()
            if char_size[0] + offset_x >= (self.size[0] - 2 * indentation[0]):
                offset_x = 0
                offset_y += char_size[1]
                global_coords = (coords[0] + offset_y, coords[1] + offset_x)
                draw.text((indentation[0] + offset_x, indentation[0] + offset_y), char, text.color, text.font)
                offset_x += char_size[0]
            else:
                global_coords = (coords[0] + offset_y, coords[1] + offset_x)
                draw.text((indentation[0] + offset_x, indentation[0] + offset_y), char, text.color, text.font)
                offset_x += char_size[0]

        obj = Object(value=char, mask=global_coords)
        text.objects.append(obj)


class Document:

    def __init__(self, size: Union[str, tuple]):
        if isinstance(size, str):
            size_low = str.lower(size)
            if size_low == 'a4':
                shape = (210, 297)
            elif size_low == 'a3':
                shape = (297, 420)
        elif isinstance(size, tuple):
            shape = size
        else:
            raise 'Unexpected size data type!'
        self.shape = shape
        self.image = Image.new('RGB', self.shape, color=(200, 255, 255))
        self.objects = []
        self.background = np.zeros(shape)

    def add_box(self, box: Box, location: tuple = (0, 0)) -> None:
        if all(x < y for x, y in zip(box.size, self.shape)):
            self.objects.append(box)
            self.image.paste(box.text_background, location)
        else:
            raise 'Box is too large for the document.'


def main():
    my_doc = Document('a4')
    box1 = Box((70, 70), 'box1')
    text_b1 = Text('It\'s a beatifull day.', ImageFont.truetype('./comic.ttf', 12))
    box1.add_text(text_b1)
    my_doc.add_box(box1, (10, 10))
    box2 = Box((100, 50), 'box2')
    strings = 'Ahoj.'
    text_b2 = Text(strings, ImageFont.truetype('./LITERPLA.ttf', 17))
    box2.add_text(text_b2)
    my_doc.add_box(box2, (80, 10))
    plt.imshow(my_doc.image)


main()