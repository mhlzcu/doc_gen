# -*- coding: utf-8 -*-

from typing import Union, Tuple, Any

import numpy
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib import patches
import yaml
import operator
import warnings
from operator import itemgetter

class Object:

    def __init__(self, value: str, mask: np.ndarray):
        self.value = value
        self.mask = mask
        self.bounding_box = []
        self.calculate_bbox()

    def calculate_bbox(self):
        min_x = self.mask[1].min()
        max_x = self.mask[1].max()
        min_y = self.mask[0].min()
        max_y = self.mask[0].max()
        self.bounding_box = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]


class Text:

    def __init__(self, text: str, font: PIL.ImageFont.FreeTypeFont, color: tuple = (0, 0, 0)):
        self.text = text
        self.font = font
        self.size = self.font.size
        self.color = color
        self.objects = []
        self.words = []
        self.words_bb = []
        self.lines = []
        self.lines_bb = []


class Box:

    def __init__(self, size: tuple, name: str = ''):
        self.size = size
        self.name = name
        self.text = []
        self.image = []
        self.text_background = []
        self.top_left_corner = (0, 0)

    def add_text(self, text: Text, background_color: tuple = (255, 255, 255), indentation: int = (10, 10)) -> None:
        self.text_background = Image.new('RGB', self.size, color=background_color)
        draw = ImageDraw.Draw(self.text_background)
        self.text.append(text)
        offset_x = 0
        offset_y = 0
        word_coords = []
        line_coords = []
        new_line = 0
        # generate text char by char
        for idx, char in enumerate(text.text):
            char_size_x, char_size_y = draw.textsize(char, text.font)
            char_offset = text.font.getoffset(char)
            if char != ' ':
                char_image = Image.new('RGB', (char_size_x + 10, char_size_y + 10), color=background_color)
                draw_tool = ImageDraw.Draw(char_image)
                draw_tool.text((10, 10), char, (0, 0, 0), text.font)
                coords = (np.asarray(char_image.convert('L')) < 255).nonzero()
                coords = (coords[0] - 10, coords[1] - 10)
            else:
                char_image = Image.new('RGB', (char_size_x, char_size_y))
                draw_tool = ImageDraw.Draw(char_image)
                draw_tool.text((0, 0), char, (0, 0, 0), text.font)
                coords = (np.asarray(char_image.convert('L')) == 0).nonzero()

            if char_size_x + offset_x >= (self.size[0] - 2 * indentation[0]):
                offset_x = 0
                offset_y += char_size_y
                new_line = 1

            if char_size_y + offset_y >= (self.size[0] - 2 * indentation[0]):
                warnings.warn('Warning: Text too large for the box.')
                if line_coords:
                    text.lines.append(line_coords)
                if word_coords:
                    text.words.append(word_coords)
                break

            global_coords = (indentation[0] + coords[0] + offset_y, indentation[1] + coords[1] + offset_x)
            draw.text((indentation[1] + offset_x, indentation[0] + offset_y), char, text.color, text.font)
            offset_x += char_size_x
            # print(char)
            obj = Object(value=char, mask=global_coords)

            if char != ' ':
                if new_line:
                    if line_coords:
                        text.lines.append(line_coords)
                    line_coords = [obj]
                    new_line = 0
                    if word_coords:
                        text.words.append(word_coords)
                    word_coords = [obj]
                else:
                    word_coords.append(obj)
                    line_coords.append(obj)
            else:
                if word_coords:
                    text.words.append(word_coords)
                word_coords = []

            if idx == (len(text.text) - 1):
                if line_coords:
                    text.lines.append(line_coords)
                if word_coords:
                    text.words.append(word_coords)

            text.objects.append(obj)

        for word in text.words:
            coords = []
            for obj in word:
                coords.extend(obj.bounding_box)
            min_y = min(coords, key=itemgetter(1))[1]
            max_y = max(coords, key=itemgetter(1))[1]
            min_x = min(coords, key=itemgetter(0))[0]
            max_x = max(coords, key=itemgetter(0))[0]
            text.words_bb.append([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])

        for line in text.lines:
            coords = []
            for obj in line:
                coords.extend(obj.bounding_box)
            min_y = min(coords, key=itemgetter(1))[1]
            max_y = max(coords, key=itemgetter(1))[1]
            min_x = min(coords, key=itemgetter(0))[0]
            max_x = max(coords, key=itemgetter(0))[0]
            text.lines_bb.append([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])


class Document:

    def __init__(self, size: Union[str, tuple], dpi: int = 72):
        if isinstance(size, str):
            size_low = str.lower(size)
            if size_low == 'a4':
                shape = (int(8.3 * dpi), int(11.7 * dpi))
            elif size_low == 'a3':
                shape = (int(11.7 * dpi), int(16.5 * dpi))
        elif isinstance(size, tuple):
            shape = size
        else:
            raise TypeError('Document size must be string with (a0,...,a5) or size in pixels.')
        self.shape = shape
        self.image = Image.new('RGB', self.shape, color=(200, 255, 255))
        self.boxes = []
        self.background = np.zeros(shape)

    def add_box(self, box: Box, location: tuple) -> None:
        size_pos = tuple(map(operator.add, box.size, location))
        if all(x < y for x, y in zip(size_pos, self.shape)):
            box.top_left_corner = location
            self.boxes.append(box)
            self.image.paste(box.text_background, location)
        else:
            raise ValueError('Box size + location is larger then the document size.',
                             box.name, size_pos, 'vs.', self.shape)

    def get_text_bounding_boxes(self) -> list:
        bb_list = []
        for box in self.boxes:
            for text in box.text:
                for obj in text.objects:
                    bbox = obj.bounding_box
                    bb_list.append([tuple(map(operator.add, x, box.top_left_corner)) for x in bbox])
        return bb_list

    def get_words_bounding_boxes(self) -> list:
        bb_list = []
        for box in self.boxes:
            for text in box.text:
                for bbox in text.words_bb:
                    bb_list.append([tuple(map(operator.add, x, box.top_left_corner)) for x in bbox])
        return bb_list

    def get_lines_bounding_boxes(self) -> list:
        bb_list = []
        for box in self.boxes:
            for text in box.text:
                for bbox in text.lines_bb:
                    bb_list.append([tuple(map(operator.add, x, box.top_left_corner)) for x in bbox])
        return bb_list

def main():
    my_doc = Document('a4')
    box1 = Box((500, 500), 'box1')
    text_b1 = Text('It\'s a beatifull day.', ImageFont.truetype('./comic.ttf', 100))
    box1.add_text(text_b1)
    my_doc.add_box(box1, (10, 10))
    box2 = Box((300, 300), 'box2')
    strings = r'Мой распорядок дня.'
    text_b2 = Text(strings, ImageFont.truetype('./LITERPLA.ttf', 100))
    box2.add_text(text_b2)
    my_doc.add_box(box2, (10, 510))
    fig, ax = plt.subplots()
    ax.imshow(my_doc.image)
    bblist = my_doc.get_text_bounding_boxes()
    for bbox in bblist:
        poly = patches.Polygon(bbox, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(poly)
    plt.savefig('test.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    ax.imshow(box2.text_background)
    for obj in box2.text[0].objects:
        poly = patches.Polygon(obj.bounding_box, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(poly)
    plt.show()

    fig, ax = plt.subplots()
    ax.imshow(box1.text_background)
    for bb in box1.text[0].words_bb:
        poly = patches.Polygon(bb, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(poly)
    plt.show()

    fig, ax = plt.subplots()
    ax.imshow(box1.text_background)
    for bb in box1.text[0].lines_bb:
        poly = patches.Polygon(bb, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(poly)
    plt.show()

    bblist = my_doc.get_lines_bounding_boxes()
    fig, ax = plt.subplots()
    ax.imshow(my_doc.image)
    for bbox in bblist:
        poly = patches.Polygon(bbox, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(poly)
    plt.show()

    a = 0


main()
