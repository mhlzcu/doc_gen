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
from perlin_numpy import generate_perlin_noise_2d
import random
import cv2


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

    def __init__(self, text: str, font: PIL.ImageFont.FreeTypeFont, color: tuple = (0, 0, 0), underline: bool = 0,
                 underline_width: int = 3, underline_offset: int = 2):
        self.text = text
        self.font = font
        self.size = self.font.size
        self.color = color
        self.underline = underline
        self.underline_width = underline_width
        self.underline_offset = underline_offset
        self.objects = []
        self.words = []
        self.words_bb = []
        self.lines = []
        self.lines_bb = []


class Augmentation:

    def __init__(self):
        self.fonts: list[PIL.ImageFont.FreeTypeFont] = []
        self.augmentation_num: list[tuple[int, int]] = []
        self.characters_offsets: list[list[str]] = []
        self.offsets: list[list[tuple[int, int]]] = []
        self.characters_masks: list[list[str]] = []
        self.noise_masks: list[list[np.ndarray]] = []

    def add_fonts(self, fonts: list[PIL.ImageFont.FreeTypeFont],
                  texts: list[Text],
                  offset_range: tuple = (-5, 5),
                  noise_octave: int = 8):

        for font in fonts:
            chars: str = ''
            for text in texts:
                if text.font == font:
                    chars += text.text.replace(" ", "")

            self.fonts.append(font)
            aug_num_noise = np.random.randint(1, 10)
            aug_num_offset = np.random.randint(1, 10)
            self.augmentation_num.append((aug_num_noise, aug_num_offset))
            offsets = []
            draw = ImageDraw.Draw(Image.new('RGB', (256, 256), color=(255, 255, 255)))
            for i in range(aug_num_offset):
                offsets.append((np.random.randint(offset_range[0], np.random.randint(offset_range[1])),
                                np.random.randint(offset_range[0], np.random.randint(offset_range[1]))))
            self.offsets.append(offsets)
            unique_chars = list(set(chars))
            self.characters_offsets.append(random.sample(unique_chars, aug_num_offset))
            masks = []
            chars_to_aug = random.sample(unique_chars, aug_num_noise)
            self.characters_masks.append(chars_to_aug)
            for character in chars_to_aug:
                np.random.seed()
                char_size_x, char_size_y = draw.textsize(character, font)
                char_size_x += 10
                char_size_y += 10
                mask_x, mask_y = (char_size_x, char_size_y)
                if char_size_x % noise_octave:
                    mask_x += noise_octave - (char_size_x % noise_octave)
                if char_size_y % noise_octave:
                    mask_y += noise_octave - (char_size_y % noise_octave)
                noise = generate_perlin_noise_2d((mask_y, mask_x), (noise_octave, noise_octave))
                noise_mask = noise[0:char_size_y, 0: char_size_x]
                masks.append(noise_mask)
            self.noise_masks.append(masks)


class Box:

    def __init__(self, size: tuple, name: str = '', background_color: tuple = (255, 255, 255)):
        self.size = size
        self.name = name
        self.text = []
        self.image = Image.new('RGB', self.size, color=background_color)
        self.top_left_corner = (0, 0)
        self.offset_x = 0
        self.offset_y = 0
        self.augmentations: Augmentation

    def add_text(self, text: Text, augmentations: Augmentation = 0, indentation: int = (10, 10)) -> None:
        draw = ImageDraw.Draw(self.image)
        self.text.append(text)
        self.offset_x = 0
        word_coords = []
        line_coords = []
        new_line = 0
        # generate text char by char
        for idx, char in enumerate(text.text):
            char_size_x, char_size_y = draw.textsize(char, text.font)
            char_offset = text.font.getoffset(char)
            im_aug = np.zeros(1)
            if char != ' ':
                char_image = Image.new('RGB', (char_size_x + 10, char_size_y + 10), color=(255, 255, 255))
                draw_tool = ImageDraw.Draw(char_image)
                draw_tool.text((10, 10), char, text.color, text.font)
                coords = (np.asarray(char_image.convert('L')) < 255).nonzero()
                coords_glob = (coords[0] - 10, coords[1] - 10)
                if text.font in augmentations.fonts:
                    font_id = augmentations.fonts.index(text.font)
                    if char in augmentations.characters_offsets[font_id]:
                        char_id = augmentations.characters_offsets[font_id].index(char)
                        offset = augmentations.offsets[font_id][char_id]
                        coords_glob = (coords_glob[0] + offset[0], coords_glob[1] + offset[1])

                if text.font in augmentations.fonts:
                    font_id = augmentations.fonts.index(text.font)
                    if char in augmentations.characters_masks[font_id]:
                        char_id = augmentations.characters_masks[font_id].index(char)
                        mask = augmentations.noise_masks[font_id][char_id]
                        char_mask = (np.asarray(char_image.convert('RGB')) == 255).astype(np.uint8) * 255
                        char_im = np.asarray(char_image.convert('RGB'))
                        mask = np.dstack([(mask * 255).astype(int)]*3)
                        im_aug = cv2.add(char_im, mask, dtype=0)
                        im_aug = cv2.bitwise_or(im_aug, char_mask)
            else:
                char_image = Image.new('RGB', (char_size_x, char_size_y))
                draw_tool = ImageDraw.Draw(char_image)
                draw_tool.text((0, 0), char, (0, 0, 0), text.font)
                coords = (np.asarray(char_image.convert('L')) == 0).nonzero()

            if char_size_x + self.offset_x >= (self.size[0] - 2 * indentation[0]):
                if text.underline:
                    self.offset_y += char_size_y + text.underline_width + text.underline_offset
                else:
                    self.offset_y += char_size_y
                self.offset_x = 0

                new_line = 1

            if char_size_y + self.offset_y >= (self.size[0] - 2 * indentation[0]):
                warnings.warn('Warning: Text too large for the box.')
                if line_coords:
                    text.lines.append(line_coords)
                if word_coords:
                    text.words.append(word_coords)
                break

            global_coords = (indentation[0] + coords_glob[0] + self.offset_y,
                             indentation[1] + coords_glob[1] + self.offset_x)
            if im_aug.any():
                pixels = self.image.load()
                for glob, loc in zip(np.asarray(global_coords).T, np.asarray(coords).T):
                    pixels[tuple(glob[::-1])] = tuple(im_aug[tuple(loc)])
            else:
                draw.text((indentation[1] + self.offset_x, indentation[0] + self.offset_y), char, text.color, text.font)
            self.offset_x += char_size_x
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

        if text.underline:
            self.offset_y += char_size_y + text.underline_width + text.underline_offset
        else:
            self.offset_y += char_size_y

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
        self.shape: tuple[int, int] = shape
        self.image: PIL.Image = Image.new('RGB', self.shape, color=(200, 255, 255))
        self.boxes: list = []
        self.background: np.ndarray = np.zeros(shape)
        self.augmentations: Augmentation = Augmentation()

    def add_box(self, box: Box, location: tuple) -> None:
        size_pos = tuple(map(operator.add, box.size, location))
        if all(x < y for x, y in zip(size_pos, self.shape)):
            box.top_left_corner = location
            self.boxes.append(box)
            self.image.paste(box.image, location)
            for idx, text in enumerate(box.text):
                if text.underline:
                    self.add_underline(len(self.boxes) - 1, idx)
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

    def add_underline(self, box_id, text_id):
        draw = ImageDraw.Draw(self.image)
        for bbox in self.boxes[box_id].text[text_id].lines_bb:
            (min_x, min_y), (max_x, _), (_, max_y), _ = \
                ([tuple(map(operator.add, x, self.boxes[box_id].top_left_corner)) for x in bbox])
            underline_y_offset = self.boxes[box_id].text[text_id].underline_offset
            draw.line((min_x, max_y + underline_y_offset + 1,
                       max_x, max_y + underline_y_offset + 1),
                      fill=self.boxes[box_id].text[text_id].color,
                      width=self.boxes[box_id].text[text_id].underline_width)


def main():
    my_doc = Document('a4')

    boxes = []
    texts = []
    fonts = []

    box1 = Box((500, 500), 'box1')
    box2 = Box((300, 300), 'box2')

    boxes.append(box1)
    boxes.append(box2)

    font1 = ImageFont.truetype('./comic.ttf', 32)
    font2 = ImageFont.truetype('./luckytw.ttf', 25)
    font3 = ImageFont.truetype('./LITERPLA.ttf', 32)

    fonts.append(font1)
    fonts.append(font2)
    fonts.append(font3)

    text_b1 = Text('It\'s a beatifull day.', font1, underline=1, underline_width=3,
                   underline_offset=3, color=(255, 0, 0))
    text_b1_2 = Text('Protikorupční organizace Transparency International současně také sdělila,'
                     'že Blažek by neměl být ministrem, protože by to budilo pochybnosti o ovlivňování vyšetřování'
                     'zmíněných kauz.', font2)
    text_b2 = Text(r'Мой распорядок дня.', font3, underline=True, underline_width=3)

    texts.append(text_b1)
    texts.append(text_b1_2)
    texts.append(text_b2)

    augment = Augmentation()
    augment.add_fonts(fonts, texts)

    box1.add_text(text_b1, augment)
    box1.add_text(text_b1_2, augment)
    box2.add_text(text_b2, augment)

    my_doc.add_box(box1, (10, 10))
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
    ax.imshow(box2.image)
    for obj in box2.text[0].objects:
        poly = patches.Polygon(obj.bounding_box, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(poly)
    plt.show()

    fig, ax = plt.subplots()
    ax.imshow(box1.image)
    for bb in box1.text[0].words_bb:
        poly = patches.Polygon(bb, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(poly)
    plt.show()

    fig, ax = plt.subplots()
    ax.imshow(box1.image)
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

    my_doc.image.show()
    a = 0


main()
