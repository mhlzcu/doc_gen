# -*- coding: utf-8 -*-

from typing import Union, Tuple, List

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import operator
import warnings
from operator import itemgetter
from perlin_numpy import generate_perlin_noise_2d
import random
import cv2
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

import consts


class Object:

    def __init__(self, value: str, mask: np.ndarray, local_coords: np.ndarray = np.zeros(1),
                 augmented_mask: np.ndarray = np.zeros(1)):
        self.value = value
        self.mask = mask
        self.augmented_image = augmented_mask
        self.augmented_image_coords = local_coords
        self.bounding_box = []
        self.calculate_bbox()

    def calculate_bbox(self):
        min_x = self.mask[1].min()
        max_x = self.mask[1].max()
        min_y = self.mask[0].min()
        max_y = self.mask[0].max()
        self.bounding_box = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]


class Text:

    def __init__(self, text: str, font: PIL.ImageFont.FreeTypeFont, color: Tuple = (0, 0, 0), underline: bool = 0,
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
        self.fonts: List[PIL.ImageFont.FreeTypeFont] = []
        self.augmentation_num: List[Tuple[int, int]] = []
        self.characters_offsets: List[List[str]] = []
        self.offsets: List[List[Tuple[int, int]]] = []
        self.characters_masks: List[List[str]] = []
        self.noise_masks: List[List[np.ndarray]] = []

    def add_fonts(self, fonts: List[PIL.ImageFont.FreeTypeFont],
                  texts: List[Text],
                  offset_range: tuple = (-3, 3),
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
            # chars_to_aug = random.sample(unique_chars, aug_num_noise)
            self.characters_masks.append(unique_chars)
            for character in unique_chars:
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

    def __init__(self, size: Tuple, name: str = '', background_color: Tuple = (255, 255, 255)):
        self.size = size
        self.name = name
        self.text = []
        self.image = Image.new('RGB', self.size, color=background_color)
        self.top_left_corner = (0, 0)
        self.offset_x = 0
        self.offset_y = 0
        self.augmentations: Augmentation

    def add_text(self, text: Text, augmentations: Augmentation = 0, indentation: Tuple[int, int] = (10, 10),
                 kern_gap: int = 0, max_lines: int = 0, max_char_per_line: int = 0) -> None:
        draw = ImageDraw.Draw(self.image)
        self.text.append(text)
        self.offset_x = 0
        word_coords = []
        line_coords = []
        new_line = 0
        # generate text char by char
        char_count = 0
        lines_count = 0
        for idx, char in enumerate(text.text):
            char_size_x, char_size_y = draw.textsize(char, text.font)
            char_offset = text.font.getoffset(char)
            im_aug = np.zeros(1)
            # print(f"Char: {char}")
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
                        # print(char)
                        im_aug = cv2.add(char_im, mask, dtype=0)
                        im_aug = cv2.bitwise_or(im_aug, char_mask)
            else:
                char_image = Image.new('RGB', (char_size_x, char_size_y))
                draw_tool = ImageDraw.Draw(char_image)
                draw_tool.text((0, 0), char, (0, 0, 0), text.font)
                coords = (np.asarray(char_image.convert('L')) == 0).nonzero()

            char_count += 1

            if (char_size_x + self.offset_x >= (self.size[0] - 2 * indentation[0]))\
                    or char_count > max_char_per_line:
                if text.underline:
                    self.offset_y += char_size_y + text.underline_width + text.underline_offset
                else:
                    self.offset_y += char_size_y
                self.offset_x = 0

                new_line = 1
                lines_count += 1

            if (char_size_y + self.offset_y >= (self.size[1] - 2 * indentation[1]))\
                    or lines_count == max_lines:
                warnings.warn('Text too large for the box.')
                if line_coords:
                    text.lines.append(line_coords)
                if word_coords:
                    text.words.append(word_coords)
                break

            gap_x = 0
            if idx > 0:
                if self.offset_x > 0:
                    prev_char = text.text[idx - 1]
                    if char != ' ' or prev_char != ' ':
                        prev_size_x, _ = draw.textsize(prev_char, text.font)
                        kern_size_x, _ = draw.textsize(prev_char + char, text.font)
                        # kern_image = Image.new('RGB', (kern_size_x + 10, kern_size_y + 10), color=(255, 255, 255))
                        gap_x = kern_size_x - (char_size_x + prev_size_x) - kern_gap
                    self.offset_x += gap_x
            global_coords = (indentation[0] + coords_glob[0] + self.offset_y,
                             indentation[1] + coords_glob[1] + self.offset_x)
            if im_aug.any():
                pixels = self.image.load()
                for glob, loc in zip(np.asarray(global_coords).T, np.asarray(coords).T):
                    pixels[tuple(glob[::-1])] = tuple(im_aug[tuple(loc)])
                obj = Object(value=char, mask=global_coords, local_coords=coords, augmented_mask=im_aug)
            else:
                draw.text((indentation[1] + self.offset_x, indentation[0] + self.offset_y), char, text.color, text.font)
                obj = Object(value=char, mask=global_coords)
            self.offset_x += char_size_x
            # print(char)
            # obj = Object(value=char, mask=global_coords)

            if char != ' ':
                if new_line:
                    if line_coords:
                        text.lines.append(line_coords)
                    line_coords = [obj]
                    new_line = 0
                    char_count = 1
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
        self.shape: Tuple[int, int] = shape
        self.image: PIL.Image = Image.new('RGB', self.shape, color=(255, 255, 255))
        self.boxes: List[Box] = []
        self.background: np.ndarray = np.zeros(shape)
        self.augmentations: Augmentation = Augmentation()

    def add_box(self, box: Box, location: Tuple) -> None:
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

    # def add_line_text_box(self, text: Text) -> None:
    #     if self.boxes:
    #         position = tuple(map(operator.add, self.boxes[-1].top_left_corner, (0, self.boxes[-1].size[1])))
    #     else:
    #         position = (0, 0)
    #     size = text.font.getsize("0123456789")
    #     box = Box((self.shape[0], size[1]), 'box' + str(len(self.boxes) + 1))
    #     box.add_text(text)
    #     self.add_box(box, position)

    def get_text_bounding_boxes(self) -> List:
        bb_list = []
        for box in self.boxes:
            for text in box.text:
                for obj in text.objects:
                    bbox = obj.bounding_box
                    bb_list.append([tuple(map(operator.add, x, box.top_left_corner)) for x in bbox])
        return bb_list

    def get_words_bounding_boxes(self) -> List:
        bb_list = []
        for box in self.boxes:
            for text in box.text:
                for bbox in text.words_bb:
                    bb_list.append([tuple(map(operator.add, x, box.top_left_corner)) for x in bbox])
        return bb_list

    def get_lines_bounding_boxes(self) -> List:
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

    def add_background(self, background_image, blend_ratio=0.15):
        image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        new_image = Image.new('RGBA', self.shape, color=(255, 255, 255, 0))
        back_image = Image.fromarray(image).convert("RGBA")
        back_image = back_image.resize(self.image.size)
        for box in self.boxes:
            for text in box.text:
                for obj in text.objects:
                    if obj.augmented_image.any():
                        pixels = new_image.load()
                        for glob, loc in zip(np.asarray(obj.mask).T, np.asarray(obj.augmented_image_coords).T):
                            glob_position = tuple(glob[::-1])
                            pixels[tuple(map(operator.add, glob_position, box.top_left_corner))] = \
                                tuple(obj.augmented_image[tuple(loc)]) + (127,)
                    elif obj.value != ' ':
                        for glob in np.asarray(obj.mask).T:
                            glob_position = tuple(glob[::-1])
                            pixels[tuple(map(operator.add, glob_position, box.top_left_corner))] = text.color + (127,)
                a = 0

        self.image = Image.alpha_composite(back_image, new_image)


def main(args):

    out_images_dir = args.output_dir / 'lines'
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels = args.output_dir / 'labels.txt'
    labels = out_labels.open('w')
    fonts_info = open(args.output_dir / 'font_info.txt', 'w')

    fonts = list(args.fonts_dir.rglob('*.ttf'))
    # font = ImageFont.truetype('./luckytw.ttf', 40)
    # font2 = ImageFont.truetype('./luckytw.ttf', 25)
    # font3 = ImageFont.truetype('./LITERPLA.ttf', 32)

    index = 0
    for text_file in args.text_files:

        with open(text_file, 'r') as in_file:
            for line in tqdm(in_file.readlines()):
                font_path = np.random.choice(fonts)
                if 'koala' in str(font_path):
                    continue
                font = ImageFont.truetype(str(font_path), 40)
                text = Text(line.strip(), font)

                my_doc = Document((5000, 500), dpi=300)
                box = Box((consts.max_width, consts.height * 2), 'box')

                augment = Augmentation()
                # print(len(line.strip()))
                # print(line.strip())
                box_indentation = (10, 10)
                try:
                    box.add_text(text, augment, max_lines=1, max_char_per_line=consts.max_characters,
                                 indentation=box_indentation)
                except UnboundLocalError as ex:
                    print(ex)
                    print(f"Error in line: {line.strip()} with font: {font_path.name}")
                    fonts_info.write(f"UnboundLocalError: {index}: {str(font_path)}\n")
                    continue

                except ValueError as err:
                    print(err)
                    print(f"Error in line: {line.strip()} with font: {font_path.name}")
                    fonts_info.write(f"ValueError: {index}: {str(font_path)}\n")
                    continue

                my_doc.add_box(box, (10, 10))
                # my_doc.add_background(cv2.imread(r'.\recycled-paper.jpg'))

                margin = 5
                bbox = my_doc.boxes[0].text[0].lines_bb[0]
                min_x = bbox[0][0] + box_indentation[0] - margin
                max_x = bbox[1][0] + box_indentation[0] + margin
                min_y = bbox[0][1] + box_indentation[1] - margin
                max_y = bbox[2][1] + box_indentation[1] + margin
                img = np.array(my_doc.image)[min_y:max_y, min_x:max_x]

                printed_text = my_doc.boxes[0].text[0].text
                label = f'{index} {printed_text}\n'
                labels.write(label)

                # my_doc.image.save(out_images_dir / f'{index}.png')
                cv2.imwrite(str(out_images_dir / f'{index}.png'), img)

                fonts_info.write(f"{index}: {str(font_path)}\n")
                index += 1

    labels.close()
    fonts_info.close()


def parse_args():

    parser = ArgumentParser()

    parser.add_argument('--text_files', required=True,
                        nargs='+', type=Path, help='Path to input file/s with texts to generate')
    parser.add_argument('--fonts_dir', required=True,
                        type=Path, help='Path to directory containing fonts in ttf format')
    parser.add_argument('--output_dir', type=Path, default='outputs', help='Path to directory to save outputs')

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
