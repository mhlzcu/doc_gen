import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2
from typing import List


def font_character_test(font_path: str, test_text: str) -> bool:
    contains_all = True
    font = TTFont(font_path)
    for character in test_text:
        for table in font['cmap'].tables:
            if ord(character) in table.cmap.keys():
                contains_all = True
                break
            else:
                contains_all = False
        if not contains_all:
            break

    return contains_all


def pil_character_test(font_path: str, test_text: str) -> (bool, bool):
    can_generate = True
    contains_empty_masks = False
    font = ImageFont.truetype(font_path, 10)
    for character in test_text:
        try:
            im = Image.new("RGB", (100, 100), (255, 255, 255))
            draw = ImageDraw.Draw(im)
            draw.text((10, 10), character, font=font)
            im_np = np.array(im)
            if im_np.min() == 255:
                contains_empty_masks = True
        except:
            can_generate = False
            return can_generate, contains_empty_masks

    return can_generate, contains_empty_masks


def main(font_list: List, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # path = './LITERPLA.ttf'
    test_string = 'LOĎ ČEŘÍ KÝLEM TŮŇ OBZVLÁŠŤ V GRÓNSKÉ ÚŽINĚ. \
    PŘÍLIŠ ŽLUŤOUČKÝ KŮŇ ÚPĚL ĎÁBELSKÉ KÓDY. \
    Loď čeří kýlem tůň obzvlášť v Grónské úžině. \
    Příliš žluťoučký kůň úpěl ďábelské kódy.'
    for path in font_list:
        contains_test = font_character_test(path, test_string)
        printable_test, empty_masks = pil_character_test(path, test_string)
        print(f'Font {path.name} supports testing string: ', test)
        print(f'Font {path.name} contains empty masks: ', empty_masks)
        print(f'PIL is able to print testing string: ', printable_test)

        if contains_test and printable_test and not empty_mask:
            copy2(str(path), str(out_dir / path.name))


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--fonts_dir', type=Path, required=True, help='Path to the directory with fonts.')
    parser.add_argument('--output_dir', type=Path, default='fonts',
                        help='Directory to save fonts which passed the test.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    fonts = list(args.fonts_dir.rglob('*.ttf'))
    fonts.extend(list(args.fonts_dir.rglob('*.TTF')))

    main(fonts, args.output_dir)
