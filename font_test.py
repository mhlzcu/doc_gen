from fontTools.ttLib import TTFont
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


def main(font_list: List, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # path = './LITERPLA.ttf'
    for path in font_list:
        test = font_character_test(path, 'LOĎ ČEŘÍ KÝLEM TŮŇ OBZVLÁŠŤ V GRÓNSKÉ ÚŽINĚ. \
        PŘÍLIŠ ŽLUŤOUČKÝ KŮŇ ÚPĚL ĎÁBELSKÉ KÓDY. \
        Loď čeří kýlem tůň obzvlášť v Grónské úžině. \
        Příliš žluťoučký kůň úpěl ďábelské kódy.')
        print(f'Font {path.name} supports testing string: ', test)

        if test:
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
