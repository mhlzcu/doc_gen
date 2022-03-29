from fontTools.ttLib import TTFont
from argparse import ArgumentParser
from pathlib import Path


def font_character_test(font_path: str, test_text: str) -> bool:
    contains_all = True
    font =TTFont(font_path)
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


def main(fonts):
    # path = './LITERPLA.ttf'
    for path in fonts:
        test = font_character_test(path, 'LOĎ ČEŘÍ KÝLEM TŮŇ OBZVLÁŠŤ V GRÓNSKÉ ÚŽINĚ. \
        PŘÍLIŠ ŽLUŤOUČKÝ KŮŇ ÚPĚL ĎÁBELSKÉ KÓDY. \
        Loď čeří kýlem tůň obzvlášť v Grónské úžině. \
        Příliš žluťoučký kůň úpěl ďábelské kódy.')
        print(f'Font {path.name} supports testing string: ', test)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--fonts_dir', required=True, help='Path to the directory with fonts')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    fonts = list(Path(args.fonts_dir).rglob('*.ttf'))
    main(fonts)
