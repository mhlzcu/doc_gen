import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

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



def main():
    path = 'Melon_Font_-_Free_Trial.ttf'
    test_string = 'LOĎ ČEŘÍ KÝLEM TŮŇ OBZVLÁŠŤ V GRÓNSKÉ ÚŽINĚ. \
    PŘÍLIŠ ŽLUŤOUČKÝ KŮŇ ÚPĚL ĎÁBELSKÉ KÓDY. \
    Loď čeří kýlem tůň obzvlášť v Grónské úžině. \
    Příliš žluťoučký kůň úpěl ďábelské kódy.'
    contains_test = font_character_test(path, test_string)
    printable_test, empty_masks = pil_character_test(path, test_string)
    print('Font supports testing string: ', contains_test)
    print('Font contains empty masks: ', empty_masks)
    print('PIL is able to print testing string: ', printable_test)


main()