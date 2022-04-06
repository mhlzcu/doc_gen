import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import label, regionprops


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
                print('Character missing:', character)
        if not contains_all:
            break

    return contains_all


def pil_character_test(font_path: str, test_text: str, connected_components_mask_limit: int = 10) -> (bool,
                                                                                                      bool, bool, bool):
    can_generate = True
    contains_empty_masks = False
    font = ImageFont.truetype(font_path, 10)
    repeating_masks = []
    contains_repeating_masks = False
    connected_components_check = True
    processed_characters = []
    for character in test_text:
        try:
            im = Image.new("RGB", (100, 100), (255, 255, 255))
            draw = ImageDraw.Draw(im)
            draw.text((50, 50), character, (0, 0, 0), font=font)
            im_np = np.array(im)
            if character not in processed_characters:
                con_comp_mask = im_np < 255
                regions = regionprops(label(con_comp_mask))
                if len(regions) > connected_components_mask_limit:
                    connected_components_check = False
                processed_characters.append(character)
                repeating_masks.append((im_np, character))
            if im_np.min() == 255 and character != ' ':
                contains_empty_masks = True
                print('Empty mask for ', character)
        except:
            can_generate = False
            return can_generate, contains_empty_masks

    for idx, arr in enumerate(repeating_masks):
        for mask in repeating_masks[(idx+1):]:
            same_val = np.array_equal(arr[0], mask[0])
            if same_val and arr[1] != mask[1]:
                contains_repeating_masks = True
                if arr[1] == ' ':
                    print('Same mask for symbols:', 'space', mask[1])
                elif mask[1] == ' ':
                    print('Same mask for symbols:', arr[1], 'space')
                else:
                    print('Same mask for symbols:', arr[1], mask[1])

    return can_generate, contains_empty_masks, contains_repeating_masks, connected_components_check



def main():
    path = 'Notera2Bold_PERSONAL.ttf'
    test_string = 'LOĎ ČEŘÍ KÝLEM TŮŇ OBZVLÁŠŤ V GRÓNSKÉ ÚŽINĚ. \n \
    PŘÍLIŠ ŽLUŤOUČKÝ KŮŇ ÚPĚL ĎÁBELSKÉ KÓDY. \n  \
    Loď čeří kýlem tůň obzvlášť v Grónské úžině. \n  \
    Příliš žluťoučký kůň úpěl ďábelské kódy.'
    contains_test = font_character_test(path, test_string.replace('\n', ''))
    printable_test, empty_masks, repeating_masks, components_check = pil_character_test(path,
                                                                                        test_string.replace('\n', ''))
    print('Font supports testing string: ', contains_test)
    print('Font contains empty masks: ', empty_masks)
    print('Font contains repeating masks: ', repeating_masks)
    print('Font passed connected components check : ', components_check)
    print('PIL is able to print testing string: ', printable_test)

    if printable_test:
        im = Image.new("RGB", (1000, 1000), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(path, 30)
        draw.multiline_text((50, 50), test_string, (0, 0, 0), font=font)
        im.show()

main()