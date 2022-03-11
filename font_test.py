from fontTools.ttLib import TTFont


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


def main():
    path = './LITERPLA.ttf'
    test = font_character_test(path, 'LOĎ ČEŘÍ KÝLEM TŮŇ OBZVLÁŠŤ V GRÓNSKÉ ÚŽINĚ. \
    PŘÍLIŠ ŽLUŤOUČKÝ KŮŇ ÚPĚL ĎÁBELSKÉ KÓDY. \
    Loď čeří kýlem tůň obzvlášť v Grónské úžině. \
    Příliš žluťoučký kůň úpěl ďábelské kódy.')
    print('Font supports testing string: ', test)


main()