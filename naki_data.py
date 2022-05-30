import numpy as np

from document_generator import Document, Box, Text, Augmentation
from PIL import ImageFont, Image
import random
import os
import cv2
import csv
import albumentations as A


def get_random_line(filepath: str) -> str:
    file_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        while True:
            pos = random.randint(0, file_size)
            if not pos:  # the first line is chosen
                return f.readline().decode()  # return str
            f.seek(pos)  # seek to random position
            f.readline()  # skip possibly incomplete line
            line = f.readline()  # read next (full) line
            if line:
                return line.decode()
            # else: line is empty -> EOF -> try another position in next iteration


def main():
    line_height_min = 80
    line_height_max = 100
    line_width_min = 1
    line_width_max = 120
    line_heights = list(range(line_height_min, line_height_max + 1))
    line_widths = list(range(line_width_min, line_width_max))
    data_num = 15000
    font_size_start = 1
    font_folder_path = r'c:\Users\miros\OneDrive - Západočeská univerzita v Plzni\Práce\NAKI3\paper22\fonts'
    background_folder_path = r'c:\Users\miros\OneDrive - Západočeská univerzita v Plzni\Práce\NAKI3\paper22\background\use'
    text_path = r'c:\Users\miros\OneDrive - Západočeská univerzita v Plzni\Práce\NAKI3\paper22\texts'
    output_dir = r'd:\work\NAKI\data_fill'
    page_resolution = (1560, 2328)
    fonts = [os.path.join(font_folder_path, fn) for fn in next(os.walk(font_folder_path))[2]]
    backgrounds = [os.path.join(background_folder_path, fn) for fn in next(os.walk(background_folder_path))[2]]
    texts = [os.path.join(text_path, fn) for fn in next(os.walk(text_path))[2]]
    test_text = 'АаБбВвГгҐґДдЕеЄєЖжЗзИиІіЇїЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЬьЮюЯя'
    for i in range(data_num):
        y_page = 0
        page = Document(page_resolution)
        while y_page < page_resolution[1]:
            line_h = random.choice(line_heights)
            line_w = random.choice(line_widths)
            box = Box((page_resolution[0] - 40, line_h + 20))
            font_size = font_size_start
            text_file = random.choice(texts)
            text = get_random_line(text_file).replace('\n', ' ').replace('\r', '')
            to_print = text[0:line_w]
            # dummy = 'АаБбВвГгҐґДдЕеЄєЖжЗзИиІіЇїЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЬьЮюЯя..........................................................................................................'
            # dummy2 = 'вересня напад співробітників Управління держохорони на знімальну групу програми журналістських розслідувань «Схем» – журналіста Михайла Ткача та оператора Бориса Троценка, у якого діагностували струс мозк'
            # text = 'але бути хіба що дорожче'
            random_font = random.choice(fonts)
            font = ImageFont.truetype(random_font, font_size)
            while (font.getmetrics()[0] + font.getmetrics()[1]) < line_h:
                # iterate until the text size is just larger than the criteria
                font_size += 1
                font = ImageFont.truetype(random_font, font_size)
            # optionally de-increment to be sure it is less than criteria
            font_size -= 1
            print(font_size)
            font = ImageFont.truetype(random_font, font_size)
            # print(to_print, i, random_font)
            text_box = Text(to_print, font)
            augment = Augmentation()
            if random.random() > 0.1:
                augment.add_fonts([font], [text_box], add_offset=False)
            added = box.add_text(text_box, augment, max_lines=1, noise_percentage=random.uniform(0.3, 0.7))

            if added:
                if (y_page + line_h) + 20 >= page_resolution[1]:
                    break
                page.add_box(box, (20, y_page))

                y_page += line_h + 20
            else:
                print('Not added.')

        page.add_background(cv2.imread(random.choice(backgrounds)))
        file_name = str(i).rjust(len(str(data_num)), '0')
        page_out = page.image.convert("RGB")
        page_out.save(output_dir + '\\' + file_name + '.png')
        annotations_lukas = open(output_dir + '\\' + file_name + '_lines.csv', 'w+', encoding='utf8', newline='')
        annotations_petr = open(output_dir + '\\' + file_name + '_lines_bboxes.csv', 'w+', encoding='utf8', newline='')
        petr_writer = csv.writer(annotations_petr, delimiter=',')
        lukas_writer = csv.writer(annotations_lukas, delimiter=',')
        bb_list = page.get_lines_bounding_boxes()
        for idx_bbox, bbox in enumerate(bb_list):
            bbox_pil = (int(bbox[0][0])-random.randint(3, 15), int(bbox[0][1])-random.randint(3, 10), int(bbox[2][0]) +
                        random.randint(3, 15), int(bbox[2][1])+random.randint(3, 10))
            bbox_pil = tuple(np.array(bbox_pil).clip(0))
            line_crop = page.image.crop(bbox_pil)
            line_out = line_crop.convert("RGB")
            line_file_name = file_name + '_line_' + str(idx_bbox).rjust(len(str(len(bb_list))), '0') + '.jpg'
            line_out.save(output_dir + '\\' + line_file_name)
            petr_writer.writerow(
                [line_file_name, ' '.join(map(str, bbox_pil)), ''.join(page.boxes[idx_bbox].printed_text)])
            lukas_writer.writerow([line_file_name, ''.join(page.boxes[idx_bbox].printed_text)])

        annotations_lukas.close()
        annotations_petr.close()
        stop = 1
        # if i > 5:
        #     exit()
        print(f'\r{i}/' + str(data_num), end='', flush=True)
        break


if __name__ == "__main__":
    main()


