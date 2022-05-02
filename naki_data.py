from document_generator import Document, Box, Text, Augmentation
from PIL import ImageFont, Image
import random
import os
import cv2
import csv


def main():
    line_height_min = 60
    line_height_max = 80
    line_width_min = 1
    line_width_max = 120
    line_heights = list(range(line_height_min, line_height_max+1))
    line_widths = list(range(line_width_min, line_width_max))
    data_num = 101425
    font_size_start = 1
    font_folder_path = r'c:\Users\miros\OneDrive - Západočeská univerzita v Plzni\Práce\NAKI3\paper22\fonts'
    background_folder_path = r'c:\Users\miros\OneDrive - Západočeská univerzita v Plzni\Práce\NAKI3\paper22\background\use'
    text_path = r'c:\Users\miros\OneDrive - Západočeská univerzita v Plzni\Práce\NAKI3\paper22\texts'
    output_dir = r'c:\Users\miros\OneDrive - Západočeská univerzita v Plzni\Práce\NAKI3\paper22\data_train'
    page_resolution = (1560, 2328)
    fonts = [os.path.join(font_folder_path, fn) for fn in next(os.walk(font_folder_path))[2]]
    backgrounds = [os.path.join(background_folder_path, fn) for fn in next(os.walk(background_folder_path))[2]]
    texts = [os.path.join(text_path, fn) for fn in next(os.walk(text_path))[2]]
    test_text = 'АаБбВвГгҐґДдЕеЄєЖжЗзИиІіЇїЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЬьЮюЯя'
    augment = Augmentation()
    for i in range(data_num):
        y_page = 0
        page = Document(page_resolution)
        while y_page < page_resolution[1]:
            line_h = random.choice(line_heights)
            line_w = random.choice(line_widths)
            box = Box((page_resolution[0] - 20, line_h + 20))
            font_size = font_size_start
            text = random.choice(texts)
            dummy2 = 'АаБбВвГгҐґДдЕеЄєЖжЗзИиІіЇїЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЬьЮюЯя..........................................................................................................'
            dummy = 'вересня напад співробітників Управління держохорони на знімальну групу програми журналістських розслідувань «Схем» – журналіста Михайла Ткача та оператора Бориса Троценка, у якого діагностували струс мозк'
            font = ImageFont.truetype(random.choice(fonts), font_size)
            while font.getsize(test_text)[1] < line_h:
                # iterate until the text size is just larger than the criteria
                font_size += 1
                font = ImageFont.truetype(random.choice(fonts), font_size)

            # optionally de-increment to be sure it is less than criteria
            font_size -= 1
            font = ImageFont.truetype(random.choice(fonts), font_size)
            to_print = dummy[0:line_w]
            text_box = Text(to_print, font)
            box.add_text(text_box, augment, max_lines=1)
            if (y_page + line_h) + 20 > page_resolution[1]:
                break
            page.add_box(box, (0, y_page))

            y_page += line_h + 20
            stop = 1
        page.add_background(cv2.imread(random.choice(backgrounds)))
        file_name = str(i).rjust(len(str(data_num)), '0')
        page_out = page.image.convert("RGB")
        page_out.save(output_dir + '\\' + file_name + '.jpg', "JPEG", quality=95, optimize=True, progressive=True)
        annotations_lukas = open(output_dir + '\\' + file_name + '_lines.csv', 'w+', encoding='utf8', newline='')
        annotations_petr = open(output_dir + '\\' + file_name + '_lines_bboxes.csv', 'w+', encoding='utf8', newline='')
        petr_writer = csv.writer(annotations_petr, delimiter=',')
        lukas_writer = csv.writer(annotations_lukas, delimiter=',')
        bb_list = page.get_lines_bounding_boxes()
        for idx_bbox, bbox in enumerate(bb_list):
            bbox_pil = (int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1]))
            line_crop = page.image.crop(bbox_pil)
            line_out = line_crop.convert("RGB")
            line_file_name = file_name + '_line_' + str(idx_bbox).rjust(len(str(len(bb_list))), '0') + '.jpg'
            line_out.save(output_dir + '\\' + line_file_name)
            petr_writer.writerow([line_file_name, ' '.join(map(str,bbox_pil)), ''.join(page.boxes[idx_bbox].printed_text)])
            lukas_writer.writerow([line_file_name, ''.join(page.boxes[idx_bbox].printed_text)])

        annotations_lukas.close()
        annotations_petr.close()


if __name__ == "__main__":
    main()


