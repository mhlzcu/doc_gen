from document_generator import Document, Box, Text, Augmentation
from PIL import ImageFont, Image
import random
import os
import cv2
import csv
from multiprocessing import Pool
import numpy as np
from itertools import islice, cycle


def get_data(filepath: str, num_lines: int = 1, char_set: list[chr] = []) -> list[list[str]]:
    file_size = os.path.getsize(filepath)
    data_set = []
    with open(filepath, 'rb') as f:
        while len(data_set) < num_lines:
            pos = random.randint(0, file_size)
            if not pos:  # the first line is chosen
                break  # error
            f.seek(pos)  # seek to random position
            f.readline()  # skip possibly incomplete line
            line = f.readline()  # read next (full) line
            if line:
                line_text = line.decode()
                if char_set:
                    for symbol in line_text:
                        if symbol not in char_set:
                            continue
                words = line_text.replace('\n', '').replace('\r', '')
                data_set.append(words.split(' '))
            # else: line is empty -> EOF -> try another position in next iteration

    return data_set




text_path = r'/auto/plzen1/home/mhlavac/naki/data_ru/texts'
texts = [os.path.join(text_path, fn) for fn in next(os.walk(text_path))[2]]
while len(text_words) < 4:
    text = get_random_line(text_file).replace('\n', ' ').replace('\r', '')
    text_words = text.split(' ')