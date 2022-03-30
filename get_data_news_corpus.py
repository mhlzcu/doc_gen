from pathlib import Path
from argparse import ArgumentParser
import numpy as np


def parse_args():
    parser = ArgumentParser(description='Process data from the news corpus.')
    parser.add_argument('--input_dir', required=True, type=str, help='Path to folder with input texts.')
    parser.add_argument('--output_dir', required=True, type=str, help='Path to folder to save generated data.')

    return parser.parse_args()


def main(input_dir, output_dir):

    max_length = 102
    nr_lines = 55000
    rows_counter = 0
    # iterate over document files
    with open(output_dir / 'news_corpus.txt', 'w') as w_file:
        for text_file in input_dir.iterdir():
            with open(text_file, 'r') as file:
                try:
                    for line in file.readlines():
                        text = line.strip()
                        # do not add empty lines
                        if len(text) < 1:
                            continue
                        rows = int(np.ceil(len(text) / max_length))
                        last_space = 0
                        for r in range(rows):
                            # get current substring
                            one_line = text[last_space:last_space + max_length]
                            # find last space in substring
                            result = one_line[::-1].find(' ')
                            space_position = len(one_line) - result
                            # cut text from the start to the last space
                            one_line = one_line[:space_position - 1]
                            # update position of last space
                            last_space = last_space + space_position
                            rows_counter += 1

                            # write text into the file
                            w_file.write(one_line + '\n')
                except UnicodeDecodeError as er:
                    print(er)
                    continue

            # break if number of lines exceeds limit
            if rows_counter > nr_lines:
                break


if __name__ == '__main__':
    arguments = parse_args()
    input_dir = Path(arguments.input_dir)
    output_dir = Path(arguments.output_dir)
    main(input_dir, output_dir)

