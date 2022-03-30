from pathlib import Path
from argparse import ArgumentParser
import numpy as np


characters = ['A', 'Á', 'B', 'C', 'Č', 'D', 'Ď', 'E', 'Ě', 'É', 'F', 'G', 'H', 'I', 'Í', 'J', 'K', 'L', 'M', 'N', 'Ň',
              'O', 'Ó', 'P', 'Q', 'R', 'Ř', 'S', 'Š', 'T', 'Ť', 'U', 'Ú', 'Ů', 'V', 'W', 'X', 'Y', 'Ý', 'Z', 'Ž', 'a',
              'á', 'b', 'c', 'č', 'd', 'ď', 'e', 'ě', 'é', 'f', 'g', 'h', 'i', 'í', 'j', 'k', 'l', 'm', 'n', 'ň', 'o',
              'ó', 'ö', 'p', 'q', 'r', 'ř', 's', 'š', 't', 'ť', 'u', 'ú', 'ů', 'ü', 'v', 'w', 'x', 'y', 'ý', 'z', 'ž']


def parse_args():
    parser = ArgumentParser(description='Automatic generation of random characters.')
    parser.add_argument('--output_dir', required=True, type=str, help='Path to folder with fonts.')

    return parser.parse_args()


def generate_random_word(mean, dev):
    """
    Generate word of random length consisting of random characters
    :param mean: mean value of normal distribution to generate the length of word
    :param dev: standard deviation of normal distribution to generate the length of word
    :return: randomly generated string
    """
    # generate random length of word
    length = np.random.normal(mean, scale=dev)
    # iterate again while length is >= 1
    while length < 1:
        length = np.random.normal(mean, scale=dev)
    # uniformly generate characters of random length
    chars = np.random.choice(characters, int(length + 0.5))
    # merge characters into one string
    word = ''.join(chars)

    return word


def main(output_dir):

    max_length = 102
    nr_lines = 1000

    with open(output_dir / 'random_characters.txt', 'w') as file:

        for line_nr in range(nr_lines):
            # random generation of line length with saturation to max_length
            line_length = np.random.normal(50, 20)
            line_length = line_length if line_length < max_length else max_length

            line = ''
            while True:
                word = generate_random_word(5, 3)

                if len(line) + len(word) > line_length:
                    file.write(line + '\n')
                    break
                else:
                    line = line + ' ' + word


if __name__ == '__main__':
    arguments = parse_args()
    output_dir = Path(arguments.output_dir)
    main(output_dir)


