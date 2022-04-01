from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import pandas as pd


def parse_args():
    parser = ArgumentParser(description='Automatic generation of random words.')
    parser.add_argument('--input_file', required=True, type=str, help='Path to file with input texts.')
    parser.add_argument('--output_file', required=True, type=str, help='Path to file to save generated data.')

    return parser.parse_args()


class RandomWordsGenerator:

    def __init__(self, input_file: Path):

        with open(input_file, "r", encoding="utf-8") as ifile:
            all_files = pd.DataFrame(map(lambda s: (s.split()[0], " ".join(s.split()[1:])),
                                          ifile.readlines()), columns=["file_name", "label"]).set_index("file_name")

        self.words = self.get_samples(all_files).to_numpy()

    @staticmethod
    def get_samples(all_files):
        words_grouped = all_files.groupby("label")
        word_frequency = pd.DataFrame(map(lambda group:
                                          {"label": group[0], "freq": len(group[1])},
                                          words_grouped)).sort_values("freq", ascending=False)

        return word_frequency

    def get_random_word(self):
        probs = self.words[:, 1] / np.sum(self.words[:, 1])
        word = np.random.choice(self.words[:, 0], p=probs.astype(np.float64))
        return word


def main(input_file, output_file):

    max_length = 102
    nr_lines = 10000

    word_generator = RandomWordsGenerator(input_file)

    with open(output_file, 'w') as ofile:
        for line_nr in range(nr_lines):

            line_length = np.random.randint(10, max_length)
            line = ''
            while True:
                word = word_generator.get_random_word()
                if len(line) + len(word) > line_length:
                    ofile.write(line + '\n')
                    break
                else:
                    line = line + ' ' + word


if __name__ == '__main__':
    arguments = parse_args()
    out_file = Path(arguments.output_file)
    in_file = Path(arguments.input_file)
    main(in_file, out_file)
