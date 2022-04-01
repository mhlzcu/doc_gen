from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import pandas as pd


def parse_args():
    parser = ArgumentParser(description='Automatic generation of random words.')
    parser.add_argument('--input_files', required=True, nargs='+', type=str, help='Path to file with input texts.')
    parser.add_argument('--output_file', required=True, type=str, help='Path to file to save generated data.')

    return parser.parse_args()


def main(input_files, output_file):

    for input_file in input_files:
        with open(input_file, 'r') as ifile:
            with open(output_file, 'a') as ofile:
                for line in ifile.readlines():
                    file_name, text = line.split()[0], " ".join(line.split()[1:])
                    ofile.write(text + '\n')


if __name__ == '__main__':
    arguments = parse_args()
    out_file = Path(arguments.output_file)
    in_files = map(Path, arguments.input_files)
    main(in_files, out_file)

