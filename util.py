import sys
import numpy as np


def read_fasta_file(filename, length=None):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    for key, value in sequences.items():
        if 'genome' in key:
            sequence = np.array(translate_observations_to_one_hots(value))
        elif 'true-ann' in key:
            sequence = np.array(translate_annotation_to_indices(value))
    if length is None:
        return sequence
    else:
        return sequence[:length]


def progress_bar(progress, total):
    prog = int(round(40*progress / float(total)))
    bar = "{}{}".format("="*prog, "-"*(40 - prog))
    percents = round(100.0 * progress / float(total), 1)
    sys.stdout.write('|{}| {}%\r'.format(bar, percents, '%'))
    sys.stdout.flush()


def translate_observations_to_one_hots(obs):
    mapping = {'a': 1, 'c': 2, 'g': 3, 't': 4}
    obs = [mapping[symbol.lower()] for symbol in obs]
    return np.eye(5)[obs]


def translate_annotation_to_indices(path):
    mapping = {'n': 1, 'c': 2, 'r': 3}
    labels = [mapping[symbol.lower()] for symbol in path]
    return labels


def write_prediction_to_file(prediction, filename):
    f = open(filename, "w")
    pred_len = len(prediction)
    anno = ""
    for i in range(pred_len):
        if i % 60 == 0 and i != 0:
            anno = anno + "\n"
        anno = anno + str(prediction[i])
    f.write(anno)
    f.close()
