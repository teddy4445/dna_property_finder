import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(7, 5), 'figure.dpi':100})

BREAK_NUMBER = 0
BREAK_LETTER = "*"

ltnm = {"A": 1, "T": 4, "G": 2, "C": 3, BREAK_LETTER: BREAK_NUMBER}
ntlm = {1: "A", 4: "T", 2: "G", 3: "C", BREAK_NUMBER: BREAK_LETTER}


def dna_letters_to_numbers(dna_string: str) -> list:
    return [ltnm[letter] for letter in dna_string.strip()]


def dna_numbers_to_letters(dna_numbers: list) -> str:
    return "".join([ntlm[number] for number in dna_numbers])


def shorten_focus(dna_number: list,
                  window: int = 30) -> list:
    break_point = -1
    for index, item in enumerate(dna_number):
        if item == BREAK_NUMBER:
            break_point = index
            break
    return dna_number[break_point - window:break_point + 3 + window] if break_point != -1 else ""


def load_data():
    with open("data.csv", "r") as data_file:
        dna_strings = [line for line_index, line in enumerate(data_file.readlines()) if len(line) > 10 and (line_index % 2) == 0]
    return dna_strings


def filter_duplicates(dna_numbers: list) -> list:
    answer = [dna_numbers[0]]
    answer.extend([dna_numbers[index]
                   for index in range(1, len(dna_numbers))
                   if reverse_swap_dna(dna_numbers[index]) != dna_numbers[index - 1] and dna_numbers[index] != ""])
    return answer


def reverse_swap_dna(dna: list) -> list:
    return [BREAK_NUMBER if item == BREAK_NUMBER else 5 - item for item in dna[::-1]]


def pattern_count(dna, pattern):
    return len([1 for i in range(len(dna)) if dna[i:i + len(pattern)] == pattern])


def cut_window(dna, location, window):
    return dna[location - window:location + 1 + window]


def frequent_words_best(dna,
                        min_length: int = 2,
                        max_length: int = 15):
    best_sequences = None
    best_count = 0
    for length_word in range(min_length, max_length + 1):
        sequences, count = frequent_words(dna, length_word)
        if best_count < count:
            best_count = count
            best_sequences = sequences
        # short cut
        if count == 0:
            break
    return best_sequences, best_count


def frequent_words(dna,
                   k):
    dna = dna.replace("*", "")
    pattern_lst = []            # list to store ALL patterns
    count = [0]*len(dna)       # initialize a list the length of the string for subequent iteration
    for i in range(len(dna)-k+1):
        pattern = dna[i:i + k]
        count[i] = pattern_count(dna, pattern)
        pattern_lst.append(pattern)

    patterncount = dict(zip(pattern_lst, count))               # zip pattern and count into a dict
    max_count = max(patterncount.items(), key=lambda x: x[1])  # returns all max values

    frequent_patterns = [key for key, value in patterncount.items() if value == max_count[1]] # list to store ONLY FREQUENT patterns
    return frequent_patterns, max_count[1]#, patterncount


def found_in_dna(dna,
                 sequence,
                 sequence_count):
    return len(sequence) * sequence_count / len(dna.replace("*", ""))


def string_representation_of_dna(dna: str,
                                 sequence: str) -> float:
    return found_in_dna(dna=dna,
                        sequence=sequence,
                        sequence_count=pattern_count(dna=dna,
                                                     pattern=sequence))


def generate_sequence_from_number(index: int) -> str:
    letters = ["A", "T", "C", "G"]
    seq = []
    # edge - case
    if index == 0:
        return 'A'
    while index != 0:
        seq.append(letters[int(index % 4)])
        index = int(index / 4)
    return ''.join(seq)


def find_sequences_in_data(data: list,
                           letters_to_create: int = 3):
    answer = {}
    for seq_index in range(4**letters_to_create):
        sequence = generate_sequence_from_number(seq_index)
        req_scores = [string_representation_of_dna(dna=dna,
                                                   sequence=sequence)
                      for dna in data]
        avg_req_score = sum(req_scores) / len(req_scores)
        answer[sequence] = avg_req_score
    return answer


def break_counts_per_size(counts: dict):
    sizes = {}
    counts_new_scores = {}
    for key, value in counts.items():
        size = len(key)
        if size not in sizes:
            sizes[size] = {}
        sizes[size][key] = value
        counts_new_scores[key] = value * size
    return sizes, counts_new_scores


def print_beautiful(dict_data: dict):
    [print("#{:05d} | {:^5}: {:.3f}\n".format(index, key, value), end='')
     for index, (key, value) in enumerate(dict_data.items())]


def plot_freq(values, size):
    plt.hist(values, bins=len(values))
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    plt.savefig("size_{}_histogram.png".format(size))
    plt.show()


def run():
    data = load_data()
    counts = find_sequences_in_data(data=data,
                                    letters_to_create=7)
    counts_per_size, counts_new_scores = break_counts_per_size(counts=counts)
    print(print_beautiful(counts_new_scores))

    for size in counts_per_size:
        size_counts = counts_per_size[size]
        best_key = None
        best_value = 0
        values_to_plot = []
        for key, value in size_counts.items():
            if value > best_value:
                best_value = value
                best_key = key
            values_to_plot.append(value)
        print("For size {}, the best sequence is: '{}' with score: {:.3f} and entropy = {:.3f}".format(size, best_key, best_value, best_value * size))
        plot_freq(values=values_to_plot,
                  size=size)

    """
    data = [dna_letters_to_numbers(dna_string=dna_string)
                   for dna_string in data]
    data = [shorten_focus(dna_number=dna_number) for dna_number in data]
    data = filter_duplicates(dna_numbers=data)
    """


if __name__ == '__main__':
    run()
