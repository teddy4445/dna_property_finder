import numpy as np
import pandas as pd

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


def run():
    data = load_data()
    data = [dna_letters_to_numbers(dna_string=dna_string)
                   for dna_string in data]
    data = [shorten_focus(dna_number=dna_number)
                   for dna_number in data]
    data = filter_duplicates(dna_numbers=data)
    print(len(data))


if __name__ == '__main__':
    run()
