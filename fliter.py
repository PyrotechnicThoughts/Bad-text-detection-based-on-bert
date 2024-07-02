import os
import re
import csv

def remove(input_folder):
    result_dict = {}

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            sentences = split_sentences(text)
            filtered_lines = detect_bad_websites(sentences)
            filtered_lines = filter_numbers_and_episode_chapter(filtered_lines)
            merged_sentences = merge_short_sentences(filtered_lines)
            final_sentences = split_long_sentences(merged_sentences)
            filtered_lines = remove_whitespace(final_sentences)

            result_dict[filename] = filtered_lines

    return result_dict

def split_sentences(text):
    sentence_delimiters = '[。！？]'
    ignore_delimiters = r'[<>,\'\'“”{}\[\]()]+|(?<=《)[^》！]+(?=》)|(?<=<)[^>！]+(?=>)|(?<=“)[^”！]+(?=”)'

    sentences = re.split(sentence_delimiters + '(?!' + ignore_delimiters + ')', text)
    cleaned_sentences = [re.sub(r'[^\w\s]', '', sentence.strip().replace(' ', '')) for sentence in sentences if sentence.strip()]
    cleaned_text = '\n'.join(cleaned_sentences)
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    return cleaned_text.split('\n')

def filter_numbers_and_episode_chapter(sentences):
    filtered_sentences = []
    for sentence in sentences:
        if not re.match(r'第\d+集|第\d+章|第\d+期|第\d+页|更新至第\d+期|更新至第\d+集|更新至\d|\d+|[A-Za-z]+', sentence):
            filtered_sentences.append(sentence)
    return filtered_sentences

def merge_short_sentences(sentences):
    merged_sentences = []
    unmerged_sentences = sentences.copy()

    while unmerged_sentences:
        current_sentence = unmerged_sentences.pop(0)
        if len(current_sentence) < 15 and unmerged_sentences:
            next_sentence = unmerged_sentences.pop(0)
            merged_sentence = current_sentence + next_sentence
            if len(merged_sentence) >= 15:
                merged_sentences.append(merged_sentence)
            else:
                unmerged_sentences.insert(0, merged_sentence)
        else:
            merged_sentences.append(current_sentence)

    return merged_sentences

def split_long_sentences(sentences):
    split_sentences = []
    for sentence in sentences:
        while len(sentence) > 100:
            split_point = 100
            if u'\u4e00' <= sentence[split_point] <= u'\u9fff':
                split_point -= 1
            split_sentences.append(sentence[:split_point])
            sentence = sentence[split_point:]
        split_sentences.append(sentence)
    return split_sentences

def load_bad_words_from_csv(file_path):
    bad_words = []
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            bad_words.extend(row)
    return bad_words

def detect_bad_websites(unique_lines):
    bad_words = load_bad_words_from_csv('words.csv')
    retained_lines = []
    for text in unique_lines:
        text = text.lower()
        matched = False
        for word in bad_words:
            if re.search(r'\b' + re.escape(word) + r'\b', text, flags=re.IGNORECASE):
                matched = True
                break
        if not matched:
            retained_lines.append(text)
    return retained_lines

def remove_whitespace(sentences):
    return [''.join(sentence.split()) for sentence in sentences]

if __name__ == '__main__':
    input_folder = "bad_text_1"  # 文件夹名
    result_dict = remove(input_folder)
    for filename, content in result_dict.items():
        print(f"Text file: {filename}")
        print("Content:")
        for sentence in content:
            print(sentence)
        print("------------------")
