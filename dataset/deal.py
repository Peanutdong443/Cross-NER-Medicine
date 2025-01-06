import jieba
from collections import Counter
with open('wenben.txt', 'r', encoding='utf-8') as file:
    corpus_text = file.read()
words = jieba.cut(corpus_text)
counts = Counter(words)
sorted_word_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
top_10_words = sorted_word_counts[:10]
for word, count in top_10_words:
    print(f'{word}: {count}')
