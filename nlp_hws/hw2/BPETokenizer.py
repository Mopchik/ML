from tqdm import tqdm
from sortedcontainers import SortedSet

class BPETokenizer:
    def __init__(self, max_size=30000, min_word_count=10, optimal_token_length=5):
        self.ids2tokens = []
        self.tokens2ids = {}
        self.ids2tokens.append('__emp')
        self.ids2tokens.append('__sow')
        self.ids2tokens.append('__eow')
        self.tokens2ids['__emp'] = 0
        self.tokens2ids['__sow'] = 1
        self.tokens2ids['__eow'] = 2
        self.max_size = max_size
        self.min_word_count = min_word_count
        self.optimal_token_length = optimal_token_length

    def fit(self, texts):
        words_with_tokens = {}
        text_words = []
        text_words_count = []
        for text in texts:
            for word in text.strip().split():
                if word not in text_words:
                    text_words.append(word)
                    text_words_count.append(1)
                else:
                    text_words_count[text_words.index(word)] += 1
        for i in range(len(text_words)):
            for j in range(len(text_words[i])):
                c = text_words[i][j]
                if c not in self.tokens2ids:
                    index = len(self.ids2tokens)
                    self.tokens2ids[c] = index
                    self.ids2tokens.append(c)
                    words_with_tokens[c] = {}
                if i not in words_with_tokens[c]:
                    words_with_tokens[c][i] = [j]
                else:
                    words_with_tokens[c][i].append(j)

        cash_counts = SortedSet()
        for first in self.ids2tokens:
            for second in self.ids2tokens:
                if first == '__sow' or first == '__eow' or first == '__emp':
                    continue
                new_token = first + second
                new_count = 0
                words_with_new_token = []
                for index in words_with_tokens[first]:
                    word = text_words[index]
                    word_count = text_words_count[index]
                    if new_token in word:
                        new_count += word_count
                        words_with_new_token.append(index)
                if new_count >= self.min_word_count:
                    cash_counts.add((new_count, new_token))
                    words_with_tokens[new_token] = words_with_new_token
        _, max_token = cash_counts.pop()
        index = len(self.ids2tokens)
        self.tokens2ids[max_token] = index
        self.ids2tokens.append(max_token)
        last_added = max_token

        for _ in tqdm(range(len(self.tokens2ids), self.max_size)):
            for first in self.ids2tokens:
                second = last_added
                if first == '__sow' or first == '__eow' or first == '__emp':
                    continue
                new_token = first + second
                if new_token in self.tokens2ids:
                    continue
                new_count = 0
                words_with_new_token = []
                checking_token = first if len(words_with_tokens[first]) < len(words_with_tokens[second]) else second
                for index in words_with_tokens[checking_token]:
                    word = text_words[index]
                    word_count = text_words_count[index]
                    if new_token in word:
                        new_count += word_count
                        words_with_new_token.append(index)
                if new_count >= self.min_word_count:
                    if len(new_token) > self.optimal_token_length:
                        new_count //= len(new_token)
                    cash_counts.add((new_count, new_token))
                    words_with_tokens[new_token] = words_with_new_token
            for second in self.ids2tokens:
                first = last_added
                if second == '__sow' or second == '__eow' or second == '__emp':
                    continue
                new_token = first + second
                if new_token in self.tokens2ids:
                    continue
                new_count = 0
                words_with_new_token = []
                checking_token = first if len(words_with_tokens[first]) < len(words_with_tokens[second]) else second
                for index in words_with_tokens[checking_token]:
                    word = text_words[index]
                    word_count = text_words_count[index]
                    if new_token in word:
                        new_count += word_count
                        words_with_new_token.append(index)
                if new_count >= self.min_word_count:
                    if len(new_token) > self.optimal_token_length:
                        new_count //= len(new_token)
                    cash_counts.add((new_count, new_token))
                    words_with_tokens[new_token] = words_with_new_token

            if len(cash_counts) == 0:
                break
            max_count, max_token = cash_counts.pop()

            index = len(self.ids2tokens)
            self.tokens2ids[max_token] = index
            self.ids2tokens.append(max_token)
            last_added = max_token

    def encode(self, texts):
        all_input_ids = []
        all_attention_masks = []
        for text in texts:
            temp_input_ids = []
            words = text.split()
            for word in words:
                temp_input_ids += self.__tokenize_word(word)
            temp_attention_mask = [1] * len(temp_input_ids)
            all_input_ids.append(temp_input_ids)
            all_attention_masks.append(temp_attention_mask)

        max_len = max([len(sublist) for sublist in all_input_ids])
        all_input_ids = [sublist + [self.tokens2ids['__emp']] * (max_len - len(sublist)) for sublist in all_input_ids]
        all_attention_masks = [sublist + [0] * (max_len - len(sublist)) for sublist in all_attention_masks]
        return {'input_ids': all_input_ids, 'attention_mask': all_attention_masks}

    def __tokenize_word(self, word):
        if word in self.tokens2ids:
            return [self.tokens2ids[word]]
        ids = []
        ids.append(self.tokens2ids['__sow'])
        ids += self.__tokenize_recursive(word)
        ids.append(self.tokens2ids['__eow'])
        return ids

    def __tokenize_recursive(self, word):
        if word in self.tokens2ids:
            return [self.tokens2ids[word]]
        subseqLen = len(word)
        while word[:subseqLen] not in self.tokens2ids:
            subseqLen -= 1
            if subseqLen <= 0:
                return []
        res = []
        res.append(self.tokens2ids[word[:subseqLen]])
        res += self.__tokenize_recursive(word[subseqLen:])
        return res

    def decode(self, ids):
        res = ""
        isLongWord = False
        for id in ids:
            token = self.ids2tokens[id]
            if token == '__sow':
                isLongWord = True
            elif token == '__eow':
                isLongWord = False
                res += " "
            elif token == '__emp':
                continue
            elif isLongWord:
                res += token
            else:
                res += token + " "
        return res