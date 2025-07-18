#!/usr/bin/env python
"""
@author: metalcorebear
@modified by honestus
"""

# Determine word affect based on the NRC emotional lexicon


from nltk.corpus import wordnet
from nltk.tokenize import TweetTokenizer, PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import csv



class AffectAnalyzer:
    def __init__(self, text, nrclex, tokenized=False, extract_phrases=True, max_phrase_length=3):
        self.nrclex = nrclex
        self.words = text if tokenized else self.__tokenize__(text)
        self.extract_phrases = extract_phrases
        self.max_phrase_length = max_phrase_length
        self.__is_analyzed__ = False
        self.affect_dict = None
        self.affect_frequencies = None
        self.affect_list = None
        self.raw_emotion_scores = None
        self.top_emotions = None
        
    def __tokenize__(self, text, tokenizer=TweetTokenizer()):
        wln = WordNetLemmatizer()
        return [wln.lemmatize(x, 'v') for x in tokenizer.tokenize(text)]

    def __extract_words_and_phrases__(self, max_phrase_length, longest_phrases_only=True):
        if self.__is_analyzed__:
            return self.words
        words_and_phrases = []
        n_words = len(self.words)
        i = 0

        while i < n_words:
            for span in range(min(self.max_phrase_length, n_words-i), 0, -1):
                phrase = ' '.join(self.words[i:i+span])

                if phrase in self.nrclex.__lexicon__:
                    words_and_phrases.append(phrase)
                    if longest_phrases_only:
                        break
            
            i=i+span if longest_phrases_only else i+1

        self.words = words_and_phrases
        
        
    def __build_word_affect__(self):
        '''
        Instantiates the following attributes:
            affect_list
            affect_dict
            raw_emotion_scores
            affect_frequencies
        '''
        affect_dict = dict()
        affect_frequencies = Counter()
        words_freq = Counter(self.words)
        for word, freq in words_freq.items():
            if word in self.nrclex.__lexicon__:
                emotions = self.nrclex.__lexicon__.get(word)
                affect_dict[word] = freq*emotions
                for emotion in emotions:
                    affect_frequencies[emotion]+=freq
        
        all_affect_attributes = self.nrclex._all_emotions_
        self.affect_dict = affect_dict
        self.affect_list = list(affect_frequencies.keys())
        self.raw_emotion_scores = dict(affect_frequencies)
        if len(affect_frequencies):
            sum_values = sum(affect_frequencies.values())
            affect_percent = {affect: float(affect_frequencies.get(affect, 0)) / float(sum_values) for affect in all_affect_attributes}
        else:
            affect_percent = {affect: 0.0 for affect in all_affect_attributes}
        self.affect_frequencies = affect_percent
        max_val = max(affect_percent.values())
        self.top_emotions = [(emo, val) for emo, val in affect_percent.items() if val == max_val]
        
        
    def analyze(self):
        if not self.__is_analyzed__:
            if self.extract_phrases:
                self.__extract_words_and_phrases__(self, longest_phrases_only=True)
            self.__build_word_affect__()
            self.__is_analyzed__ = True
        return self



def load(txt_file, colname = None):
    reader = csv.reader(txt_file, delimiter='\t')
    firstrow = next(reader)
    emotions = [
        'anger', 'anticipation', 'disgust', 'fear',
        'joy', 'negative', 'positive', 'sadness',
        'surprise', 'trust'
        ]
    lex = {}

    try:
        head = {
            **{x: firstrow.index(x) for x in emotions},
            **{x: firstrow.index(x) for x in firstrow if x not in emotions}
            }
    except ValueError:
        head = None

    if head:
        col = head[colname] if colname else head[firstrow[-1]]

        for row in reader:
            values = [x for x in emotions if row[head[x]] == '1']

            if values:
                lex[row[col]] = values
    else:
        if firstrow[2] == '1':
            lex[firstrow[0]] = [firstrow[1]]

        for row in reader:
            if row[2] == '1':
                if row[0] not in lex:
                    lex[row[0]] = []

                lex[row[0]].append(row[1])

    return lex






class NRCLex:
    """Lexicon source is (C) 2016 National Research Council Canada (NRC) and library is for research purposes only.  Source: http://sentiment.nrc.ca/lexicons-for-research/"""

    def __init__(self, lex_filename, colname = None, max_phrase_length=3, expand_lexicon=True):
        with open(lex_filename, 'r') as txt_file:
            self.__lexicon__ = load(txt_file, colname)
        self.max_phrase_length = max_phrase_length
        self.__expand__ = expand_lexicon
        #self._all_emotions_ = set([e for v in self.__lexicon__.values() for e in v ])
        self._all_emotions_ = [
        'anger', 'anticipation', 'disgust', 'fear',
        'joy', 'negative', 'positive', 'sadness',
        'surprise', 'trust'
        ]
        
    def __expand_lexicon__(self, words,):
        all_words = []
        n_words = len(words)

        for i in range(n_words):
            for span in range(min(self.max_phrase_length, n_words-i)):
                all_words.append(' '.join(words[i:i+span+1]))
                
        all_words = set(all_words)
        for w in all_words:
            if w in self.__lexicon__:
                continue

            found = False
            splits = w.split()
            use_antonyms = False

            if len(splits) > 1 and (splits[0] == 'not' or "n't" in splits[0]):
                use_antonyms = True
                context = '_'.join(splits[1:])
            else:
                context = w.replace(' ', '_')

            for syn in wordnet.synsets(context):
                for lemma in set(syn.lemmas()):
                    meaning = None

                    if use_antonyms:
                        antonyms = lemma.antonyms()

                        if antonyms:
                            meaning = antonyms[0].name().replace('_', ' ')
                    else:
                        meaning = lemma.name().replace('_', ' ')

                    if meaning in self.__lexicon__:
                        self.__lexicon__[i] = self.__lexicon__[meaning]
                        found = True
                        break

                if found:
                    break

            if found:
                continue

    def load_token_list(self, token_list, extract_phrases=True):
        '''
        Load an already tokenized text (as a list of tokens) into the NRCLex object.
        This is for when you want to use NRCLex with a text that you prefer to tokenize and/or lemmatize yourself.

        Parameters:
            token_list (list): a list of utf-8 strings.
        Returns:
            self
        '''
        analyzer = AffectAnalyzer(text=token_list, nrclex=self, tokenized=True, extract_phrases=extract_phrases, max_phrase_length=self.max_phrase_length)
        if self.__expand__:
            self.__expand_lexicon__(words=analyzer.words)
        
        analyzer = analyzer.analyze()
        return analyzer


    def load_raw_text(self, text, extract_phrases=True):
        '''
        Load a string into the NRCLex object for tokenization

        Parameters:
            text (str): a utf-8 string.
        Returns:
            self
        '''


        analyzer = AffectAnalyzer(text=text, nrclex=self, tokenized=False, extract_phrases=extract_phrases, max_phrase_length=self.max_phrase_length)
        if self.__expand__:
            self.__expand_lexicon__(words=analyzer.words)
        analyzer = analyzer.analyze()

        return analyzer



