"""
Text feature extraction module for depression biomarker discovery.
Extracts linguistic, emotional, and semantic features from text.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import re

# NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Sentiment analysis (optional)
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# BERT embeddings (optional)
try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


class TextFeatureExtractor:
    """
    Extract text features relevant to depression detection.
    """
    
    def __init__(self,
                 use_tfidf: bool = True,
                 max_features: int = 500,
                 embedding_model: Optional[str] = None):
        """
        Initialize text feature extractor.
        
        Args:
            use_tfidf: Whether to use TF-IDF features
            max_features: Maximum number of TF-IDF features
            embedding_model: Name of sentence embedding model (e.g., 'all-MiniLM-L6-v2')
        """
        self.use_tfidf = use_tfidf
        self.max_features = max_features
        
        # Initialize TF-IDF vectorizer
        if use_tfidf:
            self.tfidf = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
        else:
            self.tfidf = None
        
        # Initialize embedding model
        if embedding_model and BERT_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = None
    
    def extract_linguistic_features(self, text: str, tokens: List[str]) -> Dict[str, float]:
        """
        Extract linguistic and structural features.
        
        Args:
            text: Raw text
            tokens: Tokenized text
            
        Returns:
            Dictionary of linguistic features
        """
        # Basic counts
        word_count = len(tokens)
        char_count = len(text)
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Lexical features
        unique_words = len(set(tokens))
        type_token_ratio = unique_words / word_count if word_count > 0 else 0
        avg_word_length = np.mean([len(w) for w in tokens]) if tokens else 0
        
        # Sentence complexity
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Punctuation
        commas = text.count(',')
        periods = text.count('.')
        question_marks = text.count('?')
        exclamations = text.count('!')
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'unique_words': unique_words,
            'type_token_ratio': type_token_ratio,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'comma_count': commas,
            'period_count': periods,
            'question_count': question_marks,
            'exclamation_count': exclamations
        }
    
    def extract_emotional_features(self, text: str, tokens: List[str]) -> Dict[str, float]:
        """
        Extract emotion-related features.
        
        Args:
            text: Raw text
            tokens: Tokenized text
            
        Returns:
            Dictionary of emotional features
        """
        # Depression-related word lists
        negative_words = {
            'sad', 'depressed', 'hopeless', 'worthless', 'tired', 'exhausted',
            'lonely', 'empty', 'numb', 'guilty', 'anxious', 'worried', 'scared',
            'hurt', 'pain', 'suffering', 'dark', 'heavy', 'burden'
        }
        
        positive_words = {
            'happy', 'joy', 'excited', 'love', 'wonderful', 'amazing',
            'great', 'good', 'nice', 'beautiful', 'hope', 'blessed'
        }
        
        death_words = {
            'death', 'die', 'kill', 'suicide', 'end', 'gone', 'dead'
        }
        
        # Count emotion words
        tokens_lower = [t.lower() for t in tokens]
        word_count = len(tokens_lower)
        
        negative_count = sum(1 for w in tokens_lower if w in negative_words)
        positive_count = sum(1 for w in tokens_lower if w in positive_words)
        death_count = sum(1 for w in tokens_lower if w in death_words)
        
        # Pronoun usage (important for depression)
        first_person = sum(1 for w in tokens_lower if w in ['i', 'me', 'my', 'mine', 'myself'])
        second_person = sum(1 for w in tokens_lower if w in ['you', 'your', 'yours', 'yourself'])
        third_person = sum(1 for w in tokens_lower if w in 
                          ['he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers', 'their'])
        
        # Sentiment analysis (if available)
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity
        else:
            sentiment_polarity = 0.0
            sentiment_subjectivity = 0.0
        
        features = {
            'negative_word_count': negative_count,
            'positive_word_count': positive_count,
            'death_word_count': death_count,
            'negative_ratio': negative_count / word_count if word_count > 0 else 0,
            'positive_ratio': positive_count / word_count if word_count > 0 else 0,
            'first_person_count': first_person,
            'second_person_count': second_person,
            'third_person_count': third_person,
            'first_person_ratio': first_person / word_count if word_count > 0 else 0,
            'sentiment_polarity': sentiment_polarity,
            'sentiment_subjectivity': sentiment_subjectivity
        }
        
        return features
    
    def extract_cognitive_features(self, text: str, tokens: List[str]) -> Dict[str, float]:
        """
        Extract cognitive complexity features.
        
        Args:
            text: Raw text
            tokens: Tokenized text
            
        Returns:
            Dictionary of cognitive features
        """
        # Cognitive process words
        certainty_words = {'always', 'never', 'certain', 'definitely', 'sure', 'absolutely'}
        tentative_words = {'maybe', 'perhaps', 'might', 'could', 'possibly', 'probably'}
        causation_words = {'because', 'cause', 'reason', 'why', 'therefore', 'thus'}
        
        tokens_lower = [t.lower() for t in tokens]
        word_count = len(tokens_lower)
        
        certainty_count = sum(1 for w in tokens_lower if w in certainty_words)
        tentative_count = sum(1 for w in tokens_lower if w in tentative_words)
        causation_count = sum(1 for w in tokens_lower if w in causation_words)
        
        # Negation words
        negation_count = sum(1 for w in tokens_lower if w in 
                            {'not', 'no', 'never', "n't", 'neither', 'nobody', 'nothing'})
        
        return {
            'certainty_count': certainty_count,
            'tentative_count': tentative_count,
            'causation_count': causation_count,
            'negation_count': negation_count,
            'certainty_ratio': certainty_count / word_count if word_count > 0 else 0,
            'tentative_ratio': tentative_count / word_count if word_count > 0 else 0
        }
    
    def extract_speech_features(self, text: str) -> Dict[str, float]:
        """
        Extract features specific to spoken language transcripts.
        
        Args:
            text: Raw text
            
        Returns:
            Dictionary of speech features
        """
        # Filler words (um, uh, like, you know)
        filler_pattern = r'\b(um|uh|like|you know|i mean|sort of|kind of)\b'
        filler_matches = re.findall(filler_pattern, text.lower())
        filler_count = len(filler_matches)
        
        # Pause indicators
        pause_count = text.count('...') + text.count('..') + text.count('[pause]')
        
        # Repetition (simple detection)
        words = text.lower().split()
        repeated_words = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
        
        # Incomplete sentences (ends with -)
        incomplete_count = text.count('-')
        
        word_count = len(words)
        
        return {
            'filler_word_count': filler_count,
            'pause_count': pause_count,
            'repeated_word_count': repeated_words,
            'incomplete_sentence_count': incomplete_count,
            'filler_ratio': filler_count / word_count if word_count > 0 else 0
        }
    
    def extract_embeddings(self, text: str) -> Optional[np.ndarray]:
        """
        Extract semantic embeddings using pretrained model.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None
        """
        if self.embedding_model:
            return self.embedding_model.encode([text])[0]
        return None
    
    def extract_all_features(self, 
                            text: str, 
                            tokens: List[str]) -> Dict[str, float]:
        """
        Extract all text features.
        
        Args:
            text: Raw text
            tokens: Tokenized text
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Linguistic features
        linguistic = self.extract_linguistic_features(text, tokens)
        features.update(linguistic)
        
        # Emotional features
        emotional = self.extract_emotional_features(text, tokens)
        features.update(emotional)
        
        # Cognitive features
        cognitive = self.extract_cognitive_features(text, tokens)
        features.update(cognitive)
        
        # Speech features
        speech = self.extract_speech_features(text)
        features.update(speech)
        
        return features
    
    def fit_tfidf(self, texts: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on corpus.
        
        Args:
            texts: List of text documents
        """
        if self.tfidf:
            self.tfidf.fit(texts)
    
    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors.
        
        Args:
            texts: List of text documents
            
        Returns:
            TF-IDF matrix
        """
        if self.tfidf:
            return self.tfidf.transform(texts).toarray()
        return np.array([])
    
    def create_feature_dataframe(self, 
                                 texts: List[str],
                                 tokens_list: List[List[str]]) -> pd.DataFrame:
        """
        Create dataframe with all features for multiple texts.
        
        Args:
            texts: List of texts
            tokens_list: List of tokenized texts
            
        Returns:
            DataFrame with features
        """
        all_features = []
        
        for text, tokens in zip(texts, tokens_list):
            features = self.extract_all_features(text, tokens)
            
            # Add embeddings if available
            if self.embedding_model:
                embedding = self.extract_embeddings(text)
                for i, val in enumerate(embedding):
                    features[f'embedding_{i}'] = val
            
            all_features.append(features)
        
        df = pd.DataFrame(all_features)
        
        # Add TF-IDF features
        if self.use_tfidf and self.tfidf:
            tfidf_features = self.transform_tfidf(texts)
            tfidf_df = pd.DataFrame(
                tfidf_features,
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
            df = pd.concat([df, tfidf_df], axis=1)
        
        return df


if __name__ == "__main__":
    print("Text feature extraction module ready!")
    print("\nFeatures extracted:")
    print("- Linguistic (word count, TTR, sentence length)")
    print("- Emotional (negative/positive words, pronouns, sentiment)")
    print("- Cognitive (certainty, causation, negation)")
    print("- Speech (fillers, pauses, repetition)")
    print("- TF-IDF vectors (optional)")
    print("- BERT embeddings (optional)")
