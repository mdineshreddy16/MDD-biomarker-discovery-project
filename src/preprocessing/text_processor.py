"""
Text preprocessing module for MDD biomarker discovery.
Handles text cleaning, tokenization, and embedding preparation.
"""

import re
import string
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextProcessor:
    """
    Process text transcripts for depression biomarker analysis.
    Includes cleaning, normalization, and linguistic feature extraction.
    """
    
    def __init__(self,
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_stopwords: bool = False,
                 lemmatize: bool = True,
                 min_word_length: int = 2):
        """
        Initialize text processor with configuration.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_stopwords: Remove common stopwords
            lemmatize: Apply lemmatization
            min_word_length: Minimum word length to keep
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        
        # Initialize tools
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning operations.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def remove_filler_words(self, text: str) -> str:
        """
        Remove common filler words in speech (um, uh, like, you know).
        
        Args:
            text: Input text
            
        Returns:
            Text with filler words removed
        """
        filler_words = r'\b(um|uh|like|you know|i mean|sort of|kind of|basically)\b'
        text = re.sub(filler_words, '', text, flags=re.IGNORECASE)
        return ' '.join(text.split())
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def process_tokens(self, tokens: List[str]) -> List[str]:
        """
        Process token list (lowercase, remove stopwords, lemmatize).
        
        Args:
            tokens: List of tokens
            
        Returns:
            Processed token list
        """
        processed = []
        
        for token in tokens:
            # Lowercase
            if self.lowercase:
                token = token.lower()
            
            # Remove punctuation
            if self.remove_punctuation:
                token = token.translate(str.maketrans('', '', string.punctuation))
            
            # Skip if empty or too short
            if not token or len(token) < self.min_word_length:
                continue
            
            # Remove stopwords
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            # Lemmatize
            if self.lemmatize and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
            
            processed.append(token)
        
        return processed
    
    def get_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        return sent_tokenize(text)
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract basic linguistic features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of linguistic features
        """
        # Tokenization
        tokens = self.tokenize(text)
        sentences = self.get_sentences(text)
        
        # Word-level features
        word_count = len(tokens)
        char_count = len(text)
        avg_word_length = np.mean([len(w) for w in tokens]) if tokens else 0
        
        # Sentence-level features
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Lexical diversity
        unique_words = len(set(tokens))
        type_token_ratio = unique_words / word_count if word_count > 0 else 0
        
        # Count pronouns (depression indicator)
        first_person_pronouns = sum(1 for w in tokens if w.lower() in 
                                   ['i', 'me', 'my', 'mine', 'myself'])
        
        # Count negative words (basic sentiment)
        negative_words = sum(1 for w in tokens if w.lower() in 
                           ['not', 'no', 'never', 'nothing', 'nobody', 'none',
                            'sad', 'depressed', 'hopeless', 'worthless', 'tired'])
        
        # Punctuation features
        pause_markers = text.count(',') + text.count('...') + text.count('...')
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'unique_words': unique_words,
            'type_token_ratio': type_token_ratio,
            'first_person_pronouns': first_person_pronouns,
            'first_person_ratio': first_person_pronouns / word_count if word_count > 0 else 0,
            'negative_words': negative_words,
            'negative_ratio': negative_words / word_count if word_count > 0 else 0,
            'pause_markers': pause_markers,
            'question_marks': question_marks,
            'exclamation_marks': exclamation_marks
        }
    
    def preprocess_pipeline(self, 
                           text: str,
                           extract_features: bool = True) -> Dict:
        """
        Complete preprocessing pipeline for text.
        
        Args:
            text: Input text
            extract_features: Whether to extract linguistic features
            
        Returns:
            Dictionary containing processed text and features
        """
        # Clean text
        cleaned = self.clean_text(text)
        cleaned = self.remove_filler_words(cleaned)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        processed_tokens = self.process_tokens(tokens)
        
        # Join back to string
        processed_text = ' '.join(processed_tokens)
        
        result = {
            'original_text': text,
            'cleaned_text': cleaned,
            'processed_text': processed_text,
            'tokens': processed_tokens
        }
        
        # Extract features if requested
        if extract_features:
            result['features'] = self.extract_linguistic_features(cleaned)
        
        return result


class BatchTextProcessor:
    """
    Process multiple text files in batch.
    """
    
    def __init__(self, processor: TextProcessor):
        """
        Initialize batch processor.
        
        Args:
            processor: TextProcessor instance
        """
        self.processor = processor
    
    def process_dataframe(self,
                         df: pd.DataFrame,
                         text_column: str,
                         id_column: Optional[str] = None) -> pd.DataFrame:
        """
        Process text column in a dataframe.
        
        Args:
            df: Input dataframe
            text_column: Name of column containing text
            id_column: Name of ID column (optional)
            
        Returns:
            Dataframe with processed text and features
        """
        results = []
        
        print(f"Processing {len(df)} text entries...")
        
        for idx, row in df.iterrows():
            try:
                text = row[text_column]
                if pd.isna(text) or not text.strip():
                    continue
                
                # Process text
                processed = self.processor.preprocess_pipeline(text)
                
                # Create result row
                result_row = {
                    'processed_text': processed['processed_text'],
                    'token_count': len(processed['tokens'])
                }
                
                # Add ID if specified
                if id_column and id_column in row:
                    result_row['id'] = row[id_column]
                
                # Add features
                if 'features' in processed:
                    result_row.update(processed['features'])
                
                results.append(result_row)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        
        return pd.DataFrame(results)
    
    def process_directory(self,
                         input_dir: str,
                         file_pattern: str = "*.txt") -> Dict[str, Dict]:
        """
        Process all text files in a directory.
        
        Args:
            input_dir: Input directory path
            file_pattern: Glob pattern for text files
            
        Returns:
            Dictionary mapping filenames to processing results
        """
        input_path = Path(input_dir)
        text_files = list(input_path.glob(file_pattern))
        
        print(f"Found {len(text_files)} text files to process")
        
        results = {}
        
        for i, text_file in enumerate(text_files):
            try:
                print(f"Processing {i+1}/{len(text_files)}: {text_file.name}")
                
                # Read file
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Process
                processed = self.processor.preprocess_pipeline(text)
                
                results[text_file.stem] = processed
                
            except Exception as e:
                print(f"Error processing {text_file.name}: {e}")
                results[text_file.stem] = {'error': str(e)}
        
        return results


if __name__ == "__main__":
    # Example usage
    processor = TextProcessor(
        lowercase=True,
        remove_stopwords=False,
        lemmatize=True
    )
    
    # Test text
    sample_text = """
    I've been feeling really down lately. Like, I don't know, 
    nothing seems to matter anymore. I just feel so tired all the time.
    """
    
    result = processor.preprocess_pipeline(sample_text)
    print("Processed text:", result['processed_text'])
    print("\nLinguistic features:", result['features'])
    
    print("\nText processor module ready!")
