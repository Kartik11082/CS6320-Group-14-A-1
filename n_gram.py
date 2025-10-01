import re
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import argparse


class NgramLanguageModel:

    def __init__(
        self,
        n: int = 2,
        smoothing_method: str = "laplace",
        k: float = 1.0,
        unk_threshold: int = 1,
        unk_method: str = "threshold",
    ):
        """
        Initialize the n-gram language model

        Args:
            n: Order of n-gram (1 for unigram, 2 for bigram, etc.)
            smoothing_method: 'laplace', 'add_k', or 'none'
            k: Smoothing parameter for add-k smoothing
            unk_threshold: Threshold for unknown word handling
            unk_method: Method for handling unknown words ('threshold', 'regex')
        """
        self.n = n
        self.smoothing_method = smoothing_method
        self.k = k
        self.unk_threshold = unk_threshold
        self.unk_method = unk_method

        # Storage for n-gram counts
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)

        # Vocabulary management
        self.vocabulary = set()
        self.word_counts = Counter()
        self.unk_token = "<UNK>"

        # Special tokens
        self.start_token = "<s>"
        self.end_token = "</s>"

    def preprocess_text(self, text: str) -> List[str]:
        # Convert to lowercase
        text = text.lower()

        tokens = text.split()

        processed_tokens = []
        for token in tokens:
            token = token.strip('.,!?;:"()[]{}')
            if token:
                processed_tokens.append(token)

        return processed_tokens

    def handle_unknown_words(self, tokens: List[str]) -> List[str]:
        if self.unk_method == "threshold":
            # Replace words that appear less than threshold times
            return [
                (
                    token
                    if self.word_counts[token] >= self.unk_threshold
                    else self.unk_token
                )
                for token in tokens
            ]

        elif self.unk_method == "regex":
            # Replace words matching certain patterns (numbers, rare patterns)
            processed_tokens = []
            for token in tokens:
                # Replace numbers with <UNK>
                if re.match(r"^\d+$", token):
                    processed_tokens.append(self.unk_token)
                # Replace very short or long words
                elif len(token) < 2 or len(token) > 20:
                    processed_tokens.append(self.unk_token)
                else:
                    processed_tokens.append(token)
            return processed_tokens

        return tokens

    def add_sentence_boundaries(self, tokens: List[str]) -> List[str]:
        # Add n-1 start tokens for n-gram context
        start_tokens = [self.start_token] * (self.n - 1)
        return start_tokens + tokens + [self.end_token]

    def get_ngrams(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            ngrams.append(ngram)
        return ngrams

    def train(self, corpus_file: str):
        print(f"Training {self.n}-gram model...")

        # First pass: count word frequencies for unknown word handling
        print("First pass: counting word frequencies...")
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens = self.preprocess_text(line.strip())
                self.word_counts.update(tokens)

        # Build vocabulary based on frequency threshold
        self.vocabulary = {
            word
            for word, count in self.word_counts.items()
            if count >= self.unk_threshold
        }
        self.vocabulary.add(self.unk_token)
        self.vocabulary.add(self.start_token)
        self.vocabulary.add(self.end_token)

        print(f"Vocabulary size: {len(self.vocabulary)}")

        # Second pass: collect n-gram counts
        print("Second pass: collecting n-gram counts...")
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens = self.preprocess_text(line.strip())
                tokens = self.handle_unknown_words(tokens)
                tokens = self.add_sentence_boundaries(tokens)

                # Extract n-grams and contexts
                ngrams = self.get_ngrams(tokens)

                for ngram in ngrams:
                    self.ngram_counts[ngram] += 1
                    if self.n > 1:
                        context = ngram[:-1]
                        self.context_counts[context] += 1

        print(f"Total n-grams collected: {len(self.ngram_counts)}")
        print(f"Total contexts collected: {len(self.context_counts)}")

    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        if self.n == 1:
            # Unigram probability
            word = ngram[0]
            word_count = self.ngram_counts[ngram]
            total_words = sum(self.ngram_counts.values())

            if self.smoothing_method == "laplace":
                return (word_count + 1) / (total_words + len(self.vocabulary))
            elif self.smoothing_method == "add_k":
                return (word_count + self.k) / (
                    total_words + self.k * len(self.vocabulary)
                )
            else:  # no smoothing
                return word_count / total_words if total_words > 0 else 0

        else:
            # N-gram probability (n > 1)
            context = ngram[:-1]
            ngram_count = self.ngram_counts[ngram]
            context_count = self.context_counts[context]

            if self.smoothing_method == "laplace":
                return (ngram_count + 1) / (context_count + len(self.vocabulary))
            elif self.smoothing_method == "add_k":
                return (ngram_count + self.k) / (
                    context_count + self.k * len(self.vocabulary)
                )
            else:  # no smoothing
                return ngram_count / context_count if context_count > 0 else 0

    def calculate_perplexity(self, test_file: str) -> float:
        print("Calculating perplexity...")

        total_log_prob = 0
        total_words = 0

        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens = self.preprocess_text(line.strip())
                tokens = self.handle_unknown_words(tokens)
                tokens = self.add_sentence_boundaries(tokens)

                ngrams = self.get_ngrams(tokens)

                for ngram in ngrams:
                    prob = self.get_probability(ngram)
                    if prob > 0:
                        total_log_prob += math.log(prob)
                    else:
                        total_log_prob += math.log(1e-10)  # Small epsilon
                    total_words += 1

        if total_words == 0:
            return float("inf")

        avg_log_prob = total_log_prob / total_words
        perplexity = math.exp(-avg_log_prob)

        return perplexity

    def generate_text(self, num_words: int = 50, seed: str = None) -> str:
        if self.n == 1:
            words = list(
                self.vocabulary - {self.start_token, self.end_token, self.unk_token}
            )
            return "Unigram text generation not implemented in this version"

        return "Text generation not fully implemented in this version"


def main():
    """
    Main function to run the n-gram language model
    """
    parser = argparse.ArgumentParser(description="N-gram Language Model")
    parser.add_argument("--train", required=True, help="Training corpus file")
    parser.add_argument("--validation", required=True, help="Validation corpus file")
    parser.add_argument("--n", type=int, default=2, help="N-gram order (default: 2)")
    parser.add_argument(
        "--smoothing",
        choices=["laplace", "add_k", "none"],
        default="laplace",
        help="Smoothing method",
    )
    parser.add_argument(
        "--k", type=float, default=1.0, help="Smoothing parameter for add-k"
    )
    parser.add_argument(
        "--unk_threshold",
        type=int,
        default=1,
        help="Threshold for unknown word handling",
    )
    parser.add_argument(
        "--unk_method",
        choices=["threshold", "regex"],
        default="threshold",
        help="Unknown word handling method",
    )

    args = parser.parse_args()

    # Create and train model
    model = NgramLanguageModel(
        n=args.n,
        smoothing_method=args.smoothing,
        k=args.k,
        unk_threshold=args.unk_threshold,
        unk_method=args.unk_method,
    )

    # Train the model
    model.train(args.train)

    # Calculate perplexity on validation set
    perplexity = model.calculate_perplexity(args.validation)
    print(f"\nPerplexity on validation set: {perplexity:.2f}")

    print("\nExample probability calculations:")

    if args.n == 1:
        sample_words = ["the", "hotel", "good", "<UNK>"]
        for word in sample_words:
            if word in model.vocabulary:
                prob = model.get_probability((word,))
                print(f"P({word}) = {prob:.6f}")

    elif args.n == 2:
        sample_bigrams = [
            ("the", "hotel"),
            ("hotel", "was"),
            ("was", "good"),
            ("<s>", "the"),
            ("good", "</s>"),
        ]
        for bigram in sample_bigrams:
            prob = model.get_probability(bigram)
            print(f"P({bigram[1]}|{bigram[0]}) = {prob:.6f}")


if __name__ == "__main__":
    if len(__import__("sys").argv) == 1:
        print("test mode...")
        models_to_test = [
            {"n": 1, "smoothing": "laplace", "name": "Unigram + Laplace"},
            {"n": 2, "smoothing": "laplace", "name": "Bigram + Laplace"},
            {"n": 2, "smoothing": "add_k", "k": 0.5, "name": "Bigram + Add-0.5"},
            {"n": 2, "smoothing": "none", "name": "Bigram + No Smoothing"},
        ]
    else:
        main()
