# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rouge_score import rouge_scorer
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
#nltk.download('all')

class MultiDocSummarizer:
    def __init__(self, num_topics=5, num_sentences=3):
        # Initialize the summarizer with default parameters
        self.num_topics = num_topics
        self.num_sentences = num_sentences
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.lsa = TruncatedSVD(n_components=self.num_sentences, random_state=42)
        self.lda_model = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def preprocess(self, text):
        """Preprocess the input text."""
        # Tokenize text into sentences and words
        sentences = sent_tokenize(text)
        tokens = [word_tokenize(sentence.lower()) for sentence in sentences]
        # Lemmatize words and remove stopwords
        tokens = [[self.lemmatizer.lemmatize(word) for word in sentence if word.isalnum() and word not in self.stop_words] for sentence in tokens]
        return tokens, sentences

    def extractive_summarization(self, documents):
        """Implement extractive summarization using TF-IDF and LSA."""
        # Preprocess documents
        preprocessed_docs, original_sentences = zip(*[self.preprocess(doc) for doc in documents])
        flat_docs = [' '.join(sent) for doc in preprocessed_docs for sent in doc]  # Join tokens into strings
        
        if not flat_docs:
            return "No sentences to summarize."

        # Apply TF-IDF vectorization
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(flat_docs)
        
        # Apply LSA (Latent Semantic Analysis)
        lsa_matrix = self.lsa.fit_transform(tfidf_matrix)
        
        # Select top sentences based on LSA scores
        scores = np.sum(lsa_matrix, axis=1)
        top_sentences_indices = scores.argsort()[-self.num_sentences:][::-1]
        
        # Construct summary from top sentences
        summary = []
        total_sentences = sum(len(doc) for doc in preprocessed_docs)
        
        for i in top_sentences_indices:
            if i < total_sentences:
                doc_index = 0
                while i >= len(preprocessed_docs[doc_index]):
                    i -= len(preprocessed_docs[doc_index])
                    doc_index += 1
                if doc_index < len(original_sentences) and i < len(original_sentences[doc_index]):
                    summary.append(original_sentences[doc_index][i])
        
        return ' '.join(summary) if summary else "Could not generate summary."

    def topic_modeling(self, documents):
        """Implement topic modeling using LDA."""
        # Preprocess documents
        preprocessed_docs, _ = zip(*[self.preprocess(doc) for doc in documents])
        flat_docs = [sent for doc in preprocessed_docs for sent in doc]
        
        # Create dictionary and corpus for LDA
        dictionary = corpora.Dictionary(flat_docs)
        corpus = [dictionary.doc2bow(doc) for doc in flat_docs]
        
        # Train LDA model
        self.lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=self.num_topics, random_state=42)
        
        return self.lda_model.print_topics()

    def integrate_components(self, documents):
        """Integrate extractive summarization with topic modeling."""
        # Generate summary and perform topic modeling
        summary = self.extractive_summarization(documents)
        topics = self.topic_modeling(documents)
        
        # Ensure summary aligns with identified topics
        summary_bow = self.lda_model.id2word.doc2bow(summary.lower().split())
        summary_topics = self.lda_model.get_document_topics(summary_bow)
        
        return summary, topics, summary_topics

    def evaluate(self, generated_summary, reference_summary):
        """Evaluate the generated summary using ROUGE metrics."""
        scores = self.rouge_scorer.score(reference_summary, generated_summary)
        return scores

def main():
    # Sample documents for summarization
    documents = [
    """Climate change is one of the most pressing challenges of our time. Scientists warn that the average global temperature could rise by 1.5 degrees Celsius within the next two decades if emissions are not curbed. This rise would result in severe consequences such as rising sea levels, extreme weather patterns, and loss of biodiversity. Governments worldwide are under pressure to transition to renewable energy sources and implement strict environmental policies. Public awareness campaigns emphasize the importance of reducing carbon footprints, adopting sustainable practices, and preserving natural habitats.""",
    
    """Artificial intelligence (AI) is transforming industries at an unprecedented pace. From healthcare to finance, AI-driven solutions are streamlining operations, enhancing decision-making, and driving innovation. However, experts caution against ethical concerns such as bias in algorithms, job displacement, and data privacy. Organizations are urged to adopt transparent practices and ethical guidelines while integrating AI technologies. Meanwhile, researchers continue to develop advanced AI systems capable of understanding natural language, making predictions, and even creating art.""",
    
    """Space exploration has entered a new era with private companies playing a significant role. SpaceX, Blue Origin, and other corporations are making space travel more accessible and cost-effective. Plans for lunar bases and manned missions to Mars are no longer confined to science fiction. Despite these advancements, critics argue that resources spent on space exploration could be redirected to solving Earth's pressing problems like poverty and climate change. Nevertheless, advocates believe that exploring the cosmos is essential for humanity's long-term survival and scientific progress."""
    ]       
    
    # Reference summary for evaluation
    reference_summary = """Climate change poses serious threats, including rising sea levels and biodiversity loss. Transitioning to renewable energy and reducing emissions are critical. Artificial intelligence is revolutionizing industries, but ethical concerns such as bias and job loss need attention. Private companies are advancing space exploration, with plans for Mars missions, though some argue that resources should focus on Earth's challenges."""
    
    # Create summarizer instance
    summarizer = MultiDocSummarizer()
    
    # Generate summary and perform topic modeling
    summary, topics, summary_topics = summarizer.integrate_components(documents)
    
    # Print results
    print("Generated Summary:")
    print(summary)
    print("\nIdentified Topics:")
    for topic in topics:
        print(topic)
    print("\nSummary Topics:")
    print(summary_topics)
    
    # Evaluate the summary using ROUGE metrics
    rouge_scores = summarizer.evaluate(summary, reference_summary)
    print("\nROUGE Scores:")
    for metric, score in rouge_scores.items():
        print(f"{metric}: {score}")

if __name__ == "__main__":
    main()
