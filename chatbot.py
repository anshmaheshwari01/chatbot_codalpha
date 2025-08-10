import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class FAQChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        self.faqs = []
        self.answers = []
        
        self.load_faqs()
        self.train_model()
    
    def load_faqs(self):
        self.faqs = [
            "What is your return policy?",
            "How can I track my order?",
            "What payment methods do you accept?",
            "How do I contact customer support?",
            "Do you offer international shipping?",
            "What are your business hours?",
            "How can I reset my password?",
            "Where are your products made?",
            "What is your warranty policy?",
            "Can I change or cancel my order?"
        ]
        
        self.answers = [
            "Our return policy allows returns within 30 days of purchase with a receipt for a full refund.",
            "You can track your order by logging into your account and viewing your order history.",
            "We accept all major credit cards, PayPal, and Apple Pay.",
            "You can contact support via email at support@example.com or call us at 1-800-123-4567.",
            "Yes, we offer international shipping with rates calculated at checkout.",
            "Our customer service is available Monday to Friday from 9 AM to 5 PM EST.",
            "Visit the password reset page and enter your email to receive a reset link.",
            "Our products are manufactured in facilities that meet international quality standards.",
            "All products come with a 1-year limited warranty against manufacturing defects.",
            "You can change or cancel your order within 1 hour of placement from your order history."
        ]
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def train_model(self):
        processed_faqs = [self.preprocess_text(faq) for faq in self.faqs]
        self.vectorizer.fit(processed_faqs)
    
    def get_best_match(self, query):
        processed_query = self.preprocess_text(query)
        query_vec = self.vectorizer.transform([processed_query])
        processed_faqs = [self.preprocess_text(faq) for faq in self.faqs]
        faq_vecs = self.vectorizer.transform(processed_faqs)
        
        similarities = cosine_similarity(query_vec, faq_vecs)
        best_match_idx = np.argmax(similarities)
        score = similarities[0, best_match_idx]
        
        if score > 0.5:
            return self.answers[best_match_idx]
        else:
            return "I'm not sure I understand. Could you rephrase your question?"

# ---------------- Tkinter UI ---------------- #
class ChatUI:
    def __init__(self, root, bot):
        self.bot = bot
        root.title("FAQ Chatbot")
        root.geometry("500x500")
        
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', font=("Arial", 10))
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.entry = tk.Entry(root, font=("Arial", 10))
        self.entry.pack(padx=10, pady=5, fill=tk.X, side=tk.LEFT, expand=True)
        self.entry.bind("<Return>", self.send_message)
        
        self.send_btn = tk.Button(root, text="Send", command=self.send_message)
        self.send_btn.pack(padx=5, pady=5, side=tk.RIGHT)
        
        self.display_message("Bot", "Hello! Ask me about our products/services.")
    
    def display_message(self, sender, message):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, f"{sender}: {message}\n")
        self.chat_area.config(state='disabled')
        self.chat_area.yview(tk.END)
    
    def send_message(self, event=None):
        user_msg = self.entry.get().strip()
        if user_msg:
            self.display_message("You", user_msg)
            bot_response = self.bot.get_best_match(user_msg)
            self.display_message("Bot", bot_response)
            self.entry.delete(0, tk.END)

if __name__ == "__main__":
    chatbot = FAQChatbot()
    root = tk.Tk()
    ChatUI(root, chatbot)
    root.mainloop()
