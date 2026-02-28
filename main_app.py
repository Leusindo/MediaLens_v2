import os
import sys
import logging
import customtkinter as ctk
from tkinter import messagebox

from core.classifier import NewsClassifier
from core.news_collector import NewsCollector
from core.self_learning import SelfLearningSystem

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MediaLensApp:
    def __init__(self):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("MediaLens AI")
        self.root.geometry("1200x750")
        self.root.minsize(1100, 700)

        self.classifier = NewsClassifier()
        self.self_learning = None
        self.news_collector = None
        self.models_loaded = False

        self.category_display_names = {
            "clickbait": "Clickbait",
            "conspiracy": "Konšpirácia",
            "false_news": "Falošné správy",
            "propaganda": "Propaganda",
            "satire": "Satira",
            "misleading": "Zavádzajúce",
            "biased": "Zaujaté",
            "legitimate": "Dôveryhodné",
        }

        self.setup_logging()
        self.setup_layout()
        self.show_analysis_view()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MediaLens")

    def setup_layout(self):
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self.root, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(
            self.sidebar,
            text="MediaLens AI",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(pady=(30, 20))

        self.btn_analysis = ctk.CTkButton(
            self.sidebar, text="🔍 Analýza",
            fg_color="transparent", anchor="w",
            command=self.show_analysis_view
        )
        self.btn_analysis.pack(fill="x", padx=20, pady=8)

        self.btn_learning = ctk.CTkButton(
            self.sidebar, text="🧠 Self-Learning",
            fg_color="transparent", anchor="w",
            command=self.show_learning_view
        )
        self.btn_learning.pack(fill="x", padx=20, pady=8)

        self.btn_news = ctk.CTkButton(
            self.sidebar, text="📰 Zber správ",
            fg_color="transparent", anchor="w",
            command=self.show_news_view
        )
        self.btn_news.pack(fill="x", padx=20, pady=8)

        ctk.CTkButton(
            self.sidebar, text="⚡ Načítať modely",
            command=self.load_models
        ).pack(fill="x", padx=20, pady=(30, 10))

        # MAIN CONTENT
        self.main = ctk.CTkFrame(self.root, fg_color="#0f172a", corner_radius=15)
        self.main.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

    def clear_main(self):
        for widget in self.main.winfo_children():
            widget.destroy()

    def header(self, text):
        ctk.CTkLabel(
            self.main,
            text=text,
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(anchor="w", padx=30, pady=(25, 15))

    def card(self):
        frame = ctk.CTkFrame(self.main, fg_color="#1e293b", corner_radius=12)
        frame.pack(fill="both", expand=True, padx=30, pady=10)
        return frame

    def show_analysis_view(self):
        self.clear_main()
        self.header("Detekcia manipulácie")

        card = self.card()

        ctk.CTkLabel(card, text="Zadaj titulok článku:").pack(anchor="w", padx=20, pady=(20, 5))

        self.text_entry = ctk.CTkTextbox(card, height=80)
        self.text_entry.pack(fill="x", padx=20, pady=10)

        ctk.CTkButton(
            card,
            text="🚀 Spustiť analýzu",
            height=45,
            command=self.classify_text
        ).pack(fill="x", padx=20, pady=15)

        self.results = ctk.CTkTextbox(card)
        self.results.pack(fill="both", expand=True, padx=20, pady=(0, 20))


    def show_learning_view(self):
        self.clear_main()
        self.header("Self-Learning systém")

        card = self.card()

        self.learning_stats = ctk.CTkTextbox(card, height=200)
        self.learning_stats.pack(fill="x", padx=20, pady=20)

        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.pack(pady=10)

        ctk.CTkButton(btns, text="🔁 Pretrénovať", command=self.retrain_with_learning).pack(side="left", padx=10)
        ctk.CTkButton(btns, text="💾 Uložiť dáta", command=self.save_learning_data).pack(side="left", padx=10)
        ctk.CTkButton(btns, text="📊 Obnoviť", command=self.update_learning_stats).pack(side="left", padx=10)

    def show_news_view(self):
        self.clear_main()
        self.header("Zber a analýza správ")

        card = self.card()

        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.pack(pady=15)

        ctk.CTkButton(btns, text="⬇ Získať titulky", command=self.collect_news).pack(side="left", padx=10)
        ctk.CTkButton(btns, text="🤖 Auto-klasifikovať", command=self.auto_classify_news).pack(side="left", padx=10)

        self.news_box = ctk.CTkTextbox(card)
        self.news_box.pack(fill="both", expand=True, padx=20, pady=20)

    def load_models(self):
        try:
            self.classifier.load_models()
            self.self_learning = SelfLearningSystem(self.classifier)
            self.news_collector = NewsCollector(self.classifier)
            self.models_loaded = True
            messagebox.showinfo("OK", "Modely načítané 🧠")
            self.update_learning_stats()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def classify_text(self):
        if not self.models_loaded:
            messagebox.showwarning("Hold up", "Najprv načítaj modely")
            return

        text = self.text_entry.get("1.0", "end-1c").strip()
        if not text:
            return

        label, probs = self.classifier.predict(text)

        out = f"Titulok:\n{text}\n\nKategória: {self.category_display_names.get(label)}\n\n"
        for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            out += f"{k}: {v:.3f}\n"

        self.results.delete("1.0", "end")
        self.results.insert("1.0", out)

    def retrain_with_learning(self):
        if self.self_learning:
            self.self_learning.retrain_with_learning_data()
            self.update_learning_stats()

    def update_learning_stats(self):
        if not hasattr(self, "learning_stats"):
            return
        if not self.learning_stats.winfo_exists():
            return
        if not self.self_learning:
            return
        stats = self.self_learning.get_learning_stats()
        text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
        self.learning_stats.delete("1.0", "end")
        self.learning_stats.insert("1.0", text)

    def save_learning_data(self):
        if self.self_learning:
            self.self_learning._save_learning_data()

    def collect_news(self):
        news = self.news_collector.fetch_from_rss(limit_per_feed=5)
        self.news_box.delete("1.0", "end")
        for n in news:
            self.news_box.insert("end", f"{n['title']}\n{n['source']}\n\n")

    def auto_classify_news(self):
        items = self.news_collector.auto_classify_and_learn(self.self_learning)
        self.news_box.delete("1.0", "end")
        for i in items:
            self.news_box.insert("end", f"{i['title']} → {i['predicted_category']} ({i['confidence']:.2f})\n\n")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    MediaLensApp().run()
