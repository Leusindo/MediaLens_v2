import logging
import os
import sys
import threading

import customtkinter as ctk
import pandas as pd
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
        self.root.geometry("1280x720")
        self.root.minsize(1180, 760)

        self.classifier = NewsClassifier()
        self.self_learning = None
        self.news_collector = None
        self.models_loaded = False

        self.busy_lock = threading.Lock()
        self.is_busy = False

        self.raw_data_path = os.path.join("data", "raw", "training_data.csv")
        self.self_learning_path = os.path.join("data", "self_learning", "learning_data.csv")

        self.category_display_names = {
            "clickbait": "Clickbait",
            "conspiracy": "Konspiracia",
            "false_news": "Falosne spravy",
            "propaganda": "Propaganda",
            "satire": "Satira",
            "misleading": "Zavadzajuce",
            "biased": "Zaujate",
            "legitimate": "Doveryhodne",
        }

        self.training_model_var = ctk.StringVar(value="rf")
        self.training_aug_var = ctk.BooleanVar(value=True)
        self.train_raw_var = ctk.BooleanVar(value=True)
        self.train_self_learning_var = ctk.BooleanVar(value=True)

        self.setup_logging()
        self.setup_layout()
        self.show_analysis_view()
        self.try_auto_load_models()

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
            self.sidebar,
            text="Analyza",
            fg_color="transparent",
            anchor="w",
            command=self.show_analysis_view
        )
        self.btn_analysis.pack(fill="x", padx=20, pady=8)

        self.btn_training_view = ctk.CTkButton(
            self.sidebar,
            text="Trenovanie",
            fg_color="transparent",
            anchor="w",
            command=self.show_training_view
        )
        self.btn_training_view.pack(fill="x", padx=20, pady=8)

        self.btn_learning = ctk.CTkButton(
            self.sidebar,
            text="Self-Learning",
            fg_color="transparent",
            anchor="w",
            command=self.show_learning_view
        )
        self.btn_learning.pack(fill="x", padx=20, pady=8)

        self.btn_news = ctk.CTkButton(
            self.sidebar,
            text="Zber sprav",
            fg_color="transparent",
            anchor="w",
            command=self.show_news_view
        )
        self.btn_news.pack(fill="x", padx=20, pady=8)

        self.btn_load_models = ctk.CTkButton(
            self.sidebar,
            text="Nacitat modely",
            command=self.load_models
        )
        self.btn_load_models.pack(fill="x", padx=20, pady=(30, 10))

        self.main = ctk.CTkFrame(self.root, fg_color="#0f172a", corner_radius=15)
        self.main.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.status_var = ctk.StringVar(value="Pripraveny.")
        self.status_label = ctk.CTkLabel(self.root, textvariable=self.status_var, anchor="w")
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 10))

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

    def set_status(self, text):
        self.status_var.set(text)

    def _set_busy(self, busy, status_text=None):
        self.is_busy = busy
        state = "disabled" if busy else "normal"
        for button in (getattr(self, "btn_load_models", None),):
            if button is not None and button.winfo_exists():
                button.configure(state=state)
        if status_text is not None:
            self.set_status(status_text)

    def _run_in_background(self, target, on_success=None, on_error=None, busy_text="Pracujem..."):
        with self.busy_lock:
            if self.is_busy:
                messagebox.showwarning("Info", "Pockaj na dokoncenie aktualnej ulohy.")
                return False
            self.is_busy = True
            self.root.after(0, lambda: self._set_busy(True, busy_text))

        def worker():
            try:
                result = target()
            except Exception as exc:
                self.logger.exception("Background task failed")

                def handle_error():
                    self._set_busy(False, "Operacia zlyhala.")
                    if on_error:
                        on_error(exc)
                    else:
                        messagebox.showerror("Error", str(exc))

                self.root.after(0, handle_error)
                return

            def handle_success():
                self._set_busy(False, "Pripraveny.")
                if on_success:
                    on_success(result)

            self.root.after(0, handle_success)

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _load_raw_training_df(self):
        try:
            df = pd.read_csv(self.raw_data_path)
        except FileNotFoundError:
            return pd.DataFrame(columns=["title", "category"])
        return df.dropna(subset=["title", "category"]).copy()

    def _load_self_learning_training_df(self):
        try:
            df = pd.read_csv(self.self_learning_path)
        except FileNotFoundError:
            return pd.DataFrame(columns=["title", "category"])

        if df.empty:
            return pd.DataFrame(columns=["title", "category"])

        if "text" not in df.columns or "category" not in df.columns:
            return pd.DataFrame(columns=["title", "category"])

        filtered = df.dropna(subset=["text", "category"]).copy()
        filtered = filtered[filtered["category"].astype(str).str.strip() != ""]
        filtered = filtered.rename(columns={"text": "title"})
        return filtered[["title", "category"]]

    def _build_training_dataset(self):
        selected_frames = []
        sources = []

        if self.train_raw_var.get():
            raw_df = self._load_raw_training_df()
            if not raw_df.empty:
                selected_frames.append(raw_df[["title", "category"]].copy())
                sources.append("raw")

        if self.train_self_learning_var.get():
            learning_df = self._load_self_learning_training_df()
            if not learning_df.empty:
                selected_frames.append(learning_df[["title", "category"]].copy())
                sources.append("self-learning")

        if not selected_frames:
            raise ValueError("Vyber aspon jeden dataset s dostupnymi datami.")

        dataset = pd.concat(selected_frames, ignore_index=True)
        dataset = dataset.dropna(subset=["title", "category"])
        dataset["title"] = dataset["title"].astype(str).str.strip()
        dataset["category"] = dataset["category"].astype(str).str.strip().str.lower()
        dataset = dataset[(dataset["title"] != "") & (dataset["category"] != "")]
        dataset = dataset.drop_duplicates(subset=["title", "category"])

        if dataset.empty:
            raise ValueError("Vybrane datasety neobsahuju ziadne validne treningove zaznamy.")

        return dataset, sources

    def _refresh_training_preview(self):
        raw_df = self._load_raw_training_df()
        learning_df = self._load_self_learning_training_df()

        if hasattr(self, "raw_summary_var"):
            self.raw_summary_var.set(self._dataset_summary_text("Raw dataset", raw_df))
        if hasattr(self, "self_learning_summary_var"):
            self.self_learning_summary_var.set(self._dataset_summary_text("Self-learning", learning_df))

        if hasattr(self, "raw_titles_box") and self.raw_titles_box.winfo_exists():
            self._fill_titles_box(self.raw_titles_box, raw_df, "title")
        if hasattr(self, "self_learning_titles_box") and self.self_learning_titles_box.winfo_exists():
            self._fill_titles_box(self.self_learning_titles_box, learning_df, "title")

    def _dataset_summary_text(self, label, df):
        count = len(df)
        categories = df["category"].value_counts().to_dict() if not df.empty and "category" in df.columns else {}
        category_text = ", ".join([f"{key}: {value}" for key, value in categories.items()]) or "ziadne"
        return f"{label}: {count} zaznamov | Kategorie: {category_text}"

    def _fill_titles_box(self, box, df, title_column):
        box.delete("1.0", "end")
        if df.empty:
            box.insert("1.0", "Ziadne data.")
            return

        lines = []
        for _, row in df.iterrows():
            lines.append(f"[{row['category']}] {row[title_column]}")
        box.insert("1.0", "\n".join(lines))

    def show_analysis_view(self):
        self.clear_main()
        self.header("Detekcia manipulacie")

        card = self.card()
        ctk.CTkLabel(card, text="Zadaj titulok clanku:").pack(anchor="w", padx=20, pady=(20, 5))

        self.text_entry = ctk.CTkTextbox(card, height=80)
        self.text_entry.pack(fill="x", padx=20, pady=10)

        self.btn_classify = ctk.CTkButton(card, text="Spustit analyzu", height=45, command=self.classify_text)
        self.btn_classify.pack(fill="x", padx=20, pady=15)

        self.results = ctk.CTkTextbox(card)
        self.results.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def show_training_view(self):
        self.clear_main()
        self.header("Trenovanie modelu")

        card = self.card()

        controls = ctk.CTkFrame(card, fg_color="transparent")
        controls.pack(fill="x", padx=20, pady=(20, 10))

        dataset_box = ctk.CTkFrame(controls)
        dataset_box.pack(side="left", fill="y", padx=(0, 15))
        ctk.CTkLabel(dataset_box, text="Datasety", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(12, 8))
        ctk.CTkCheckBox(dataset_box, text="Raw training data", variable=self.train_raw_var).pack(anchor="w", padx=15, pady=4)
        ctk.CTkCheckBox(dataset_box, text="Self-learning data", variable=self.train_self_learning_var).pack(anchor="w", padx=15, pady=(4, 12))

        model_box = ctk.CTkFrame(controls)
        model_box.pack(side="left", fill="y", padx=(0, 15))
        ctk.CTkLabel(model_box, text="Model", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(12, 8))
        ctk.CTkRadioButton(model_box, text="Random Forest (rf)", value="rf", variable=self.training_model_var).pack(anchor="w", padx=15, pady=4)
        ctk.CTkRadioButton(model_box, text="MLP (mlp)", value="mlp", variable=self.training_model_var).pack(anchor="w", padx=15, pady=(4, 12))

        actions_box = ctk.CTkFrame(controls)
        actions_box.pack(side="left", fill="both", expand=True)
        ctk.CTkLabel(actions_box, text="Nastavenia", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(12, 8))
        self.training_aug_checkbox = ctk.CTkCheckBox(
            actions_box,
            text="Pouzit augmentaciu dat",
            variable=self.training_aug_var,
            onvalue=True,
            offvalue=False
        )
        self.training_aug_checkbox.pack(anchor="w", padx=15, pady=4)

        actions_row = ctk.CTkFrame(actions_box, fg_color="transparent")
        actions_row.pack(fill="x", padx=15, pady=(10, 12))
        self.btn_refresh_training_data = ctk.CTkButton(actions_row, text="Obnovit data", command=self._refresh_training_preview)
        self.btn_refresh_training_data.pack(side="left", padx=(0, 10))
        self.btn_train_model = ctk.CTkButton(actions_row, text="Spustit trenovanie", command=self.train_model)
        self.btn_train_model.pack(side="left")

        summaries = ctk.CTkFrame(card, fg_color="transparent")
        summaries.pack(fill="x", padx=20, pady=(0, 10))
        self.raw_summary_var = ctk.StringVar(value="")
        self.self_learning_summary_var = ctk.StringVar(value="")
        ctk.CTkLabel(summaries, textvariable=self.raw_summary_var, anchor="w").pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(summaries, textvariable=self.self_learning_summary_var, anchor="w").pack(fill="x")

        preview_frame = ctk.CTkFrame(card, fg_color="transparent")
        preview_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(1, weight=1)
        preview_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(preview_frame, text="Raw dataset titles", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=(0, 10), pady=(0, 8))
        ctk.CTkLabel(preview_frame, text="Self-learning titles", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1, sticky="w", padx=(10, 0), pady=(0, 8))

        self.raw_titles_box = ctk.CTkTextbox(preview_frame)
        self.raw_titles_box.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        self.self_learning_titles_box = ctk.CTkTextbox(preview_frame)
        self.self_learning_titles_box.grid(row=1, column=1, sticky="nsew", padx=(10, 0))

        ctk.CTkLabel(card, text="Vysledky trenovania", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(10, 8))
        self.training_results = ctk.CTkTextbox(card, height=220)
        self.training_results.pack(fill="x", padx=20, pady=(0, 20))
        self.training_results.insert(
            "1.0",
            "Vyber dataset(y), model a spusti trenovanie.\n"
            "Logger bude dalej vypisovat detailny priebeh do terminalu."
        )

        self._refresh_training_preview()

    def show_learning_view(self):
        self.clear_main()
        self.header("Self-Learning system")

        card = self.card()
        self.learning_stats = ctk.CTkTextbox(card, height=200)
        self.learning_stats.pack(fill="x", padx=20, pady=20)

        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.pack(pady=10)
        self.btn_retrain = ctk.CTkButton(btns, text="Pretrenovat", command=self.retrain_with_learning)
        self.btn_retrain.pack(side="left", padx=10)
        self.btn_save_learning = ctk.CTkButton(btns, text="Ulozit data", command=self.save_learning_data)
        self.btn_save_learning.pack(side="left", padx=10)
        self.btn_refresh_learning = ctk.CTkButton(btns, text="Obnovit", command=self.update_learning_stats)
        self.btn_refresh_learning.pack(side="left", padx=10)

        self.update_learning_stats()

    def show_news_view(self):
        self.clear_main()
        self.header("Zber a analyza sprav")

        card = self.card()
        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.pack(pady=15)

        self.btn_collect_news = ctk.CTkButton(btns, text="Ziskat titulky", command=self.collect_news)
        self.btn_collect_news.pack(side="left", padx=10)
        self.btn_auto_classify = ctk.CTkButton(btns, text="Auto-klasifikovat", command=self.auto_classify_news)
        self.btn_auto_classify.pack(side="left", padx=10)

        self.news_box = ctk.CTkTextbox(card)
        self.news_box.pack(fill="both", expand=True, padx=20, pady=20)

    def try_auto_load_models(self):
        self.load_models(show_message=False)

    def load_models(self, show_message=True):
        def do_load():
            loaded = self.classifier.load_models()
            if not loaded:
                raise RuntimeError("Modely sa nepodarilo nacitat.")

            self.self_learning = SelfLearningSystem(self.classifier)
            self.news_collector = NewsCollector(self.classifier)
            self.models_loaded = True
            return True

        def on_success(_):
            self.update_learning_stats()
            if show_message:
                messagebox.showinfo("OK", "Modely nacitane.")

        def on_error(exc):
            self.models_loaded = False
            if show_message:
                messagebox.showerror("Error", str(exc))

        self._run_in_background(do_load, on_success=on_success, on_error=on_error, busy_text="Nacitavam modely...")

    def train_model(self):
        model_type = self.training_model_var.get().strip().lower()
        enable_augmentation = bool(self.training_aug_var.get())

        if model_type not in {"rf", "mlp"}:
            messagebox.showwarning("Info", "Vyber platny typ modelu.")
            return

        try:
            dataset, sources = self._build_training_dataset()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        if hasattr(self, "training_results") and self.training_results.winfo_exists():
            self.training_results.delete("1.0", "end")
            self.training_results.insert(
                "1.0",
                f"Spustam trenovanie modelu '{model_type}'\n"
                f"Datasety: {', '.join(sources)}\n"
                f"Zaznamy: {len(dataset)}\n"
                f"Augmentacia: {'ano' if enable_augmentation else 'nie'}\n\n"
                "Detailny priebeh sleduj v terminali."
            )

        def do_train():
            classifier = NewsClassifier(model_type=model_type)
            original_data = pd.read_csv(classifier.config.DATA_PATH)
            try:
                dataset.to_csv(classifier.config.DATA_PATH, index=False)
                results = classifier.train(enable_augmentation=enable_augmentation)
            finally:
                original_data.to_csv(classifier.config.DATA_PATH, index=False)

            if not classifier.load_models(model_type=model_type):
                raise RuntimeError("Trenovanie skoncilo, ale novy model sa nepodarilo nacitat.")

            self.classifier = classifier
            self.self_learning = SelfLearningSystem(self.classifier)
            self.news_collector = NewsCollector(self.classifier)
            self.models_loaded = True
            return {
                "results": results,
                "sources": sources,
                "dataset_size": len(dataset),
                "model_type": model_type,
                "augmentation": enable_augmentation,
            }

        def on_success(payload):
            summary = self._format_training_results(payload)
            if hasattr(self, "training_results") and self.training_results.winfo_exists():
                self.training_results.delete("1.0", "end")
                self.training_results.insert("1.0", summary)
            self.update_learning_stats()
            self._refresh_training_preview()
            messagebox.showinfo("OK", "Trenovanie bolo dokoncene a model je nacitany v aplikacii.")

        self._run_in_background(
            do_train,
            on_success=on_success,
            busy_text=f"Trenujem model {model_type}..."
        )

    def _format_training_results(self, payload):
        results = payload["results"]
        report = results.get("classification_report", {})
        macro_avg = report.get("macro avg", {})
        weighted_avg = report.get("weighted avg", {})

        lines = [
            f"Model: {payload['model_type']}",
            f"Datasety: {', '.join(payload['sources'])}",
            f"Pocet zaznamov: {payload['dataset_size']}",
            f"Augmentacia: {'ano' if payload['augmentation'] else 'nie'}",
            "",
            f"Accuracy: {results.get('accuracy', 0):.4f}",
            f"Average confidence: {results.get('average_confidence', 0):.4f}",
            f"Test set size: {results.get('test_set_size', 0)}",
            f"Feature dimensions: {results.get('feature_dimensions', 0)}",
            "",
            "Macro avg:",
            f"precision={macro_avg.get('precision', 0):.4f} recall={macro_avg.get('recall', 0):.4f} f1={macro_avg.get('f1-score', 0):.4f}",
            "Weighted avg:",
            f"precision={weighted_avg.get('precision', 0):.4f} recall={weighted_avg.get('recall', 0):.4f} f1={weighted_avg.get('f1-score', 0):.4f}",
            "",
            "Per-category scores:"
        ]

        for category in results.get("categories", []):
            category_report = report.get(category, {})
            lines.append(
                f"{category}: precision={category_report.get('precision', 0):.4f} "
                f"recall={category_report.get('recall', 0):.4f} "
                f"f1={category_report.get('f1-score', 0):.4f} "
                f"support={int(category_report.get('support', 0))}"
            )

        return "\n".join(lines)

    def classify_text(self):
        if not self.models_loaded:
            messagebox.showwarning("Info", "Najprv nacitaj modely.")
            return

        text = self.text_entry.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Info", "Zadaj text na analyzu.")
            return

        def do_classify():
            return text, self.classifier.predict(text)

        def on_success(result):
            original_text, prediction = result
            label, probs = prediction
            out = f"Titulok:\n{original_text}\n\nKategoria: {self.category_display_names.get(label, label)}\n\n"
            for key, value in sorted(probs.items(), key=lambda item: item[1], reverse=True):
                display_name = self.category_display_names.get(key, key)
                out += f"{display_name}: {value:.3f}\n"
            self.results.delete("1.0", "end")
            self.results.insert("1.0", out)

        self._run_in_background(do_classify, on_success=on_success, busy_text="Prebieha analyza...")

    def retrain_with_learning(self):
        if not self.self_learning:
            messagebox.showwarning("Info", "Najprv nacitaj modely.")
            return

        def do_retrain():
            return self.self_learning.retrain_with_learning_data()

        def on_success(success):
            self.update_learning_stats()
            if success:
                messagebox.showinfo("OK", "Pretrenovanie bolo dokoncene.")
            else:
                messagebox.showwarning("Info", "Pretrenovanie sa nespustilo alebo zlyhalo.")

        self._run_in_background(do_retrain, on_success=on_success, busy_text="Prebieha pretrenovanie modelu...")

    def update_learning_stats(self):
        if not hasattr(self, "learning_stats"):
            return
        if not self.learning_stats.winfo_exists():
            return
        if not self.self_learning:
            self.learning_stats.delete("1.0", "end")
            self.learning_stats.insert("1.0", "Self-learning nie je inicializovany.")
            return

        stats = self.self_learning.get_learning_stats()
        text = "\n".join([f"{key}: {value}" for key, value in stats.items()])
        self.learning_stats.delete("1.0", "end")
        self.learning_stats.insert("1.0", text)

    def save_learning_data(self):
        if not self.self_learning:
            messagebox.showwarning("Info", "Self-learning nie je inicializovany.")
            return

        def do_save():
            self.self_learning._save_learning_data()
            return True

        def on_success(_):
            self.update_learning_stats()
            self._refresh_training_preview()
            messagebox.showinfo("OK", "Learning data boli ulozene.")

        self._run_in_background(do_save, on_success=on_success, busy_text="Ukladam learning data...")

    def collect_news(self):
        if not self.news_collector:
            messagebox.showwarning("Info", "Najprv nacitaj modely.")
            return

        def do_collect():
            return self.news_collector.fetch_from_rss(limit_per_feed=5)

        def on_success(news):
            self.news_box.delete("1.0", "end")
            for item in news:
                self.news_box.insert("end", f"{item['title']}\n{item['source']}\n\n")

        self._run_in_background(do_collect, on_success=on_success, busy_text="Zbieram titulky z RSS...")

    def auto_classify_news(self):
        if not self.news_collector or not self.self_learning:
            messagebox.showwarning("Info", "Najprv nacitaj modely.")
            return

        def do_auto_classify():
            return self.news_collector.auto_classify_and_learn(self.self_learning)

        def on_success(items):
            self.news_box.delete("1.0", "end")
            for item in items:
                category = self.category_display_names.get(item["predicted_category"], item["predicted_category"])
                self.news_box.insert("end", f"{item['title']} -> {category} ({item['confidence']:.2f})\n\n")
            self.update_learning_stats()
            self._refresh_training_preview()

        self._run_in_background(
            do_auto_classify,
            on_success=on_success,
            busy_text="Klasifikujem a ukladam titulky..."
        )

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    MediaLensApp().run()
