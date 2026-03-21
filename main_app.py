import os
import sys
import logging
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
        self.root.geometry("1200x750")
        self.root.minsize(1100, 700)

        self.classifier = NewsClassifier()
        self.self_learning = None
        self.news_collector = None
        self.models_loaded = False
        self.training_in_progress = False
        self.training_results = None
        self.raw_dataset_path = os.path.join("data", "raw", "training_data.csv")
        self.self_learning_dataset_path = os.path.join("data", "self_learning", "learning_data.csv")

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

        self.btn_training = ctk.CTkButton(
            self.sidebar, text="🏋️ Trénovanie",
            fg_color="transparent", anchor="w",
            command=self.show_training_view
        )
        self.btn_training.pack(fill="x", padx=20, pady=8)

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

    def show_training_view(self):
        self.clear_main()
        self.header("Trénovanie modelov")

        container = ctk.CTkFrame(self.main, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        container.grid_columnconfigure((0, 1), weight=1, uniform="training")
        container.grid_rowconfigure(1, weight=1)

        controls_card = ctk.CTkFrame(container, fg_color="#1e293b", corner_radius=12)
        controls_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))

        dataset_card = ctk.CTkFrame(container, fg_color="#1e293b", corner_radius=12)
        dataset_card.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        results_card = ctk.CTkFrame(container, fg_color="#1e293b", corner_radius=12)
        results_card.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(10, 0))
        results_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            controls_card,
            text="Vyber datasety a model",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(anchor="w", padx=20, pady=(20, 10))

        ctk.CTkLabel(
            controls_card,
            text="Môžeš kombinovať raw tréningové dáta aj self-learning dáta naraz.",
            justify="left",
            wraplength=420
        ).pack(anchor="w", padx=20, pady=(0, 10))

        self.dataset_raw_var = ctk.BooleanVar(value=True)
        self.dataset_self_learning_var = ctk.BooleanVar(value=True)

        ctk.CTkCheckBox(
            controls_card,
            text="Raw training data",
            variable=self.dataset_raw_var,
            command=self.refresh_training_dataset_preview
        ).pack(anchor="w", padx=20, pady=6)

        ctk.CTkCheckBox(
            controls_card,
            text="Self-learning data",
            variable=self.dataset_self_learning_var,
            command=self.refresh_training_dataset_preview
        ).pack(anchor="w", padx=20, pady=6)

        ctk.CTkLabel(
            controls_card,
            text="Model na trénovanie",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=20, pady=(18, 8))

        self.training_model_var = ctk.StringVar(value="rf")
        model_switcher = ctk.CTkSegmentedButton(
            controls_card,
            values=["rf", "mlp"],
            variable=self.training_model_var,
            command=lambda _value: self.refresh_training_dataset_preview()
        )
        model_switcher.pack(fill="x", padx=20, pady=(0, 16))

        self.training_selection_summary = ctk.CTkLabel(
            controls_card,
            text="",
            justify="left",
            wraplength=420
        )
        self.training_selection_summary.pack(anchor="w", padx=20, pady=(0, 12))

        self.training_status_label = ctk.CTkLabel(
            controls_card,
            text="Pripravené na spustenie trénovania.",
            justify="left",
            wraplength=420
        )
        self.training_status_label.pack(anchor="w", padx=20, pady=(0, 12))

        actions = ctk.CTkFrame(controls_card, fg_color="transparent")
        actions.pack(fill="x", padx=20, pady=(0, 20))

        self.training_button = ctk.CTkButton(
            actions,
            text="🚀 Spustiť trénovanie",
            height=42,
            command=self.start_training_from_ui
        )
        self.training_button.pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            actions,
            text="🔄 Obnoviť datasety",
            height=42,
            command=self.refresh_training_dataset_preview
        ).pack(side="left")

        ctk.CTkLabel(
            dataset_card,
            text="Titulky v datasetoch",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(anchor="w", padx=20, pady=(20, 10))

        self.training_dataset_stats = ctk.CTkTextbox(dataset_card, height=140)
        self.training_dataset_stats.pack(fill="x", padx=20, pady=(0, 12))

        self.training_titles_box = ctk.CTkTextbox(dataset_card)
        self.training_titles_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        ctk.CTkLabel(
            results_card,
            text="Výsledky testovania po trénovaní",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(anchor="w", padx=20, pady=(20, 10))

        self.training_results_box = ctk.CTkTextbox(results_card)
        self.training_results_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.refresh_training_dataset_preview()
        self.render_training_results()

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

    def try_auto_load_models(self):
        self.load_models(show_message=False)

    def load_models(self, show_message: bool = True):
        try:
            loaded = self.classifier.load_models()
            if not loaded:
                if show_message:
                    messagebox.showwarning("Info", "Modely sa nepodarilo načítať.")
                return

            self.self_learning = SelfLearningSystem(self.classifier)
            self.news_collector = NewsCollector(self.classifier)
            self.models_loaded = True
            if show_message:
                messagebox.showinfo("OK", "Modely načítané 🧠")
            self.update_learning_stats()
            self.refresh_training_dataset_preview()
        except Exception as e:
            if show_message:
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
            self.refresh_training_dataset_preview()

    def load_raw_training_data(self):
        if not os.path.exists(self.raw_dataset_path):
            return pd.DataFrame(columns=["title", "category"])
        return pd.read_csv(self.raw_dataset_path)

    def load_self_learning_training_data(self):
        if self.self_learning:
            self.self_learning._save_learning_data()

        if not os.path.exists(self.self_learning_dataset_path):
            return pd.DataFrame(columns=["title", "category", "confidence", "verified", "processed"])

        df = pd.read_csv(self.self_learning_dataset_path)
        if df.empty:
            return pd.DataFrame(columns=["title", "category", "confidence", "verified", "processed"])

        prepared = pd.DataFrame({
            "title": df.get("text", pd.Series(dtype=str)).fillna(""),
            "category": df.get("category", pd.Series(dtype=str)).fillna(""),
            "confidence": pd.to_numeric(df.get("confidence", pd.Series(dtype=float)), errors="coerce"),
            "verified": self.normalize_bool_series(df.get("verified", pd.Series(dtype=object))),
            "processed": self.normalize_bool_series(df.get("processed", pd.Series(dtype=object)))
        })
        prepared = prepared[(prepared["title"].str.strip() != "") & (prepared["category"].str.strip() != "")]
        return prepared

    def normalize_bool_series(self, series):
        normalized = series.fillna(False).apply(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        )
        return normalized

    def get_selected_training_datasets(self):
        selections = []
        if hasattr(self, "dataset_raw_var") and self.dataset_raw_var.get():
            selections.append("raw")
        if hasattr(self, "dataset_self_learning_var") and self.dataset_self_learning_var.get():
            selections.append("self_learning")
        return selections

    def build_selected_training_dataframe(self):
        selected_datasets = self.get_selected_training_datasets()
        frames = []

        if "raw" in selected_datasets:
            raw_df = self.load_raw_training_data()
            if not raw_df.empty:
                raw_df = raw_df[["title", "category"]].copy()
                raw_df["source"] = "raw"
                frames.append(raw_df)

        if "self_learning" in selected_datasets:
            learning_df = self.load_self_learning_training_data()
            if not learning_df.empty:
                learning_df = learning_df[["title", "category"]].copy()
                learning_df["source"] = "self_learning"
                frames.append(learning_df)

        if not frames:
            return pd.DataFrame(columns=["title", "category", "source"])

        combined_df = pd.concat(frames, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["title", "category"])
        return combined_df

    def refresh_training_dataset_preview(self):
        if not hasattr(self, "training_titles_box"):
            return

        raw_df = self.load_raw_training_data()
        learning_df = self.load_self_learning_training_data()
        selected = self.get_selected_training_datasets()
        combined_df = self.build_selected_training_dataframe()

        summary_lines = [
            f"Raw training data: {len(raw_df)} titulkov",
            f"Self-learning data: {len(learning_df)} titulkov",
            f"Aktuálne vybrané datasety: {', '.join(selected) if selected else 'žiadne'}",
            f"Spolu na trénovanie: {len(combined_df)} unikátnych titulkov"
        ]

        if hasattr(self, "training_selection_summary"):
            selected_model = self.training_model_var.get() if hasattr(self, "training_model_var") else "rf"
            self.training_selection_summary.configure(
                text=(
                    "Vybraný model: "
                    f"{selected_model.upper()} | "
                    f"Datasety: {', '.join(selected) if selected else 'žiadne'}"
                )
            )

        self.training_dataset_stats.delete("1.0", "end")
        self.training_dataset_stats.insert("1.0", "\n".join(summary_lines))

        title_lines = []
        if "raw" in selected:
            title_lines.append("=== RAW TRAINING DATA ===")
            title_lines.extend(
                f"[raw] {row.title} → {row.category}"
                for row in raw_df[["title", "category"]].itertuples(index=False)
            )
            title_lines.append("")

        if "self_learning" in selected:
            title_lines.append("=== SELF-LEARNING DATA ===")
            title_lines.extend(
                f"[self-learning] {row.title} → {row.category}"
                for row in learning_df[["title", "category"]].itertuples(index=False)
            )

        if not title_lines:
            title_lines = ["Nie je vybraný žiadny dataset."]

        self.training_titles_box.delete("1.0", "end")
        self.training_titles_box.insert("1.0", "\n".join(title_lines))

    def start_training_from_ui(self):
        if self.training_in_progress:
            messagebox.showinfo("Trénovanie", "Trénovanie už práve prebieha.")
            return

        selected_datasets = self.get_selected_training_datasets()
        if not selected_datasets:
            messagebox.showwarning("Dataset", "Vyber aspoň jeden dataset na trénovanie.")
            return

        training_df = self.build_selected_training_dataframe()
        if training_df.empty:
            messagebox.showwarning("Dataset", "Vybrané datasety neobsahujú žiadne použiteľné dáta.")
            return

        model_type = self.training_model_var.get()
        self.training_in_progress = True
        self.training_button.configure(state="disabled")
        self.training_status_label.configure(
            text=(
                f"Trénovanie prebieha... Model: {model_type.upper()}, "
                f"datasety: {', '.join(selected_datasets)}, vzorky: {len(training_df)}."
            )
        )
        self.training_results_box.delete("1.0", "end")
        self.training_results_box.insert(
            "1.0",
            "Trénovanie bolo spustené. Logy sleduj v termináli, výsledky sa zobrazia tu po dokončení."
        )

        worker = threading.Thread(
            target=self.run_training_job,
            args=(training_df[["title", "category"]].copy(), model_type, selected_datasets),
            daemon=True
        )
        worker.start()

    def run_training_job(self, training_df, model_type, selected_datasets):
        try:
            classifier = NewsClassifier(model_type=model_type)
            results = classifier.train(enable_augmentation=False, training_data=training_df)
            payload = {
                "results": results,
                "model_type": model_type,
                "datasets": selected_datasets,
                "sample_count": len(training_df)
            }
            self.root.after(0, lambda: self.on_training_success(payload))
        except Exception as exc:
            self.logger.exception("Chyba pri UI trénovaní")
            self.root.after(0, lambda: self.on_training_error(exc))

    def on_training_success(self, payload):
        self.training_in_progress = False
        self.training_button.configure(state="normal")
        self.training_results = payload
        self.training_status_label.configure(
            text=(
                f"Trénovanie dokončené. Model {payload['model_type'].upper()} bol "
                f"natrénovaný na {payload['sample_count']} vzorkách."
            )
        )
        self.render_training_results()
        self.load_models(show_message=False)
        messagebox.showinfo("Trénovanie", "Trénovanie bolo úspešne dokončené.")

    def on_training_error(self, exc):
        self.training_in_progress = False
        self.training_button.configure(state="normal")
        self.training_status_label.configure(text=f"Trénovanie zlyhalo: {exc}")
        self.training_results_box.delete("1.0", "end")
        self.training_results_box.insert("1.0", f"Chyba pri trénovaní:\n{exc}")
        messagebox.showerror("Trénovanie", str(exc))

    def render_training_results(self):
        if not hasattr(self, "training_results_box"):
            return

        if not self.training_results:
            self.training_results_box.delete("1.0", "end")
            self.training_results_box.insert(
                "1.0",
                "Po spustení trénovania sa tu zobrazí accuracy, priemerná istota, "
                "precision/recall/F1 pre jednotlivé kategórie aj súhrnné metriky."
            )
            return

        payload = self.training_results
        results = payload["results"]
        report = results.get("classification_report", {})
        summary_lines = [
            f"Model: {payload['model_type'].upper()}",
            f"Datasety: {', '.join(payload['datasets'])}",
            f"Počet vzoriek: {payload['sample_count']}",
            f"Accuracy: {results.get('accuracy', 0) * 100:.2f}%",
            f"Priemerná istota: {results.get('average_confidence', 0) * 100:.2f}%",
            f"Veľkosť testovacej množiny: {results.get('test_set_size', 0)}",
            f"Počet feature dimenzií: {results.get('feature_dimensions', 0)}",
            ""
        ]

        aggregate_keys = ["macro avg", "weighted avg"]
        class_keys = [
            key for key, value in report.items()
            if isinstance(value, dict) and key not in aggregate_keys
        ]

        for key in class_keys:
            metrics = report[key]
            display_name = self.category_display_names.get(key, key)
            summary_lines.extend([
                f"{display_name}:",
                f"  Precision: {metrics.get('precision', 0) * 100:.2f}%",
                f"  Recall: {metrics.get('recall', 0) * 100:.2f}%",
                f"  F1-score: {metrics.get('f1-score', 0) * 100:.2f}%",
                f"  Support: {metrics.get('support', 0)}",
                ""
            ])

        for key in aggregate_keys:
            if key in report:
                metrics = report[key]
                summary_lines.extend([
                    f"{key}:",
                    f"  Precision: {metrics.get('precision', 0) * 100:.2f}%",
                    f"  Recall: {metrics.get('recall', 0) * 100:.2f}%",
                    f"  F1-score: {metrics.get('f1-score', 0) * 100:.2f}%",
                    f"  Support: {metrics.get('support', 0)}",
                    ""
                ])

        self.training_results_box.delete("1.0", "end")
        self.training_results_box.insert("1.0", "\n".join(summary_lines))

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
