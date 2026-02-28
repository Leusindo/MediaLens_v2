import customtkinter as ctk

# ===== THEME =====
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

BG_MAIN = "#0f172a"
CARD_BG = "#1e293b"
NAV_ACTIVE_BG = "#1e3a8a"
NAV_NORMAL_BG = "transparent"


# ===== APP =====
class MediaLensPresentation:
    def __init__(self):
        self.nav_buttons = {}
        self.font_title = 40
        self.font_header = 34
        self.font_body = 22

        self.wrap_width = 1100

        ctk.set_widget_scaling(1.4)
        ctk.set_window_scaling(1.4)

        self.root = ctk.CTk()
        self.root.title("MediaLens AI — SOČ prezentácia")
        self.root.geometry("1200x750")
        self.root.attributes('-fullscreen', True)
        self.root.minsize(1100, 700)

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.build_sidebar()
        self.build_main()

        self.title()

    # ===== LAYOUT =====
    def build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self.root, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(
            self.sidebar,
            text="MediaLens AI",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(pady=(30, 20))

        self.add_nav("SOČ", self.title, "soc")
        self.add_nav("Problematika", self.problem, "problem")
        self.add_nav("Cieľ práce", self.goal, "goal")
        self.add_nav("Metodika", self.tech, "tech")
        self.add_nav("Použité technológie", self.medialens, "medialens")
        self.add_nav("Výsledky a overenie", self.functionality, "functionality")
        self.add_nav("Ukážka", self.video, "video")
        self.add_nav("Závery a prínos práce", self.results, "results")
        self.add_nav("Plány a budúci rozvoj", self.future, "future")
        self.add_nav("      Rozšírenie datasetu", self.future_data, "future_data")
        self.add_nav("      Hlbšia analýza", self.future_anal, "future_anal")
        self.add_nav("      Optimalizácia modelov", self.future_model, "future_model")
        self.add_nav("Koniec", self.thanks, "thanks")

    def add_nav(self, text, cmd, key):
        btn = ctk.CTkButton(
            self.sidebar,
            text=text,
            fg_color=NAV_NORMAL_BG,
            hover_color=NAV_ACTIVE_BG,
            anchor="w",
            command=lambda: self.activate_nav(key, cmd)
        )
        btn.pack(fill="x", padx=20, pady=6)
        self.nav_buttons[key] = btn

    def activate_nav(self, key, cmd):
        # reset všetkých
        for b in self.nav_buttons.values():
            b.configure(fg_color=NAV_NORMAL_BG)

        # zvýrazni aktívny
        self.nav_buttons[key].configure(fg_color=NAV_ACTIVE_BG)

        # zobraz slide
        cmd()

    def build_main(self):
        self.main = ctk.CTkFrame(self.root, fg_color=BG_MAIN, corner_radius=15)
        self.main.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

    def clear_main(self):
        for w in self.main.winfo_children():
            w.destroy()

    # ===== UI HELPERS =====
    def header(self, text):
        ctk.CTkLabel(
            self.main,
            text=text,
            font=ctk.CTkFont(size=self.font_header, weight="bold"),
            wraplength=self.wrap_width,
            justify="left"
        ).pack(anchor="w", padx=50, pady=(40, 25))

    def card(self):
        frame = ctk.CTkFrame(self.main, fg_color=CARD_BG, corner_radius=16)
        frame.pack(fill="both", expand=True, padx=40, pady=20)
        return frame

    def bullet(self, parent, text):
        ctk.CTkLabel(
            parent,
            text="• " + text,
            font=ctk.CTkFont(size=self.font_body),
            wraplength=self.wrap_width,
            justify="left"
        ).pack(anchor="w", padx=30, pady=(30, 6))

    def tab_bullet(self, parent, text):
        ctk.CTkLabel(
            parent,
            text="• " + text,
            font=ctk.CTkFont(size=self.font_body - 2),
            wraplength=self.wrap_width,
            justify="left"
        ).pack(anchor="w", padx=70, pady=6)

    def section(self, parent, text):
        ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=self.font_body + 2, weight="bold"),
            wraplength=self.wrap_width,
            justify="left"
        ).pack(anchor="w", padx=50, pady=(18, 6))

    def spacer(self, parent, h=10):
        ctk.CTkFrame(parent, fg_color="transparent", height=h).pack(fill="x")

    # ===== SLIDES =====
    def title(self):
        self.clear_main()
        self.header("Využitie hybridných NLP modelov pri detekcii manipulatívnych techník v mediálnych titulkoch")
        card = self.card()

        self.bullet(card, "Súkromná stredná odborná škola, Ul. 29 Augusta 4812, Poprad")
        self.bullet(card, "Č. odboru: 11 – Informatika")
        self.bullet(card, "Autor: Leo Ondrejka")

    def problem(self):
        self.clear_main()
        self.header("Problematika")
        card = self.card()

        self.bullet(card, "Informačné preťaženie")
        self.bullet(card, "Clickbait")
        self.bullet(card, "Emócie namiesto faktov")
        self.bullet(card, "Slabé rozpoznávanie manipulácie")

    def goal(self):
        self.clear_main()
        self.header("Cieľ práce")
        card = self.card()

        self.bullet(card, "AI systém na analýzu titulkov")
        self.bullet(card, "Zameranie na slovenský jazyk")
        self.bullet(card, "Kombinácia lexiky a kontextu")
        self.bullet(card, "Praktický a zrozumiteľný nástroj")

    def tech(self):
        self.clear_main()
        self.header("Metodika")
        card = self.card()

        self.bullet(card, "Experimentálna pipeline")
        self.bullet(card, "Vlastný dataset (SK titulky)")
        self.bullet(card, "Manuálna anotácia")
        self.bullet(card, "TF-IDF + BERT")
        self.bullet(card, "Random Forest")
        self.bullet(card, "Vyvažovanie tried")
        self.bullet(card, "Konzervatívny self-learning")

    def medialens(self):
        self.clear_main()
        self.header("Použité technológie")
        card = self.card()

        self.bullet(card, "Python")
        self.bullet(card, "Scikit-Learn, PyTorch, Numpy")
        self.bullet(card, "TF-IDF (lexikálny prístup)")
        self.bullet(card, "BERT (kontextový prístup)")
        self.bullet(card, "Random Forest")
        self.bullet(card, "Hybridný model")

    def functionality(self):
        self.clear_main()
        self.header("Výsledky a overenie")
        card = self.card()

        self.bullet(card, "Vysoká presnosť pri clickbaite")
        self.bullet(card, "Slabšie pri jemných klamoch")
        self.bullet(card, "Dotazník: 90+ respondentov")
        self.bullet(card, "Ľudia majú podobné problémy")

    def video(self):
        self.clear_main()
        self.header("Praktická ukážka")
        card = self.card()

        self.bullet(card, "Zadanie titulku")
        self.bullet(card, "Klasifikácia + pravdepodobnosti")
        self.bullet(card, "Porovnanie clickbait vs. neutrálny")
        self.bullet(card, "Auto-klasifikácia titulkov")

    def results(self):
        self.clear_main()
        self.header("Záver a prínos")
        card = self.card()

        self.bullet(card, "Splnené ciele práce")
        self.bullet(card, "Funkčný prototyp pre SK jazyk")
        self.bullet(card, "Podpora kritického myslenia")
        self.bullet(card, "Vzdelávacie využitie")

    def future(self):
        self.clear_main()
        self.header("Plány a budúci rozvoj")
        card = self.card()

        self.bullet(card, "Rozšírenie datasetu")
        self.bullet(card, "Hlbšia analýza textov")
        self.bullet(card, "Optimalizácia modelov")

    def future_data(self):
        self.clear_main()
        self.header("Rozšírenie datasetu")
        card = self.card()

        self.bullet(card, "Zber väčšieho množstva dát")
        self.bullet(card, "Vyváženie tried")
        self.bullet(card, "Lepšia generalizácia modelu")

    def future_anal(self):
        self.clear_main()
        self.header("Hlbšia analýza")
        card = self.card()

        self.bullet(card, "Analýza celých článkov")
        self.bullet(card, "Overovanie faktov voči externým zdrojom")

    def future_model(self):
        self.clear_main()
        self.header("Optimalizácia modelov")
        card = self.card()

        self.bullet(card, "TF-IDF + SVM")
        self.bullet(card, "BERT + Logistic Regression")
        self.bullet(card, "MLP – neurónové siete")

    def thanks(self):
        self.clear_main()
        self.header("Ďakujem za pozornosť")
        card = self.card()

        self.bullet(card, "MediaLens – detekcia manipulácie v médiách")
        self.bullet(card, "Otázky?")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    MediaLensPresentation().run()
