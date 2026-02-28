# core/news_collector.py
import logging
import os
import time
from datetime import datetime
from typing import List, Dict

import feedparser
import pandas as pd

from .config import Config


class NewsCollector:


    def __init__(self, classifier=None):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.classifier = classifier

        self.rss_feeds = [
            # mainstream
            "https://www.aktuality.sk/rss/",
            "https://www.teraz.sk/rss/vsetky-spravy.rss",
            "https://sita.sk/feed/",
            "https://www1.pluska.sk/rss.xml",
            "https://www.sme.sk/rss-title",
            "https://spravy.pravda.sk/rss/xml/",
            "https://hnonline.sk/feed",
            "https://www.korzar.sme.sk/rss/",
            "https://tech.sme.sk/rss-title",
            "https://ekonomika.sme.sk/rss-title",
            "https://sport.sme.sk/rss-title",
            "https://www.dennikn.sk/feed/",
            "https://www.startitup.sk/feed/",

            # bulvár
            "https://www.topky.sk/rss/",
            "https://www.cas.sk/rss/",
            "https://www.pluska.sk/rss.xml",
            "https://www.noviny.sk/rss",
            "https://www.emma.sk/rss",
            "https://www.zivot.sk/rss",

            # alternatívne / kontroverzné
            "https://www.hlavnespravy.sk/feed/",
            "https://www.infovojna.sk/rss",
            "https://www.armadnymagazin.sk/feed/",
            "https://www.badatel.net/feed/",
            "https://www.slovanskenoviny.sk/feed/",
            "https://www.hlavnydennik.sk/feed/",
            "https://www.extraplus.sk/feed/",
            "https://www.eurorespekt.sk/feed/",

            # tech / blog
            "https://www.zive.sk/rss/",
            "https://touchit.sk/feed/",
            "https://fontech.sk/feed/",
            "https://www.techbox.sk/feed/",

            # regionálne
            "https://presov.korzar.sme.sk/rss",
            "https://kosice.korzar.sme.sk/rss",
        ]

        self.news_websites = [
            "https://www.sme.sk",
            "https://spravy.pravda.sk",
            "https://www.aktuality.sk",
            "https://www.pluska.sk",
            "https://www.topky.sk",
            "https://www.cas.sk",
            "https://www.noviny.sk",
            "https://www.hlavnespravy.sk",
            "https://www.infovojna.sk",
            "https://www.badatel.net",
            "https://www.armadnymagazin.sk",
            "https://www.startitup.sk",
            "https://www.dennikn.sk",
        ]

        self.collected_file = "data/collected_news/collected_titles.csv"
        os.makedirs("data/collected_news", exist_ok=True)

        self.logger.info("News collector inicializovaný")

    def fetch_from_rss(self, limit_per_feed: int = 25) -> List[Dict[str, str]]:
        all_titles = []
        seen_titles = set()

        for feed_url in self.rss_feeds:
            try:
                self.logger.info(f"Načítavam RSS: {feed_url}")

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/rss+xml, application/xml, text/xml'
                }

                feed = feedparser.parse(feed_url, request_headers=headers)
                entries_to_process = feed.entries[:limit_per_feed] if hasattr(feed, 'entries') else []

                titles_from_feed = 0
                for entry in entries_to_process:
                    if hasattr(entry, 'title'):
                        title = entry.title.strip()

                        normalized_title = self._normalize_title(title)

                        if not normalized_title or len(normalized_title) < 15:
                            continue

                        if normalized_title in seen_titles:
                            self.logger.debug(f"Duplikát preskočený: '{normalized_title}'")
                            continue

                        if self._is_slovak_title(normalized_title):
                            news_item = {
                                'title': title,
                                'normalized_title': normalized_title,
                                'source': feed_url,
                                'published': entry.get('published', ''),
                                'link': entry.get('link', ''),
                                'collected_at': datetime.now().isoformat()
                            }
                            all_titles.append(news_item)
                            seen_titles.add(normalized_title)
                            titles_from_feed += 1

                self.logger.info(f"Získaných {titles_from_feed} titulkov z {feed_url}")

                time.sleep(0.5)

            except Exception as e:
                self.logger.error(f"Chyba pri {feed_url}: {e}")
                continue

        self.logger.info(f"Celkovo získaných {len(all_titles)} UNIKÁTNYCH titulkov")
        return all_titles

    def _normalize_title(self, title: str) -> str:
        if not title:
            return ""

        normalized = title.lower()

        import re
        normalized = re.sub(r'[^\w\sáäčďéíľĺňóôřŕšťúýž]', '', normalized)

        normalized = re.sub(r'\s+', ' ', normalized).strip()

        prefixes = ['video:', 'foto:', 'video |', 'foto |', 'exkluzívne:', 'breaking:']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()

        return normalized

    def _is_slovak_title(self, title: str) -> bool:
        if not title:
            return False

        slovak_chars = ['á', 'ä', 'č', 'ď', 'é', 'í', 'ľ', 'ĺ', 'ň', 'ó', 'ô', 'ŕ', 'š', 'ť', 'ú', 'ý', 'ž']

        slovak_words = [
            'a', 'o', 'v', 's', 'z', 'čo', 'ako', 'kde', 'prečo', 'ktorý', 'ktorá', 'ktoré',
            'pri', 'po', 'na', 'do', 'za', 'so', 'sa', 'si', 'je', 'bol', 'bola', 'bolo'
        ]

        title_lower = title.lower()

        slovak_char_count = sum(1 for char in title_lower if char in slovak_chars)

        slovak_word_count = sum(1 for word in slovak_words if word in title_lower.split())

        return slovak_char_count >= 2 or slovak_word_count >= 2

    def auto_classify_and_learn(self, self_learning_system=None, min_confidence: float = 0.85) -> List[Dict]:
        try:
            self.logger.info("Začínam automatickú klasifikáciu a učenie...")

            news_items = self.fetch_from_rss(limit_per_feed=25)

            if not news_items:
                self.logger.info("Neboli nájdené žiadne nové titulky")
                return []

            classified_items = []
            classified_cache = {}

            for item in news_items:
                title = item['title']
                normalized_title = item.get('normalized_title', self._normalize_title(title))

                if normalized_title in classified_cache:
                    self.logger.debug(f"Používam cache pre: '{normalized_title}'")
                    cached_result = classified_cache[normalized_title]

                    classified_item = {
                        **item,
                        'predicted_category': cached_result['category'],
                        'confidence': cached_result['confidence'],
                        'added_to_learning': False,
                        'probabilities': cached_result['probabilities']
                    }
                    classified_items.append(classified_item)
                    continue

                if len(title) < 15:
                    continue

                try:
                    if self_learning_system and self.classifier:
                        category, probabilities, added_to_learning = self_learning_system.predict_with_learning(title)
                    elif self.classifier:
                        category, probabilities = self.classifier.predict(title)
                        added_to_learning = False
                    else:
                        continue

                    confidence = max(probabilities.values())

                    if confidence < min_confidence:
                        continue

                    classified_cache[normalized_title] = {
                        'category': category,
                        'confidence': confidence,
                        'probabilities': probabilities
                    }

                    classified_item = {
                        **item,
                        'predicted_category': category,
                        'confidence': confidence,
                        'added_to_learning': added_to_learning,
                        'probabilities': probabilities
                    }
                    classified_items.append(classified_item)

                    if confidence > 0.85:
                        self.logger.info(f"{category}: '{title[:50]}...' ({confidence:.3f})")

                except Exception as e:
                    self.logger.error(f"Chyba pri klasifikácii: {e}")
                    continue

            self._save_collected_news(classified_items)

            unique_titles = len(set(item.get('normalized_title', self._normalize_title(item['title']))
                                    for item in classified_items))
            self.logger.info(f"Spracovaných {len(classified_items)} titulkov ({unique_titles} unikátnych)")

            return classified_items

        except Exception as e:
            self.logger.error(f"Chyba v auto_classify_and_learn: {e}")
            return []

    def _save_collected_news(self, news_items: List[Dict]):
        try:
            if not news_items:
                return

            try:
                existing_df = pd.read_csv(self.collected_file)
            except FileNotFoundError:
                existing_df = pd.DataFrame()

            new_df = pd.DataFrame(news_items)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            combined_df = combined_df.drop_duplicates(subset=['title'])

            combined_df.to_csv(self.collected_file, index=False)

            self.logger.info(f"Uložených {len(new_df)} nových titulkov")

        except Exception as e:
            self.logger.error(f"Chyba pri ukladaní titulkov: {e}")

    def get_recent_news(self, hours: int = 24) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.collected_file)

            if 'collected_at' in df.columns:
                df['collected_at'] = pd.to_datetime(df['collected_at'])
                cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
                recent_df = df[df['collected_at'] > cutoff_time]
            else:
                recent_df = df.tail(50)

            return recent_df

        except FileNotFoundError:
            self.logger.info("Súbor s nazbieranými titulkmi neexistuje")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Chyba pri načítavaní recent news: {e}")
            return pd.DataFrame()

    def get_news_stats(self) -> Dict[str, any]:
        try:
            df = self.get_recent_news(hours=168)

            if df.empty:
                return {'total_news': 0}

            stats = {
                'total_news': len(df),
                'sources': df['source'].value_counts().to_dict(),
                'categories': df[
                    'predicted_category'].value_counts().to_dict() if 'predicted_category' in df.columns else {},
                'high_confidence_news': len(df[df['confidence'] > 0.85]) if 'confidence' in df.columns else 0
            }

            return stats

        except Exception as e:
            self.logger.error(f"Chyba pri získavaní štatistík: {e}")
            return {}
