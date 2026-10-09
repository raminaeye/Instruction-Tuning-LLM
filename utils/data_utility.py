"""Lightweight text cleaning utilities."""

from unidecode import unidecode
import re


class DataUtil():
    """Simple text cleaning helpers (character replacement, counting)."""

    def replace_chars(self, text):
        """Replace punctuation and symbols with spaces or underscores.

        Also drops pure-digit tokens and tokens longer than 25 characters.

        Args:
            text: Input string.

        Returns:
            The cleaned string.
        """
        chars = {'?': ' ', ',': ' ', '/': '_', '!': ' ', '"': '_', '-': '_',
                 '{': ' ', '}': ' ', '(': ' ', ')': ' ', '[': ' ', ']': ' ',
                 '<': ' ', '>': ' ', '$': ' ', '%': ' ', ':': ' ', '&': 'and',
                 '|': ' ', '+': '_', '=': '_', '*': '_', ';': '_', '^': '_',
                 '\\': ' ', '~': ' ', '\'': '', 'http': ' ', 'nan': ' ',
                 '#': ' ', '\n': ' '}
        for k, v in chars.items():
            text = text.replace(k, v)

        text = ' '.join(text.replace(' _ ', ' ').replace('_ ', ' ')
                        .replace(' _', ' ').replace('__', '_').split())
        new_text = []
        for w in text.split():
            if not w.isdigit() and len(w) < 25:
                new_text.append(w)
        new_text = ' '.join(new_text)

        return new_text

    def split_sentence(self, text):
        """Split text into sentences on '. '.

        Args:
            text: Input string.

        Returns:
            A list of sentence strings.
        """
        return text.split('. ')

    def cleaning_text(self, text):
        """Lowercase text, strip accents, and collapse whitespace.

        Args:
            text: Input string.

        Returns:
            The normalized string.
        """
        text = unidecode(str(text))
        return ' '.join(str(text).lower().split())

    def count_qs(self, text):
        """Count question-like segments by splitting on '?'.

        Args:
            text: Input string.

        Returns:
            The number of non-empty segments.
        """
        pattern = r"[?]"
        sentences = [sentence for sentence in re.split(pattern, text)
                     if sentence]
        return len(sentences)

    def count_links_cited(self, text):
        """Count URLs (citations) in text with a URL regex.

        Args:
            text: Input string.

        Returns:
            The number of URLs found, or 0 for empty input.
        """
        regex=r"\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b"

        if len(text):
            links = re.findall(regex, text)
            return len(links)
        else:
            return 0
