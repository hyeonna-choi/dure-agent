# bioforge/utils/text_parser.py

import re
from typing import Optional


class TextParser:
    """Text parsing helper class."""

    @staticmethod
    def extract_tag(text: str, tag: str) -> Optional[str]:
        """Extract the content of a specific tag from text."""
        pattern = rf"<{tag}>(.+?)</{tag}>"
        match = re.search(pattern, text, flags=re.S)
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_int(pattern: str, text: str, default: int = 0) -> int:
        """Extract an integer using a regular expression pattern."""
        match = re.search(pattern, text, flags=re.I)
        return int(match.group(1)) if match else default

    @staticmethod
    def extract_volume_ul(line: str, default_ul: int) -> int:
        """Extract volume (microliters)."""
        match = re.search(r"(\d+)\s*(?:u?µ?l)\b", line, flags=re.I)
        return int(match.group(1)) if match else default_ul

    @staticmethod
    def extract_minutes(line: str, default_min: int) -> int:
        """Extract time (minutes)."""
        # Minute unit
        match = re.search(r"(\d+)\s*min", line, flags=re.I)
        if match:
            return int(match.group(1))

        # Korean "minutes"
        match = re.search(r"(\d+)\s*분", line, flags=re.I)
        if match:
            return int(match.group(1))

        # Hours -> minutes conversion
        match = re.search(r"(\d+)\s*시간", line, flags=re.I)
        if match:
            return int(match.group(1)) * 60

        return default_min

    @staticmethod
    def extract_rpm(line: str, default_rpm: int) -> int:
        """Extract RPM."""
        match = re.search(r"(\d+)\s*rpm", line, flags=re.I)
        return int(match.group(1)) if match else default_rpm

    @staticmethod
    def extract_temperature(line: str, default_temp: int) -> int:
        """Extract temperature (degrees C)."""
        match = re.search(r"(\d+)\s*°?\s*c", line, flags=re.I)
        return int(match.group(1)) if match else default_temp

    @staticmethod
    def parse_reservoir_col(line: str, default_col: int = 1) -> int:
        """Parse the Reservoir column - no count limit."""
        # Direct numeric specification
        match = re.search(r"reservoir\s*(\d+)", line, flags=re.I)
        if match:
            col = int(match.group(1))
            return max(1, col)

        # Keyword-based default mapping
        keyword_mappings = [
            (r"wash\s*buffer|세척", 1),
            (r"detection\s*antibody|검출\s*항체", 2),
            (r"streptavidin[-\s]*hrp", 3),
            (r"\btmb\b|substrate", 4),
            (r"stop\s*solution|1\s*m\s*hcl", 5),
            (r"blocking|봉쇄", 6),
            (r"primary.*antibody|1차.*항체", 7),
            (r"secondary.*antibody|2차.*항체", 8),
        ]
        
        for pattern, col in keyword_mappings:
            if re.search(pattern, line, flags=re.I):
                return col
        
        return default_col
    
    @staticmethod
    def format_parameters(*args) -> str:
        """Format parameters as a space-separated string."""
        return " ".join(str(arg) for arg in args)

    @staticmethod
    def extract_section(text: str, section_number: int) -> str:
        """Extract the content of a specific section number."""
        pattern = rf"\[\s*{section_number}\s*\][^\n]*\n(.*?)(?=\n\[\s*\d+\s*\]|$)"
        match = re.search(pattern, text, flags=re.S|re.I)
        return match.group(1).strip() if match else ""

    @staticmethod
    def clean_line(line: str) -> str:
        """Clean a line (strip whitespace, tidy special characters)."""
        return line.strip()

    @staticmethod
    def is_automation_step(line: str) -> bool:
        """Determine whether this is an automation step."""
        automation_keywords = [
            "분주", "dispense", "aspirate", "흡인", "pipette",
            "wash", "세척", "incubat", "배양", "shake", "흔들"
        ]
        return any(keyword in line.lower() for keyword in automation_keywords)

    @staticmethod
    def extract_repeat_count(line: str) -> int:
        """Extract the repeat count."""
        patterns = [
            r"(\d+)\s*[회xX×]",
            r"(\d+)\s*times?",
            r"repeat\s*(\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.I)
            if match:
                return int(match.group(1))
        
        return 1