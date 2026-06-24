# bioforge/utils/file_handler.py

from pathlib import Path
import pandas as pd


class FileHandler:
    """File handling helper class."""

    def __init__(self, config):
        self.config = config

    def save_structured_protocol(self, output_path: Path, kit_title: str,
                                natural_body: str, instrument_body: str):
        """Save the structured protocol to a file."""
        content = (
            f"<KIT_TITLE>{kit_title}</KIT_TITLE>\n\n"
            f"<NATURAL>\n{natural_body}\n</NATURAL>\n\n"
            f"<INSTRUMENT>\n{instrument_body}\n</INSTRUMENT>\n"
        )
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    def read_protocol_file(self, file_path: Path) -> str:
        """Read a protocol file."""
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def ensure_directory(self, directory: Path):
        """Ensure the directory exists."""
        directory.mkdir(parents=True, exist_ok=True)

    def save_sequence_excel(self, sequences: list, output_path: Path):
        """Save the sequence to an Excel file."""
        df = pd.DataFrame(sequences, columns=["Handler", "Command", "Input Parameters"])
        df.to_excel(output_path, index=False)

    def read_excel_to_text(self, excel_path: Path) -> str:
        """Convert an Excel file to text (for reference)."""
        if not excel_path.exists():
            return ""
        try:
            df = pd.read_excel(excel_path)
            return df.to_string(index=False)
        except Exception as e:
            return f"(Failed to load Excel file: {e})"