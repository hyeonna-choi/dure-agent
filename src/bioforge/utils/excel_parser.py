# bioforge/utils/excel_parser.py

import re
from pathlib import Path
from typing import Dict, List
import pandas as pd


class ExcelParser:
    """Excel file parsing helper class."""

    def __init__(self, config):
        self.config = config

    def load_all_sheets(self, xlsx_path: Path = None) -> Dict[str, pd.DataFrame]:
        """Load all sheets from an Excel file."""
        if xlsx_path is None:
            xlsx_path = self.config.EXCEL_PATH

        if not xlsx_path.exists():
            raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

        with pd.ExcelFile(xlsx_path) as xls:
            return {name: xls.parse(name) for name in xls.sheet_names}

    def get_allowed_commands(self, sheets: Dict[str, pd.DataFrame]) -> List[str]:
        """Extract the list of allowed commands from the Excel file."""
        cmd_list = []

        for sheet_name, df in sheets.items():
            # Normalize column names: strip whitespace + lowercase
            normalized_cols = {
                col: re.sub(r"\s+", "", str(col)).lower()
                for col in df.columns
            }

            # Find the actual column name that matches 'command'
            cmd_col = None
            for original_col, normalized_col in normalized_cols.items():
                if normalized_col == "command":
                    cmd_col = original_col
                    break

            if cmd_col is None:
                continue

            # Extract and normalize the commands
            commands = (
                df[cmd_col]
                .dropna()
                .astype(str)
                .map(lambda s: s.strip())
                .tolist()
            )
            cmd_list.extend(commands)

        # Intersect with the whitelist
        allowed_commands = sorted(
            set(cmd_list).intersection(self.config.COMMAND_WHITELIST)
        )

        return allowed_commands

    def infer_deck_slots(self, sheets: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """Extract the deck slot mapping."""
        # If there is no explicit deck map sheet or its format is unclear, use the default mapping
        # Logic to parse a deck mapping sheet from the Excel file can be added later
        deck_slots = dict(self.config.DEFAULT_SLOTS)

        # Check whether a deck mapping sheet exists
        deck_sheet_names = ['deck', 'slot', 'mapping', '덱', '슬롯']
        for sheet_name in sheets.keys():
            if any(name in sheet_name.lower() for name in deck_sheet_names):
                try:
                    df = sheets[sheet_name]
                    # Deck mapping parsing logic (implement if needed)
                    pass
                except Exception:
                    continue

        return deck_slots

    def extract_reservoir_mapping(self, sheets: Dict[str, pd.DataFrame]) -> Dict[int, str]:
        """Extract Reservoir mapping information."""
        reservoir_map = {}

        # Find the Reservoir sheet
        reservoir_sheet_names = ['reservoir', 'reagent', '시약', '저장소']
        for sheet_name in sheets.keys():
            if any(name in sheet_name.lower() for name in reservoir_sheet_names):
                try:
                    df = sheets[sheet_name]
                    # Reservoir mapping parsing logic (implement if needed)
                    pass
                except Exception:
                    continue

        return reservoir_map