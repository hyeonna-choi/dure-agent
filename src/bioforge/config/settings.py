# bioforge/config/settings.py

from pathlib import Path
from typing import Dict, List

# Project root (the `src` directory), resolved relative to this file so that all
# data and output paths are independent of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config:
    """Project configuration class."""

    def __init__(self):
        # Default path settings
        self.INPUT_DIR = PROJECT_ROOT / "data" / "input" / "pdfs"
        self.EXCEL_PATH = PROJECT_ROOT / "data" / "rules" / "bioforge_commands_and_rules.xlsx"
        self.OUTPUT_DIR = PROJECT_ROOT / "output" / "structured_protocols"
        self.SEQUENCE_OUTPUT_DIR = PROJECT_ROOT / "output" / "sequences"
        self.WELLPLATE_INPUT = None

        # File suffixes
        self.STRUCTURED_SUFFIX = "_StructuredProtocol.txt"
        self.SEQUENCE_SUFFIX = "_Seq.xlsx"

        # LLM settings
        # Single model: "gpt-5"  /  Multiple models: ["gpt-5", "gpt-4.1", "gpt-4.1-mini", "o4-mini", "gpt-4.1-nano"]
        # Direct mode (PDF -> Structured -> LLM mapping -> Sequence, no rule-based step):
        #   ["gpt-5-direct", "gpt-4.1-direct", "gpt-4.1-mini-direct", "o4-mini-direct", "gpt-4.1-nano-direct"]
        # self.LLM_MODEL = ["gpt-5", "gpt-4.1", "gpt-4.1-mini", "o4-mini", "gpt-4.1-nano", "llama-4-maverick", "llama-3.3-70b"]
        self.LLM_MODEL = ["gpt-5"]

        # Validator model settings - single str or list
        # Claude: "claude-sonnet-4-6", "claude-opus-4-6"
        # GPT:    "gpt-5", "gpt-4.1"
        # Llama:  "llama-4-maverick", "llama-3.3-70b"
        # e.g.) single: "claude-sonnet-4-6"
        # e.g.) multiple: ["claude-sonnet-4-6", "llama-4-maverick", "gpt-5"]
        self.VALIDATOR_MODEL = "claude-sonnet-4-6"

        # Validation-only mode: True = skip Structuring/Sequence generation, run Validation only
        # The structured/sequence files must already exist in the results folder
        self.VALIDATION_ONLY = False

        # V0 re-validation mode: True = clone the output folder + restore v0_initial files into structured/sequences
        # -> Use together with VALIDATION_ONLY=True. Clone folder name: output_{validator_model}
        self.V0_REVALIDATION = False

        # V0 re-validation source folder (absolute or relative path)
        # If left empty, use the output/ folder based on OUTPUT_DIR (default)
        # If the output/ folder does not exist, specify directly: "bioforge/output_claude-sonnet-4-6"
        self.V0_REVALIDATION_SRC = "bioforge/output_claude-sonnet-4-6"

        # temperature is determined automatically per model by get_temperature() (o4-mini=0, others=1)
        self.LLM_TEMPERATURE = 0
        self.MAX_RETRIES = 6
        self.COOLDOWN_SEC = 1.5

        # Text splitting settings
        self.CHUNK_SIZE = 1500
        self.CHUNK_OVERLAP = 200

        # Default deck slot mapping (based on the reference image)
        self.DEFAULT_SLOTS: Dict[str, int] = {
            "plate_slot": 1,          # Sample plate
            "magnet_slot": 2,         # Magnet
            "hotplate_slot": 3,       # Incubator/Hotplate (incubation/reaction)
            "tip_slot": 4,            # Clean Tip
            "reservoir_slot": 5,      # Reservoir
            "shaker_slot": 6,         # Shaker
            "tip_waste_slot": 7,      # Tip disposal
            "waste_slot": 8,          # Solution disposal (treated like a plate)
            "cap_parking_slot": 9,    # Cap parking
        }

        # Default values
        self.DEFAULTS: Dict[str, int] = {
            "aspirate_speed": 120,
            "dispense_speed": 120,
            "volume_ul": 0,
            "incubation_min": 0,   # 0 if no time is given in the sentence
            "shake_speed": 300,    # Default RPM (used when rpm is not specified in the PDF)
            "shake_time_sec": 5,
            "hotplate_temp": 20,   # When incubation temperature is not specified (room temperature = 20 degrees C)
        }

        # Allowed command whitelist (only the intersection with the Excel file is used in the end)
        self.COMMAND_WHITELIST: List[str] = [
            "A#GET", "A#PUT",
            "A#TEMPPLATE_ON", "A#TEMPPLATE_OFF",
            "A#CAP_OPEN", "A#CAP_CLOSE",
            "A#ADP_SELECT_1_CHANNEL", "A#ADP_SELECT_4_CHANNEL",
            "A#ADP_INSTALL_PIPETTE", "A#ADP_EJECT_PIPETTE",
            "A#ADP_ASPIRATE_FROM_PLATE", "A#ADP_DISPENSE_INTO_PLATE",
            "A#TRANS", "A#SHAKE_START",
            "A#WAIT_MOTION",
            "A#ANALYZER_OPEN", "A#ANALYZER_CLOSE", "A#ANALYZER_START",
        ]

        # Default Reservoir mapping (keyword-based)
        self.RESERVOIR_MAPPING: Dict[str, int] = {
            "wash_buffer": 1,
            "detection_antibody": 2,
            "streptavidin_hrp": 3,
            "tmb_substrate": 4,
            "stop_solution": 5,
            "blocking_buffer": 6,
            "primary_antibody": 7,
            "secondary_antibody": 8,
        }
        
        # Handler number settings
        self.HANDLERS: Dict[str, int] = {
            "pipette": 73,        # Pipette-related
            "transport": 74,      # Plate transport
            "incubator": 75,      # Incubator/Analyzer
            "shaker": 76,         # Shaker
        }

        # File handling settings
        self.SKIP_EXISTING = False  # Whether to skip existing files
        self.BACKUP_EXISTING = True  # Whether to back up existing files

        # Logging settings
        self.LOG_LEVEL = "INFO"
        self.LOG_FORMAT = "[%(levelname)s] %(message)s"

        # Validation settings
        self.VALIDATE_SEQUENCES = True
        self.STRICT_VALIDATION = False  # Strict validation mode

        # Performance settings
        self.PARALLEL_PROCESSING = False  # Parallel processing (to be implemented)
        self.MAX_WORKERS = 2

    def ensure_output_dirs(self):
        """Create output directories."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.SEQUENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def validate_paths(self) -> List[str]:
        """Validate paths."""
        errors = []

        if not self.INPUT_DIR.exists():
            errors.append(f"Input directory does not exist: {self.INPUT_DIR}")

        if not self.EXCEL_PATH.exists():
            errors.append(f"Excel rules file does not exist: {self.EXCEL_PATH}")

        return errors

    def get_reservoir_col(self, reagent_type: str) -> int:
        """Return the Reservoir column number for the given reagent type."""
        return self.RESERVOIR_MAPPING.get(reagent_type, 1)

    def get_handler(self, operation_type: str) -> int:
        """Return the handler number for the given operation type."""
        return self.HANDLERS.get(operation_type, 73)

    def get_slot(self, slot_name: str) -> int:
        """Return the slot number for the given slot name."""
        return self.DEFAULT_SLOTS.get(slot_name, 1)

    @staticmethod
    def get_temperature(model_name: str) -> int:
        """Determine temperature automatically per model: o-series (o1, o3, o4-mini, etc.)=0, others=1."""
        if model_name.startswith("o"):
            return 0
        return 1

    def get_model_list(self) -> list:
        """If LLM_MODEL is a str return [str], if a list return it as is."""
        if isinstance(self.LLM_MODEL, list):
            return list(self.LLM_MODEL)
        return [self.LLM_MODEL]

    def get_validator_model_list(self) -> list:
        """If VALIDATOR_MODEL is a str return [str], if a list return it as is."""
        if isinstance(self.VALIDATOR_MODEL, list):
            return list(self.VALIDATOR_MODEL)
        return [self.VALIDATOR_MODEL]

    def get_default_value(self, parameter: str) -> int:
        """Return the default value."""
        return self.DEFAULTS.get(parameter, 0)

    def is_command_allowed(self, command: str) -> bool:
        """Check whether the command is allowed."""
        return command in self.COMMAND_WHITELIST

    def update_paths(self, **kwargs):
        """Update path settings."""
        for key, value in kwargs.items():
            if hasattr(self, key.upper()):
                setattr(self, key.upper(), Path(value))
            else:
                raise ValueError(f"Unknown path setting: {key}")

    def to_dict(self) -> Dict:
        """Convert the configuration to a dictionary."""
        return {
            "paths": {
                "input_dir": str(self.INPUT_DIR),
                "excel_path": str(self.EXCEL_PATH),
                "output_dir": str(self.OUTPUT_DIR),
                "sequence_output_dir": str(self.SEQUENCE_OUTPUT_DIR),
            },
            "llm": {
                "model": self.LLM_MODEL,
                "temperature": self.LLM_TEMPERATURE,
                "max_retries": self.MAX_RETRIES,
            },
            "processing": {
                "chunk_size": self.CHUNK_SIZE,
                "chunk_overlap": self.CHUNK_OVERLAP,
                "cooldown_sec": self.COOLDOWN_SEC,
            },
            "defaults": self.DEFAULTS,
            "slots": self.DEFAULT_SLOTS,
            "commands": len(self.COMMAND_WHITELIST),
        }
    
    def print_summary(self):
        """Print a summary of the configuration."""
        print("=== Bioforge Protocol Processor Settings ===")
        print(f"Input directory: {self.INPUT_DIR}")
        print(f"Rules Excel file: {self.EXCEL_PATH}")
        print(f"Structured protocol output: {self.OUTPUT_DIR}")
        print(f"Sequence output: {self.SEQUENCE_OUTPUT_DIR}")
        print(f"LLM model: {self.LLM_MODEL}")
        print(f"Number of allowed commands: {len(self.COMMAND_WHITELIST)}")
        print(f"Number of deck slots: {len(self.DEFAULT_SLOTS)}")
        print("=" * 45)


# Global configuration instance (singleton pattern)
_config_instance = None

def get_config() -> Config:
    """Return the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

def reset_config():
    """Reset the configuration."""
    global _config_instance
    _config_instance = None