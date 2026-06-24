# bioforge/models/protocol.py

from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


@dataclass
class ProtocolSection:
    """Protocol section data class."""
    kit_title: str
    natural_steps: str
    instrument_steps: str

    @classmethod
    def from_text(cls, protocol_text: str) -> 'ProtocolSection':
        """Parse protocol sections from text."""
        from utils.text_parser import TextParser

        kit_title = TextParser.extract_tag(protocol_text, "KIT_TITLE") or "Not specified"
        natural_steps = TextParser.extract_tag(protocol_text, "NATURAL") or "Not specified"
        instrument_steps = TextParser.extract_tag(protocol_text, "INSTRUMENT") or "Not specified"

        return cls(
            kit_title=kit_title,
            natural_steps=natural_steps,
            instrument_steps=instrument_steps
        )

    @classmethod
    def from_file(cls, file_path: Path) -> 'ProtocolSection':
        """Load protocol sections from a file."""
        protocol_text = file_path.read_text(encoding="utf-8", errors="ignore")
        return cls.from_text(protocol_text)

    def get_section_content(self, section_number: int) -> str:
        """Extract the content of a specific section number."""
        from utils.text_parser import TextParser

        if section_number <= 4:
            return TextParser.extract_section(self.natural_steps, section_number)
        else:
            return TextParser.extract_section(self.instrument_steps, section_number)

    def get_automation_steps(self) -> str:
        """Extract the automation execution steps (section [5])."""
        return self.get_section_content(5)

    def get_readout_steps(self) -> str:
        """Extract the readout/data-processing steps (section [6])."""
        return self.get_section_content(6)


@dataclass
class ProtocolStep:
    """Individual protocol step."""
    step_number: int
    description: str
    volume_ul: Optional[int] = None
    temperature_c: Optional[int] = None
    duration_min: Optional[int] = None
    repeats: int = 1
    rpm: Optional[int] = None
    reservoir_col: Optional[int] = None
    
    @classmethod
    def from_line(cls, line: str, step_number: int = 0) -> 'ProtocolStep':
        """Create a protocol step from a text line."""
        from utils.text_parser import TextParser
        
        return cls(
            step_number=step_number,
            description=line.strip(),
            volume_ul=TextParser.extract_volume_ul(line, 0) or None,
            temperature_c=TextParser.extract_temperature(line, 0) or None,
            duration_min=TextParser.extract_minutes(line, 0) or None,
            repeats=TextParser.extract_repeat_count(line),
            rpm=TextParser.extract_rpm(line, 0) or None,
            reservoir_col=TextParser.parse_reservoir_col(line, 0) or None
        )
    
    def is_automatable(self) -> bool:
        """Determine whether the step can be automated."""
        from core.exceptions import ImpossibleActionDetector
        return ImpossibleActionDetector.is_automatable(self.description)

    def is_automation_step(self) -> bool:
        """Determine whether this is an automation step."""
        from utils.text_parser import TextParser
        return TextParser.is_automation_step(self.description)

    def get_step_type(self) -> str:
        """Classify the step type."""
        import re
        desc = self.description.lower()

        if re.search(r"용액\s*제거|remove", desc):
            return "remove"
        elif re.search(r"wash\s*buffer|세척", desc):
            return "wash"
        elif re.search(r"분주|dispense", desc):
            return "dispense"
        elif re.search(r"배양|incubat", desc):
            return "incubate"
        elif re.search(r"shake|흔들", desc):
            return "shake"
        elif re.search(r"뚜껑|cap", desc):
            return "cap"
        elif re.search(r"판독|read|od", desc):
            return "readout"
        else:
            return "other"


@dataclass
class ReservoirMapping:
    """Reservoir mapping information."""
    col_number: int
    reagent_name: str
    concentration: Optional[str] = None
    volume_ul: Optional[int] = None
    notes: Optional[str] = None

    def __str__(self) -> str:
        """String representation."""
        result = f"Reservoir {self.col_number}: {self.reagent_name}"
        if self.concentration:
            result += f" ({self.concentration})"
        if self.volume_ul:
            result += f", {self.volume_ul}µL"
        if self.notes:
            result += f", {self.notes}"
        return result


@dataclass
class ProtocolMetadata:
    """Protocol metadata."""
    protocol_name: str
    kit_title: str
    total_steps: int
    automation_steps: int
    estimated_time_min: Optional[int] = None
    required_reagents: List[str] = None
    special_requirements: List[str] = None
    
    def __post_init__(self):
        if self.required_reagents is None:
            self.required_reagents = []
        if self.special_requirements is None:
            self.special_requirements = []
    
    @classmethod
    def from_protocol_section(cls, protocol: ProtocolSection, name: str) -> 'ProtocolMetadata':
        """Create metadata from a protocol section."""
        # Count the number of steps
        natural_lines = [line.strip() for line in protocol.natural_steps.splitlines() if line.strip()]
        instrument_lines = [line.strip() for line in protocol.instrument_steps.splitlines() if line.strip()]
        
        return cls(
            protocol_name=name,
            kit_title=protocol.kit_title,
            total_steps=len(natural_lines) + len(instrument_lines),
            automation_steps=len(instrument_lines)
        )