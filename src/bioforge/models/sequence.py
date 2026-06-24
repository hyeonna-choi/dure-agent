# bioforge/models/sequence.py

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd


@dataclass
class SequenceCommand:
    """Sequence command data class."""
    handler: str
    command: str
    parameters: str

    @classmethod
    def create(cls, handler: int, command: str, parameters: str) -> 'SequenceCommand':
        """Create a sequence command."""
        return cls(
            handler=str(handler),
            command=command,
            parameters=parameters
        )

    def to_dict(self) -> Dict[str, str]:
        """Convert to a dictionary."""
        return {
            "Handler": self.handler,
            "Command": self.command,
            "Input Parameters": self.parameters
        }

    def is_valid(self, allowed_commands: List[str]) -> bool:
        """Check whether the command is in the allowed command list."""
        return self.command in allowed_commands

    def get_command_type(self) -> str:
        """Classify the command type."""
        if "ASPIRATE" in self.command:
            return "aspirate"
        elif "DISPENSE" in self.command:
            return "dispense"
        elif "INSTALL" in self.command:
            return "install_tip"
        elif "EJECT" in self.command:
            return "eject_tip"
        elif "GET" in self.command or "PUT" in self.command:
            return "transport"
        elif "CAP" in self.command:
            return "cap_control"
        elif "SHAKE" in self.command:
            return "shake"
        elif "WAIT" in self.command:
            return "wait"
        elif "TEMP" in self.command:
            return "temperature"
        elif "ANALYZER" in self.command:
            return "analyzer"
        else:
            return "other"


class TipAllocator:
    """4-channel/1-channel tip coordinate allocator.

    Coordinates: (X, Y)  X=row (top->bottom, 1-12)  Y=column (right->left, 1-8)
    When Y is exhausted, X is incremented.

    4ch: consumes 4 consecutive cells (Y, Y+1, Y+2, Y+3). The INSTALL parameter is the starting Y.
    1ch: consumes 1 empty cell. Searched in order.
    """

    MAX_X = 12
    MAX_Y = 8

    def __init__(self, tip_slot: int):
        self.tip_slot = tip_slot
        self.used: set = set()

    def next_4ch(self) -> Tuple[int, int, int]:
        """4-channel: only a starting Y of 1 or 5 is valid (1->1,2,3,4 / 5->5,6,7,8)."""
        for x in range(1, self.MAX_X + 1):
            for y in (1, 5):
                if all((x, y + d) not in self.used for d in range(4)):
                    for d in range(4):
                        self.used.add((x, y + d))
                    return (self.tip_slot, x, y)
        return (self.tip_slot, self.MAX_X, 5)

    def next_1ch(self) -> Tuple[int, int, int]:
        """1-channel: return the next empty cell."""
        for x in range(1, self.MAX_X + 1):
            for y in range(1, self.MAX_Y + 1):
                if (x, y) not in self.used:
                    self.used.add((x, y))
                    return (self.tip_slot, x, y)
        return (self.tip_slot, self.MAX_X, self.MAX_Y)

    def next_install_param(self) -> Tuple[int, int, int]:
        """Backward compatibility: behaves as 4ch when the channel is not specified."""
        return self.next_4ch()

    def reset(self):
        """Reset the allocator."""
        self.used.clear()

    def get_usage_summary(self) -> Dict[str, int]:
        """Summarize tip usage."""
        return {
            "total_tips_used": len(self.used),
            "columns_used": len(set(pos[0] for pos in self.used)),
            "current_column": self.col,
            "current_index": self.idx
        }


@dataclass
class SequenceBlock:
    """Sequence block (a group of related commands)."""
    block_type: str
    commands: List[SequenceCommand]
    description: str = ""

    def add_command(self, command: SequenceCommand):
        """Add a command."""
        self.commands.append(command)

    def to_dict_list(self) -> List[Dict[str, str]]:
        """Convert to a list of dictionaries."""
        return [cmd.to_dict() for cmd in self.commands]

    def get_command_count(self) -> int:
        """Return the number of commands."""
        return len(self.commands)

    def validate(self, allowed_commands: List[str]) -> List[str]:
        """Return the invalid commands."""
        invalid_commands = []
        for cmd in self.commands:
            if not cmd.is_valid(allowed_commands):
                invalid_commands.append(cmd.command)
        return invalid_commands


@dataclass
class SequenceBuilder:
    """Sequence build result."""
    commands: List[SequenceCommand]
    blocks: List[SequenceBlock]
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        # Compute statistics automatically
        self.metadata.update({
            "total_commands": len(self.commands),
            "total_blocks": len(self.blocks),
            "command_types": self._count_command_types(),
            "handlers_used": self._count_handlers()
        })

    def _count_command_types(self) -> Dict[str, int]:
        """Count commands by command type."""
        type_counts = {}
        for cmd in self.commands:
            cmd_type = cmd.get_command_type()
            type_counts[cmd_type] = type_counts.get(cmd_type, 0) + 1
        return type_counts

    def _count_handlers(self) -> Dict[str, int]:
        """Count commands by handler."""
        handler_counts = {}
        for cmd in self.commands:
            handler = cmd.handler
            handler_counts[handler] = handler_counts.get(handler, 0) + 1
        return handler_counts

    def add_block(self, block: SequenceBlock):
        """Add a block."""
        self.blocks.append(block)
        self.commands.extend(block.commands)
        # Update metadata
        self.__post_init__()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame."""
        data = [cmd.to_dict() for cmd in self.commands]
        return pd.DataFrame(data, columns=["Handler", "Command", "Input Parameters"])

    def save_to_excel(self, output_path):
        """Save to an Excel file."""
        df = self.to_dataframe()
        df.to_excel(output_path, index=False)

    def validate_sequence(self, allowed_commands: List[str]) -> Dict[str, List[str]]:
        """Validate the sequence."""
        validation_result = {
            "invalid_commands": [],
            "missing_commands": [],
            "warnings": []
        }

        # Check individual commands
        for cmd in self.commands:
            if not cmd.is_valid(allowed_commands):
                validation_result["invalid_commands"].append(cmd.command)

        # Check each block
        for block in self.blocks:
            invalid_in_block = block.validate(allowed_commands)
            validation_result["invalid_commands"].extend(invalid_in_block)

        # Remove duplicates
        validation_result["invalid_commands"] = list(set(validation_result["invalid_commands"]))

        return validation_result

    def get_summary(self) -> str:
        """Return summary information for the sequence."""
        summary = f"Sequence summary:\n"
        summary += f"- Total commands: {self.metadata['total_commands']}\n"
        summary += f"- Total blocks: {self.metadata['total_blocks']}\n"
        summary += f"- Count by command type:\n"

        for cmd_type, count in self.metadata['command_types'].items():
            summary += f"  * {cmd_type}: {count}\n"

        summary += f"- Command count by handler:\n"
        for handler, count in self.metadata['handlers_used'].items():
            summary += f"  * Handler {handler}: {count}\n"

        return summary


@dataclass
class DeckSlotMapping:
    """Deck slot mapping information."""
    slot_number: int
    slot_name: str
    description: str
    is_active: bool = True

    def __str__(self) -> str:
        status = "active" if self.is_active else "inactive"
        return f"Slot {self.slot_number}: {self.slot_name} ({self.description}) - {status}"


@dataclass
class SequenceConfig:
    """Sequence generation configuration."""
    deck_slots: Dict[str, int]
    allowed_commands: List[str]
    default_speeds: Dict[str, int]
    default_volumes: Dict[str, int]
    tip_allocator: TipAllocator

    def get_slot(self, slot_name: str) -> int:
        """Return the slot number."""
        return self.deck_slots.get(slot_name, 1)

    def is_command_allowed(self, command: str) -> bool:
        """Check whether the command is allowed."""
        return command in self.allowed_commands

    def get_default_speed(self, operation: str) -> int:
        """Return the default speed."""
        return self.default_speeds.get(operation, 120)

    def get_default_volume(self, operation: str) -> int:
        """Return the default volume."""
        return self.default_volumes.get(operation, 100)