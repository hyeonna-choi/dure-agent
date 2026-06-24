# bioforge/core/direct_sequence_generator.py
"""
LLM mapping sequence generator: Structured Output -> LLM -> Command Sequence (.xlsx) generated directly.
Takes the structured output produced by pdf_processor.py as input and converts it
directly into instrument commands using an LLM, without using the rule-based sequence_builder.
For comparison experiments -- to measure the performance difference versus the rule-based pipeline.
"""

import re
import time
import traceback
import pandas as pd
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# -- Source file paths (read at runtime) --
_CORE_DIR = Path(__file__).parent
_SEQUENCE_BUILDER_PATH = _CORE_DIR / "sequence_builder.py"


def _read_source_file(path: Path) -> str:
    """Read a source file as text"""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"(failed to read file: {path} -- {e})"


SYSTEM_PROMPT_TEMPLATE = r"""You are an expert who converts the structured output (Structured Protocol) of an ELISA protocol into a Command Sequence for the BioForge instrument.

The input is the structured output already produced by pdf_processor.py.
Read this structured output and, referring to the rules in sequence_builder.py below,
output the final Command Sequence in TSV format.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Reference file] sequence_builder.py — sequence conversion rules (Rule-based Engine)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This file contains all the logic for converting structured natural language into actual instrument commands.
In particular, follow these exactly:
- TipAllocator: 4ch (Y=1,5 only) / 1ch (sequential) coordinate allocation + shared rack
- Wash block: Mixed approach (4ch->1ch alternating, per round), tip change for every group/well
- Remove block: tip change for every group/well (aspirate from plate -> waste)
- Dispense block: tip reuse OK (does not touch the well)
- Incubation block: TRANS 1->3 -> TEMPPLATE_ON -> WAIT_MOTION -> TRANS 3->1
- Shake block: TRANS 1->6 -> SHAKE_START -> TRANS 6->1
- Initialization: TEMPPLATE_ON -> GET -> PUT -> CAP_OPEN
- Finalization (OD readout): CAP_CLOSE -> GET -> PUT -> ANALYZER sequence
- Handler codes: 73=pipette/temp, 74=transport, 75=analyzer

```python
{sequence_builder_source}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Output rules]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Output format: TSV inside a ``` code block (Handler\tCommand\tInput Parameters)
2. Never include explanatory text, comments, or blank lines. Output TSV data only.
3. Output every row without omission. Never use "..." or "(repeat)".
4. The sequence may be 200 to 1500 rows. Output them all.
5. Handler: integer (73, 74, 75)
6. Command: a command starting with A#
7. Input Parameters: space-separated parameters

Available commands:
A#GET, A#PUT, A#TEMPPLATE_ON, A#TEMPPLATE_OFF,
A#CAP_OPEN, A#CAP_CLOSE,
A#ADP_SELECT_1_CHANNEL, A#ADP_SELECT_4_CHANNEL,
A#ADP_INSTALL_PIPETTE, A#ADP_EJECT_PIPETTE,
A#ADP_ASPIRATE_FROM_PLATE, A#ADP_DISPENSE_INTO_PLATE,
A#TRANS, A#SHAKE_START, A#WAIT_MOTION,
A#ANALYZER_OPEN, A#ANALYZER_CLOSE, A#ANALYZER_START

Slot numbers (fixed):
plate_slot=1, magnet_slot=2, hotplate_slot=3, tip_slot=4,
reservoir_slot=5, shaker_slot=6, tip_waste_slot=7, waste_slot=8,
cap_parking_slot=9
"""


class DirectSequenceGenerator:
    """LLM-based generator that produces a Command Sequence directly from Structured Output"""

    def __init__(self, config, wellplate_text: str = "",
                 reservoir_offset: int = 1, reservoir_guide: str = ""):
        self.config = config
        self.wellplate_text = wellplate_text
        self.reservoir_offset = reservoir_offset
        self.reservoir_guide = reservoir_guide

        model_name = config.LLM_MODEL
        # Strip the "-direct" suffix to obtain the actual model name
        if isinstance(model_name, str) and model_name.endswith("-direct"):
            model_name = model_name[:-len("-direct")]

        # Llama models -> Together AI API
        LLAMA_MODELS = {
            "llama-3.2-3b":    "meta-llama/Llama-3.2-3B-Instruct",
            "llama-3.1-8b":    "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "llama-4-scout":   "kimmairobot_46cf/meta-llama/Llama-4-Scout-17B-16E-a1dbbc6e",
            "llama-3.3-70b":   "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        }

        import os
        if model_name in LLAMA_MODELS:
            self.llm = ChatOpenAI(
                model=LLAMA_MODELS[model_name],
                base_url="https://api.together.xyz/v1",
                api_key=os.environ.get("LLAMA_API_KEY", ""),
                temperature=0,
            )
            print(f"[DirectSequenceGenerator] Together AI: {LLAMA_MODELS[model_name]}")
        elif model_name.startswith("o"):
            self.llm = ChatOpenAI(model=model_name)
        else:
            self.llm = ChatOpenAI(model=model_name, temperature=0)

        # Read only the sequence_builder source (once)
        self._sequence_builder_source = _read_source_file(_SEQUENCE_BUILDER_PATH)

        # Assemble the system prompt
        self._system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            sequence_builder_source=self._sequence_builder_source,
        )
        print(f"[DirectSequenceGenerator] system prompt length: {len(self._system_prompt)} chars "
              f"(~{len(self._system_prompt) // 4} tokens)")

    def _build_reservoir_info(self) -> str:
        """Build dynamic reservoir information"""
        if self.reservoir_offset <= 1 or not self.reservoir_guide:
            return ""

        type_lines = []
        for line in self.reservoir_guide.split('\n'):
            m = re.match(r'\s*Reservoir\s+Col\s+(\d+):\s*(\S+)', line)
            if m:
                col_num = m.group(1)
                group_key = m.group(2).strip()
                if 'BLANK' in group_key.upper():
                    type_lines.append(f"  - Reservoir {col_num}: Blank (0 concentration / Diluent)")
                elif 'CALIBRANT' in group_key.upper():
                    idx_m = re.search(r'_(\d+)$', group_key)
                    idx = int(idx_m.group(1)) + 1 if idx_m else '?'
                    type_lines.append(f"  - Reservoir {col_num}: Standard/Calibrant S{idx}")
                elif 'SAMPLE' in group_key.upper():
                    idx_m = re.search(r'_(\d+)$', group_key)
                    idx = int(idx_m.group(1)) + 1 if idx_m else '?'
                    type_lines.append(f"  - Reservoir {col_num}: Sample {idx}")

        type_guide = "\n".join(type_lines) if type_lines else "(none)"

        return (
            f"[Reservoir assignment information]\n"
            f"  Per-type reservoirs (Col 1~{self.reservoir_offset - 1}): assigned automatically by the system\n"
            f"{type_guide}\n"
            f"  Common reagents: assigned in order starting from Reservoir {self.reservoir_offset}\n"
            f"  (Wash Buffer, Assay Diluent, Detection Ab, Conjugate, Substrate, Stop, etc.)\n"
        )

    def generate(self, structured_text: str) -> pd.DataFrame:
        """Generate a Command Sequence DataFrame from Structured Output text"""
        reservoir_info = self._build_reservoir_info()

        user_parts = []
        user_parts.append(f"[WELLPLATE COORDINATES]\n{self.wellplate_text}")
        user_parts.append(f"[RESERVOIR INFO]\n{reservoir_info}")
        user_parts.append(f"[STRUCTURED PROTOCOL]\n{structured_text}")
        user_parts.append(
            "[INSTRUCTIONS]\n"
            "Analyze the structured protocol above and generate a Command Sequence.\n"
            "- Always start with the initialization sequence (TEMPPLATE_ON->GET->PUT->CAP_OPEN)\n"
            "- Convert each step of the <INSTRUMENT> section into instrument commands in order\n"
            "- Convert each step into the appropriate block (dispense/wash/remove/incubation/shaker)\n"
            "- Track tip coordinates precisely (follow the 4ch/1ch rules; do not reuse previously used coordinates)\n"
            "- For well coordinates, use 4ch_groups/1ch_wells from WELLPLATE COORDINATES\n"
            "- End with the finalization sequence (OD block or CAP_CLOSE+GET+PUT)\n"
            "- Output in TSV format inside a ``` code block (Handler\\tCommand\\tInput Parameters)\n"
            "- Output every row without omission. Never omit anything. Do not use '...' or '(repeat)'."
        )

        user_content = "\n\n".join(user_parts)

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=user_content),
        ]

        response_text = self._invoke_with_retry(messages)
        df = self._parse_llm_output(response_text)
        return df

    def _invoke_with_retry(self, messages, max_retries: int = 3) -> str:
        """LLM call + retry"""
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[DirectSequenceGenerator] LLM call attempt {attempt}/{max_retries}")
                response = self.llm.invoke(messages)
                text = response.content if hasattr(response, 'content') else str(response)
                print(f"[DirectSequenceGenerator] response length: {len(text)} chars")
                return text
            except Exception as e:
                print(f"[DirectSequenceGenerator] call failed (attempt {attempt}): {e}")
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"[DirectSequenceGenerator] retrying after {wait}s...")
                    time.sleep(wait)
                else:
                    traceback.print_exc()
                    raise

    def _parse_llm_output(self, text: str) -> pd.DataFrame:
        """Parse the LLM response text into a DataFrame"""
        code_block_match = re.search(r'```(?:tsv|csv|text|plain)?\s*\n?(.*?)```', text, flags=re.S)
        if code_block_match:
            content = code_block_match.group(1).strip()
        else:
            content = text.strip()

        rows = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('#') or line.startswith('//') or line.startswith('---'):
                continue
            if 'Handler' in line and 'Command' in line:
                continue

            if '\t' in line:
                parts = line.split('\t')
            elif '|' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
            else:
                parts = re.split(r'\s{2,}', line, maxsplit=2)

            if len(parts) >= 3:
                handler = parts[0].strip()
                command = parts[1].strip()
                params = parts[2].strip() if len(parts) > 2 else ""
                try:
                    int(handler)
                except ValueError:
                    continue
                if not command.startswith('A#'):
                    continue
                rows.append({
                    "Handler": handler,
                    "Command": command,
                    "Input Parameters": params,
                })
            elif len(parts) == 2:
                handler = parts[0].strip()
                command = parts[1].strip()
                try:
                    int(handler)
                except ValueError:
                    continue
                if command.startswith('A#'):
                    rows.append({
                        "Handler": handler,
                        "Command": command,
                        "Input Parameters": "",
                    })

        if not rows:
            print(f"[DirectSequenceGenerator] WARNING: parsing produced 0 rows. Partial raw response:\n{text[:500]}")

        df = pd.DataFrame(rows, columns=["Handler", "Command", "Input Parameters"])
        print(f"[DirectSequenceGenerator] parsing complete: {len(df)} rows")
        return df
