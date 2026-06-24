# bioforge/core/pdf_processor.py

import os
import time
import random
import traceback
import pandas as pd
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from .exceptions import ProtocolProcessingError, ImpossibleActionDetector


class PDFProcessor:
    """PDF protocol processor"""

    def __init__(self, config, reservoir_offset: int = 1, reservoir_guide: str = ""):
        self.config = config
        self.reservoir_offset = reservoir_offset  # Starting number for common-reagent reservoirs
        self.reservoir_guide = reservoir_guide    # Full text of the per-type reservoir loading guide
        # Llama models -> Together AI API
        import os
        LLAMA_MODELS = {
            "llama-3.2-3b":    "meta-llama/Llama-3.2-3B-Instruct",
            "llama-3.1-8b":    "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "llama-4-scout":   "kimmairobot_46cf/meta-llama/Llama-4-Scout-17B-16E-a1dbbc6e",
            "llama-3.3-70b":   "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        }
        model_name = config.LLM_MODEL
        # Strip the "-direct" suffix to obtain the actual model name
        if isinstance(model_name, str) and model_name.endswith("-direct"):
            model_name = model_name[:-len("-direct")]

        if model_name in LLAMA_MODELS:
            self.llm = ChatOpenAI(
                model=LLAMA_MODELS[model_name],
                base_url="https://api.together.xyz/v1",
                api_key=os.environ.get("LLAMA_API_KEY", ""),
                temperature=config.LLM_TEMPERATURE,
            )
        # o-series models (o1, o3, o4-mini, etc.) do not support the temperature parameter -> omit it
        elif model_name.startswith("o"):
            self.llm = ChatOpenAI(model=model_name)
        else:
            self.llm = ChatOpenAI(model=model_name, temperature=config.LLM_TEMPERATURE)
        self.chain = self._create_protocol_chain()

    def _create_protocol_chain(self):
        """Create the protocol conversion chain"""
        protocol_prompt = PromptTemplate(
            input_variables=["protocol_text", "extra_instructions", "previous_output", "reservoir_info"],
            template="""
You are a converter for PDF-based biology experiment protocols.
Derive your conclusions only from the PDF text provided below.
Never invent information or values that are not in the document; mark them as 'Not specified'.

[Additional validation feedback / revision instructions]
{extra_instructions}

[Previous version output (revision mode)]
{previous_output}
* If a previous version exists: revise only the parts pointed out in the feedback, based on that content.
  [Absolute regeneration rules]
  1. Modify only the items/reagents/values explicitly pointed out in the feedback.
  2. Keep all sections not pointed out (the rest of NATURAL and INSTRUMENT) identical to the previous version.
  3. An instruction to remove a specific reagent from the Reservoir mapping -> remove only that single line. Do not change that reagent's dispense/aspirate steps, other reagents' steps, or the overall protocol flow.
  4. The total line count before and after the revision must not change significantly. Limit deletions to the items pointed out.
  5. Do not break anything that was already correct.
* If there is no previous version (first generation): ignore this and write from scratch.

The output must contain only the following three tag blocks: <KIT_TITLE>, <NATURAL>, <INSTRUMENT>

[Plate coating/blocking (Plate Preparation) classification rules — top priority!]
*** Coating/blocking is always a [2] user manual step! Never put it in [5] automation! ***
- Capture Antibody coating (dispense + overnight/multi-hour incubation) is always recorded in [2].
- Blocking (Block Buffer dispense + incubation) is also always recorded in [2].
- Post-coating wash and post-blocking wash are also included in [2].
- Reagents used for coating/blocking (Capture Ab, Block Buffer) are not included in the [4] Reservoir mapping.
  (The user performs them directly, so no reservoir is needed.)
- * Whether or not the PDF has a "Plate Preparation" section, if a coating/blocking procedure is described, write it in [2]!
- * Regardless of the kit type (DuoSet, Development Kit, etc.), coating/blocking is always a user step!
- Examples of items that go in [2]:
  - Dilute Capture Ab in PBS -> dispense 100 µL into each well -> overnight incubation
  - Remove solution -> wash N times with Wash Buffer
  - Dispense Block Buffer 300 µL/well -> incubate 1 hour
  - Remove solution -> wash N times with Wash Buffer
- Starting point of [5] automation execution: the **first dispense** after coating/blocking is complete (standard/sample/Assay Diluent, etc.)

[NATURAL / INSTRUMENT boundary rules — very important!]
*** This system assigns Blank, Calibrant (Standard), and Sample each to a separate reservoir and dispenses them automatically! ***
  Therefore "standard/sample/Blank dispensing" can all be performed by the instrument — it is not manual!
* "Manual-only actions" means: **reagent dilution/preparation** and **coating/blocking** actions.
  e.g., "serially dilute the standard stock to prepare 7 concentration levels", "dilute the sample with Diluent", etc. -> [3]
  e.g., "Capture Ab coating -> incubation -> wash -> Blocking -> incubation -> wash", etc. -> [2]
* "Instrument-capable actions" (= INSTRUMENT [5]) means:
  - **Dispensing actions after coating/blocking**: Assay Diluent, Standard, Sample, Blank,
    Conjugate, Detection Ab, Streptavidin-HRP, Substrate, Stop Solution, Wash Buffer, etc.
    — well dispensing of every solution except coating/blocking is fully automated!
  - Washing, incubation, shaking, solution removal, etc. after coating/blocking
* Always separate reagent **dilution/preparation** from **dispensing**:
  - Dilution/preparation (e.g., "dilute Detection Ab with Diluent to 0.25 µg/mL") -> put in NATURAL [1] reagent preparation.
  - Dispensing (e.g., "dispense 100 µL of Detection Ab into each well") -> map to a reservoir and put in INSTRUMENT [5].
  - Even if the PDF states "dilute then dispense and incubate" in one sentence, the dilution goes in [1] and the dispense+incubation in [5].
* The order for determining the boundary:
  1) NATURAL covers up through reagent **dilution/preparation** ([1]~[3]).
  2) The entire coating/blocking is NATURAL [2].
  3) The mapping that puts the diluted solutions into reservoirs is [4].
  4) INSTRUMENT [5] begins from the **first dispensing action** after coating/blocking is complete!
* In [3] write only the dilution/preparation method. Never put dispensing actions ("dispense ~~ into each well") in [3]!
* Dispensing order in [5]:
  1) Assay Diluent dispense first (if such a reagent exists) — uses a common reservoir
  2) Standard/Sample/Blank dispense — supplied from the per-type reservoirs
  3) Then continue with common steps such as incubation, washing, Conjugate dispense, etc.
* Always include the Assay Diluent dispense in the Reservoir mapping [4] as well!
  e.g., "Reservoir N: Assay Diluent RD1-63 (undiluted, 50 µL/well)"
* Record standard/sample/Blank dispensing in [5] as one line per type:
  e.g., "- Dispense 50 µL of standard/sample into each well (supplied from the per-type Blank/Calibrant/Sample reservoirs)."
  * The keyword "per-type" in this line is important! sequence_builder recognizes per-type dispensing by this keyword.
* Even if the PDF states "incubate 2 hours after dispensing" in one sentence, write the dispense in [5] and the incubation on the next line of [5].
* Incubation, washing, and shaking always go in [5]! Do not put them in [3] or [4]!

[Control handling rules — very important!]
*** Even if the PDF mentions "control", do not map Control to a separate reservoir! ***
- This system has only 3 per-type reservoirs: Blank / Calibrant (Standard) / Sample.
- "Control" is not a type that exists in plate_mapper.
- If the PDF says "add standard, control, or sample", ignore the control.
  -> Process only Standard and Sample as per-type reservoirs.
- Do not assign Control to a separate reservoir!
- Do not add Control as a separate dispense line in [5] automation execution!
- It is allowed to briefly describe only the preparation method for Control in [1] reagent preparation.
Unify terminology as much as possible. (e.g., color development -> incubation)
Do not mark unknowns with '?'. Just leave that area unwritten.
Write the natural-language text in a friendly, polite tone. But do not make it overly long.
"In automation execution, put only one action per line." (i.e., one action, and the next action on the next line.)
The volume of every liquid is very important. Make sure none are omitted. **(The volume of solution removal is also very important!!)**

[Natural-language [1]~[3] parameter accuracy — very important!]
*** Every volume (µL), count (times), and time (min/hour) recorded in [1]~[3] must exactly match the original PDF text! ***
1. **No missing reagents**: record every reagent mentioned in the PDF in [1] without omission.
   - In particular, do not omit reaction-stop/detection reagents such as Stop Solution, Substrate Solution, Wash Buffer, Conjugate!
   - If the PDF has a Stop Solution (e.g., 2N H₂SO₄, 1M HCl, etc.), be sure to include it in [1]!
2. **Wash volume required**: if [2] coating/blocking includes washing, you must record the wash volume (µL) and count.
   - Search the entire PDF to find the wash volume. The wash volume may be in the Plate Preparation section as well as the Assay Procedure.
   - The wash volume used in [5] is often the same as the wash volume in [2], so apply the wash volume recorded in [5] equally to [2].
3. **Read the serial-dilution transfer volume accurately**:
   - In the PDF's dilution diagram, the "volume above the arrow" is the transfer volume.
   - If the text says "Pipette 200 µL into the remaining tubes" -> the volume going into the remaining tubes (200 µL) is the diluent,
     and the transfer volume is usually the same value (200 µL).
   - * Even if the diagram shows "50 µL Std.", this is the stock->first tube transfer amount, which may not be the transfer volume of the subsequent serial dilution!
   - Always determine the transfer volume based on the text.
Liquids are what go into the reservoir.
There is no need to mention 'Not specified'. Just leave that part unwritten. Write room temperature as room temperature!
Shaking/Agitation rules (only when present in the document)
- "gently tap", "tap the plate", "gentle mixing", etc. are converted into short shaking actions and included in [5] automation execution.
  **If a time is specified, you must include it!**
  e.g., "gently tap the plate" -> "short shaking (mixing)"
  e.g., "tap for ~1 minute" -> "short shaking 1 min (mixing)"
  e.g., "gently tap for 30 seconds" -> "short shaking 30 sec (mixing)"
- For Shaker actions, you must record **rpm and time together on one line**. Do not split "move to Shaker", "SHAKE_START", "WAIT_MOTION"!
  e.g., "Shaker 500 rpm, incubate 120 min." (OK — one line)
  e.g., "Move to Shaker.\n SHAKE_START 500 rpm.\n WAIT_MOTION 120 min." (NO — do not split into 3 lines)
- Room-temperature (benchtop) incubation, without a shaker/incubator: record as "room temperature (benchtop) incubation 30 min" on **one line**.
  * Never split TRANS, TEMPPLATE_ON, WAIT_MOTION into individual lines!
  e.g., "room temperature (benchtop) incubation 30 min" (OK — one line)
  e.g., "TRANS 1->3\n TEMPPLATE_ON 20\n WAIT_MOTION 1800\n TRANS 3->1" (NO — no individual lines)

[Substrate color-development incubation rules — very important!]
*** After dispensing the Substrate, you must record an incubation step! Never omit it! ***
- The order substrate dispense -> incubation -> Stop Solution dispense must always be followed.
  e.g., "- Dispense Reservoir 3 (Substrate Solution) 100 µL/well."
      "- Room temperature (benchtop) incubation 30 min."        <- never drop this line!
      "- Dispense Reservoir 4 (Stop Solution) 100 µL/well."
- Substrate types: ABTS, TMB, pNPP, OPD, Substrate Solution, Color Reagent, etc.
- If the PDF states only "Incubate at room temperature for color development" or "color development reaction"
  and **no specific time is given**, use the following default time depending on the substrate type:
  - ABTS -> room temperature (benchtop) incubation 25 min (1500 sec)
  - TMB  -> room temperature (benchtop) incubation 20 min (1200 sec)
  - pNPP -> room temperature (benchtop) incubation 30 min (1800 sec)
  - OPD  -> room temperature (benchtop) incubation 20 min (1200 sec)
  - Substrate type unknown -> room temperature (benchtop) incubation 20 min (default)
- [WARN] A "1 min" or "60 sec" color-development incubation is almost certainly a wrong value. Substrate color development takes at least 10 min!
- If a time is specified in the PDF (e.g., "15 min", "30 min"), always use that time!

[Aspirate/Wash pattern — very important!]
* The patterns "Aspirate each well and wash", "aspiration/wash", "Repeat the aspiration/wash" in the PDF
  must be recorded **split into 2 steps**! Never write only the wash!
  Step 1: (solution removal) remove the previous-step solution N µL/well
  Step 2: wash N times with Reservoir X (Wash Buffer), each N µL/well
  e.g., PDF: "Aspirate each well and wash 5 times with 400µL Wash Buffer"
      -> "- (solution removal) remove the previous-step solution 100 µL/well"
      -> "- Wash 5 times with Reservoir N (Wash Buffer), each 400 µL/well." (N is the number assigned to Wash Buffer in the [4] Reservoir mapping)
* Solution removal volume = the sum of the volumes dispensed into the well in the previous step.
  e.g., Assay Diluent 50µL + Sample 50µL = remove 100µL
  e.g., Conjugate 100µL = remove 100µL
  e.g., Assay Diluent 100µL + Sample 100µL = remove 200µL

[Preventing PDF page-break duplication — very important!]
- In a PDF, the same step may be extracted twice across a **page break**.
  e.g., substrate dispense + color-development incubation repeated with identical content at the end of one page and the start of the next.
- If the **dispense+incubation of the same reagent repeats twice in a row**, record it only once.
- Decision criterion: if the same reagent name, same volume, and same incubation condition appear consecutively, it is a duplicate.

[Format/flow mandatory rules]
- No text outside the tags.
- You must write in the following research-flow order:
  (NATURAL) [1] common solution/reagent preparation -> [2] plate coating/blocking (if absent, 'Not specified') ->
             [3] standard/sample dilution/preparation (do not include dispensing!) -> [4] Reservoir mapping (user, just before automation)
  (INSTRUMENT) [5] automation execution (instrument) -> [6] readout/data processing/QC
- Each bullet is a single sentence of 'what, how'. Units are µL, min, "*times*". If a value is absent, '?' or 'Not specified'.
- The [4] Reservoir mapping lists **both per-type and common reagents** in numeric order.
*** There is no limit on the number of reservoirs! 12, 15, 17, 20 or more are all possible! ***
  - However many reagents the protocol needs, assign each a separate reservoir number.
  - Reservoir 13, 14, 15, 16, 17, 18, 19, 20 and beyond may be used freely!
  - Never merge two reagents into the same reservoir to reduce the reservoir count!
  - Never omit a reagent just because the reservoir number exceeds 12!
{reservoir_info}
*** You must follow the reservoir-numbering rules above!
  - If per-type Reservoir (Blank/Calibrant/Sample) numbers are provided in [4] -> use those numbers, but **write the description based on the PDF content**!
    (e.g., Blank -> "Diluent (0 pg/mL)", Standard S1 -> use the actual concentration value extracted from the PDF)
  - Common reagents (Wash, Assay Diluent, Conjugate, Substrate, Stop, etc.) are assigned consecutively starting from the specified start number.
  - The per-type dispensing in [5] must be written as **exactly one line per reservoir**!
    [X] Prohibited 1: "dispense 100 µL of standard/control/sample into each well" <- must not be lumped into one line!
    [X] Prohibited 2: do not repeat the same reservoir over multiple lines! (replicate/triplicate is handled automatically by the system via well coordinates)
    [OK] Correct example:
        "- Dispense Reservoir 1 (Blank, 0 pg/mL) into the designated wells, 100 µL/well."   <- one line only!
        "- Dispense Reservoir 2 (Standard S1, 200 pg/mL) into the designated wells, 100 µL/well."  <- one line only!
        "- Dispense Reservoir 3 (Standard S2) into the designated wells, 100 µL/well."  <- one line only!
        ...
  - If there is an Assay Diluent, be sure to include it in the common reagents!
- The first part of automation execution ([5]) starts from the dispensing steps!:
  1) Assay Diluent dispense (if present): "Dispense Reservoir N (Assay Diluent) 50 µL/well."
  2) Per-type dispense: one line each per reservoir (see example above). Never lump into one line!
  3) Then the common steps (incubation, washing, Conjugate, etc.)
- # Per-row variable-volume dispensing (Bradford, dilution series, etc.):
  When the protocol has a dilution table that puts a different volume in each row (e.g., Standard 1: 15µL, Standard 2: 30µL...):
  Write one line per row in the format "Dispense Reservoir N (reagent name) Y µL into Row X."
  Repeat the same reservoir for each row, varying only the volume.
  * Write only the row number! Never write Y coordinates / column coordinates (Y=1, Y=5, etc.)! Column distribution is handled automatically by the system.
  * For a uniform-volume reagent (e.g., Bradford Reagent), do not repeat the same volume per row; write it as a single line "dispense X µL into all wells".
  * Dispensing order: complete all rows of one reservoir first, then move to the next reservoir! Do not alternate reservoirs per row!
    [X] Prohibited: Res1 Row1 -> Res2 Row1 -> Res1 Row2 -> Res2 Row2 (alternating)
    [OK] Correct: Res1 Row1 -> Res1 Row2 -> ... -> Res1 Row12 -> Res2 Row1 -> Res2 Row2 -> ... (grouped by reservoir)
  e.g., - Dispense Reservoir 1 (BSA) 20 µL into Row 1.
      - Dispense Reservoir 1 (BSA) 30 µL into Row 2.
      ...
      - Dispense Reservoir 1 (BSA) 130 µL into Row 12.
      - Dispense Reservoir 2 (Water) 110 µL into Row 1.
      - Dispense Reservoir 2 (Water) 100 µL into Row 2.
      ...
      - Dispense Reservoir 3 (Bradford Reagent) 170 µL into all wells.
  Note: do not mix this with ELISA-style per-type dispensing ("into the designated wells").
- Automation execution must include the (solution removal) step of each cycle (the removal volume equals what was dispensed before!):
  e.g., "(solution removal) remove the previous-step solution (if the method is not specified, 'Not specified')"
  e.g., 'Aspirate and wash' means Wash N times after the solution removal
- In automation execution, reservoir notation is in the form "Reservoir {{number}} ({{reagent name}})" and must match the NATURAL mapping.
- For Wash, do not split into dispense/removal; write it as Wash, e.g., wash ? times with Reservoir N (Wash Buffer), each ? µL/well. (N is the number assigned to Wash in [4]. ""Volume is important! If a wash volume has been mentioned anywhere in the content, keep that volume."")


[Input document]
{protocol_text}

[Output — return only the three tags]
<KIT_TITLE>Not specified</KIT_TITLE>

<NATURAL>
[1] Common solution/reagent preparation (user)
Not specified

[2] Plate coating/blocking (user)
Not specified

[3] Standard dilution/preparation / sample preparation (user)
(* Record only dilution/preparation! Do not put dispensing actions like "dispense ~~ into each well" here; put them in [5]!)
Not specified

[4] Reservoir mapping (user, just before automation)
(* Record the per-type reservoirs first, with descriptions based on the PDF content, then record the common-reagent reservoirs!)
Not specified

</NATURAL>

<INSTRUMENT>
[5] Automation execution (instrument)
(* Start from the first dispense! Assay Diluent dispense -> per-type (standard/sample) dispense, one line per reservoir -> incubation -> washing -> ...)
Not specified

[6] Readout/data processing / QC
Not specified
(The first line starts with the readout method:
  e.g., "O.D. readout: 450 nm)
</INSTRUMENT>
""".strip()
        )
        return protocol_prompt | self.llm
        
    def extract_tag(self, text: str, tag: str) -> str:
        """Extract the content of a tag from text"""
        import re
        m = re.search(rf"<{tag}>(.+?)</{tag}>", text, flags=re.S)
        return m.group(1).strip() if m else ""

    def read_excel_to_text(self, path: Path) -> str:
        """Convert an Excel file to text"""
        if not path.exists():
            return ""
        try:
            df = pd.read_excel(path)
            return df.to_string(index=False)
        except Exception as e:
            return f"(failed to load Excel: {e})"

    def process_single_pdf(self, pdf_path: Path):
        """Process a single PDF file"""
        base = pdf_path.stem
        print(f"\n[INFO] Processing started: {base}")

        # Load PDF -> combine text
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE, 
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        protocol_text = "\n".join(d.page_content for d in chunks)

        # LLM call (retry/backoff)
        resp = self._invoke_with_retry(protocol_text, extra_instructions="", previous_output="")
        result_text = resp.content

        kit_title = self.extract_tag(result_text, "KIT_TITLE") or "ELISA Protocol (Not specified)"
        natural_body = self.extract_tag(result_text, "NATURAL") or "Not specified"
        inst_body = self.extract_tag(result_text, "INSTRUMENT") or "Not specified"

        # Detect actions that cannot be automated
        detected = ImpossibleActionDetector.detect(natural_body)
        if detected:
            warning = "\n\n[WARN] The following steps cannot be performed by the automation instrument.\n"
            warning += "\n".join(f"- {kw}" for kw in detected)
            natural_body += warning

        # Save
        out_path = self.config.OUTPUT_DIR / f"{base}{self.config.STRUCTURED_SUFFIX}"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(
                f"<KIT_TITLE>{kit_title}</KIT_TITLE>\n\n"
                f"<NATURAL>\n{natural_body}\n</NATURAL>\n\n"
                f"<INSTRUMENT>\n{inst_body}\n</INSTRUMENT>\n"
            )

        print(f"[OK] KIT_TITLE={kit_title}")
        print(f" └─ Saved: {out_path}")

    def _invoke_with_retry(self, protocol_text: str, extra_instructions: str = "", previous_output: str = ""):
        # Build the guidance text for the common-reagent reservoir start number
        if self.reservoir_offset > 1 and self.reservoir_guide:
            # Extract the per-type assignment structure from reservoir_guide
            import re as _re
            type_lines = []
            calibrant_count = 0  # Total number of assigned Calibrant reservoirs
            for line in self.reservoir_guide.split('\n'):
                m = _re.match(r'\s*Reservoir\s+Col\s+(\d+):\s*(\S+)', line)
                if m:
                    col_num = m.group(1)
                    group_key = m.group(2).strip()  # e.g. "BLANK_0", "CALIBRANT_0", "SAMPLE_0"
                    # Extract the type name from group_key
                    if 'BLANK' in group_key.upper():
                        type_lines.append(f"  - Reservoir {col_num}: Blank (0 concentration / Diluent) — write an appropriate description based on the PDF content")
                    elif 'CALIBRANT' in group_key.upper():
                        calibrant_count += 1
                        idx_m = _re.search(r'_(\d+)$', group_key)
                        idx = int(idx_m.group(1)) + 1 if idx_m else '?'
                        type_lines.append(f"  - Reservoir {col_num}: Standard/Calibrant S{idx} — write the description according to the PDF standard-curve concentration (if the PDF has no such concentration, mark it as 'User-defined Calibrant')")
                    elif 'SAMPLE' in group_key.upper():
                        idx_m = _re.search(r'_(\d+)$', group_key)
                        idx = int(idx_m.group(1)) + 1 if idx_m else '?'
                        type_lines.append(f"  - Reservoir {col_num}: Sample {idx} — sample")
            type_guide = "\n".join(type_lines) if type_lines else ""
            calibrant_count_info = f"  * Total number of Calibrant reservoirs assigned above: {calibrant_count}\n" if calibrant_count else ""

            reservoir_info = (
                f"*** [Reservoir numbering rules — dynamic assignment] ***\n"
                f"  This system assigns Blank/Calibrant(Standard)/Sample each to a separate reservoir.\n"
                f"  * No limit on the number of reservoirs! 13, 14, 15, 17, 20 and beyond may be used freely!\n"
                f"  * Never merge or omit reagents to reduce the reservoir count!\n"
                f"  When placing wells in the UI, the per-type reservoirs were automatically assigned as follows:\n"
                f"{type_guide}\n"
                f"{calibrant_count_info}"
                f"  -> Reservoirs 1~{self.reservoir_offset - 1} are for per-type dispensing (Blank/Calibrant/Sample).\n"
                f"  -> In [4], record the per-type reservoirs above first, with descriptions based on the PDF content!\n"
                f"    e.g., Blank -> 0 pg/mL (Diluent), Standard S1 -> highest standard-curve concentration, S2 -> next concentration ...\n"
                f"\n"
                f"  ** Calibrant count-mismatch handling rules (very important!) **\n"
                f"  Compare the number of Standard points specified in the PDF with the number of Calibrant reservoirs assigned above ({calibrant_count}):\n"
                f"  (1) PDF standard count = Calibrant reservoir count -> map all from the PDF (record concentrations)\n"
                f"  (2) PDF standard count < Calibrant reservoir count -> assign the PDF standards first in order,\n"
                f"      and mark the remaining reservoirs as 'User-defined Calibrant N (User-defined, concentration not specified)'.\n"
                f"      *** Never write only 'Not specified'! Always mark it as 'User-defined Calibrant'! ***\n"
                f"      *** User-defined Calibrants must also have a dispense line written in [5]! Do not omit them! ***\n"
                f"      e.g., if the PDF has 7 Standards and 9 Calibrant reservoirs are assigned:\n"
                f"        [4]: - Reservoir 2: Standard S1 (1000 pg/mL).  <- PDF-based\n"
                f"             - ...\n"
                f"             - Reservoir 8: Standard S7 (15.6 pg/mL).  <- PDF-based\n"
                f"             - Reservoir 9: User-defined Calibrant 8 (User-defined, concentration not specified).\n"
                f"             - Reservoir 10: User-defined Calibrant 9 (User-defined, concentration not specified).\n"
                f"        [5]: - Dispense Reservoir 9 (User-defined Calibrant 8) into the designated wells, 100 µL/well.  <- dispense line required!\n"
                f"             - Dispense Reservoir 10 (User-defined Calibrant 9) into the designated wells, 100 µL/well. <- dispense line required!\n"
                f"  (3) PDF standard count > Calibrant reservoir count -> map only as many as the assigned reservoirs, in PDF order\n"
                f"\n"
                f"  -> Common reagents (Wash Buffer, Detection Ab, Streptavidin-HRP, Substrate, Stop, etc.)\n"
                f"    must be assigned consecutively starting from Reservoir {self.reservoir_offset}!\n"
                f"  -> [4] example:\n"
                f"    - Reservoir 1: Blank (Diluent, 0 pg/mL).\n"
                f"    - Reservoir 2: Standard S1 (200 pg/mL).\n"
                f"    - ...\n"
                f"    - Reservoir {self.reservoir_offset}: Wash Buffer (0.05% Tween-20 in PBS, for washing).\n"
                f"    - Reservoir {self.reservoir_offset + 1}: Detection Antibody 0.25 µg/mL in Diluent (100 µL/well).\n"
            )
        elif self.reservoir_offset > 1:
            reservoir_info = (
                f"*** [Reservoir numbering rules] ***\n"
                f"  Reservoirs 1~{self.reservoir_offset - 1} are already assigned to per-type dispensing (Blank/Calibrant/Sample).\n"
                f"  Common reagents must be assigned starting from Reservoir {self.reservoir_offset}!\n"
                f"  * No limit on the number of reservoirs! 13 and beyond may be used freely!\n"
            )
        else:
            reservoir_info = (
                "  * No limit on the number of reservoirs! Assign as many as needed, freely!\n"
                "  e.g., Reservoir 1: Wash Buffer (1X, for washing)\n"
                "  e.g., Reservoir 2: Assay Diluent RD1-63 (undiluted, 50 µL/well)\n"
                "  e.g., Reservoir 3: Conjugate (undiluted, 100 µL/well)"
            )
        for attempt in range(self.config.MAX_RETRIES):
            try:
                return self.chain.invoke({
                    "protocol_text": protocol_text,
                    "extra_instructions": extra_instructions or "",
                    "previous_output": previous_output or "(first generation — no previous version)",
                    "reservoir_info": reservoir_info,
                })
            except Exception as e:
                msg = str(e)
                if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
                    print("[FAIL] OpenAI quota/billing limit exhausted. Check billing/limits.")
                    raise
                wait = min(60, (2 ** attempt)) + random.uniform(0, 1.0)
                print(f"[Retry {attempt+1}] error: {msg[:300]}")
                print(f"  -> waiting {wait:.1f}s before retry")
                time.sleep(wait)
        raise ProtocolProcessingError(f"Maximum retry count ({self.config.MAX_RETRIES}) exceeded")

    def process_all_pdfs(self):
        """Process all PDFs in the folder"""
        self.config.ensure_output_dirs()

        pdf_paths = sorted(self.config.INPUT_DIR.glob("*.pdf"))
        if not pdf_paths:
            print(f"[WARN] No PDFs found: {self.config.INPUT_DIR}")
            return

        skip_done = False
        for p in pdf_paths:
            base = p.stem
            out_path = self.config.OUTPUT_DIR / f"{base}{self.config.STRUCTURED_SUFFIX}"

            if skip_done and out_path.exists():
                print(f"[SKIP] Already processed: {out_path}")
                continue

            try:
                self.process_single_pdf(p)
            except Exception:
                print(f"[FAIL] Processing failed: {p}")
                traceback.print_exc()

            # Ease consecutive calls
            time.sleep(self.config.COOLDOWN_SEC)