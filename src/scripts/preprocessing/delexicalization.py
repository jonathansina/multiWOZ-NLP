
import re
import copy
import string
from typing import Dict, List, Tuple


def slot_to_placeholder(slot_name: str) -> str:
    return slot_name.replace('-', '_').upper()


def extract_slot_pairs_from_act(act: Dict) -> List[Tuple[str, str]]:
    pairs = []
    for aslots in act.get("act_slots", []):
        names = aslots.get("slot_name", [])
        values = aslots.get("slot_value", [])

        for s, v in zip(names, values):
            if v in string.punctuation:
                v = "unknown"
    
            if s and v:
                pairs.append((s, v))
    return pairs


def delexicalize_text(text: str, slot_pairs: List[Tuple[str, str]]) -> str:
    delexicalized_text = copy.deepcopy(text)
    for s, v in sorted(slot_pairs, key=lambda kv: len(kv[1]), reverse=True):
        if not v:
            continue

        placeholder = f"[{slot_to_placeholder(s)}]"
        pattern = re.escape(v)
        delexicalized_text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)

    return delexicalized_text


def delexicalize_action_label(act: Dict) -> str:
    parts = []
    
    for act_type, aslots in zip(act.get("act_type", []), act.get("act_slots", [])):
        slot_strs = []
        names = aslots.get("slot_name", [])
        values = aslots.get("slot_value", [])
        
        for s, v in zip(names, values):
            if v in string.punctuation:
                placeholder = "UNKNOWN"
            else:
                placeholder = slot_to_placeholder(s)
                
            slot_strs.append(f"{s.split('-')[-1]}={placeholder}")
        parts.append(f"{act_type}({', '.join(slot_strs)})")

    return " | ".join(parts)


def lexicalize_from_placeholders(text: str, slot_pairs: List[Tuple[str, str]]) -> str:
    for s, v in sorted(slot_pairs, key=lambda kv: len(kv[0]), reverse=True):
        placeholder = f"[{slot_to_placeholder(s)}]"
        lexicalized_text = text.replace(placeholder, v)

    return lexicalized_text