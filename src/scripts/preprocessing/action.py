import sys
sys.path.append("..")

import tqdm
from typing import List, Dict, Tuple

from scripts.preprocessing.delexicalization import (
    delexicalize_text, 
    delexicalize_action_label,
    extract_slot_pairs_from_act
)

from transformers import T5Tokenizer
from torch.utils.data import Dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def get_evaluation_score(predictions: List[str], references: List[str]) -> float:
    total_slots = 0
    missed_slots = 0
    smoothing = SmoothingFunction().method1
    
    predictions = [pred.strip().lower() for pred in predictions]
    references = [ref.strip().lower() for ref in references]
    
    bleu_scores = [
        sentence_bleu(
            [ref.split()], 
            pred.split(), 
            smoothing_function=smoothing, 
            weights=(1, 0, 0, 0)
        ) 
        for pred, ref in zip(predictions, references)
    ]

    for pred, act in zip(predictions, references):
        slots = []
        for part in act.split('|'):
            if '(' in part and ')' in part:
                slot_str = part.split('(')[1].rstrip(')')
                for s in slot_str.split(', '):
                    if '=' in s:
                        slots.append(s.split('=')[1].strip())

        total_slots += len(slots)
        for slot_val in slots:
            if slot_val.lower() not in pred.lower():
                missed_slots += 1

    ser = 1 - (missed_slots / total_slots if total_slots > 0 else 0.0)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    return {
        "bleu_score": avg_bleu,
        "ser_score": ser
    }


def preprocess_action_prediction(dialogue: Dict, max_turns: int, delex: bool = False) -> List[Tuple[str, str]]:
    samples = []

    turns = dialogue["turns"]
    speakers = turns["speaker"]
    utterances = turns["utterance"]
    acts_all = turns["dialogue_acts"]

    context = []
    for i, (utterance, speaker) in enumerate(zip(utterances, speakers)):
        context.append(("USER" if speaker == 0 else "SYS") + ": " + utterance)
        
        if len(context) > max_turns:
            context = context[-max_turns:]

        if i < len(acts_all):
            act = acts_all[i]["dialog_act"]
            slot_pairs = extract_slot_pairs_from_act(act)

            if delex:
                context_str = " ".join([delexicalize_text(u, slot_pairs) for u in context])
                action_label = delexicalize_action_label(act)
            else:
                context_str = " ".join([u for u in context])
                parts = []
                
                for act_type, aslots in zip(act.get("act_type", []), act.get("act_slots", [])):
                    names = aslots.get("slot_name", [])
                    values = aslots.get("slot_value", [])
                    slots = [f"{n.split('-')[-1]}={v}" for n, v in zip(names, values)]
                    parts.append(f"{act_type}({', '.join(slots)})")
        
                action_label = " | ".join(parts)

            samples.append((context_str, action_label))

    return samples


class ActionDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: T5Tokenizer, max_turns: int, max_output_len: int = 64, max_input_len: int = 64, delex: bool = False):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        
        self.inputs = []
        self.actions = []

        for dialogue in tqdm.tqdm(data, desc="Processing dialogues"):
            preprocessed_text = preprocess_action_prediction(
                dialogue, 
                max_turns=max_turns, 
                delex=delex
            )

            for (input, action) in preprocessed_text:
                self.inputs.append(input)
                self.actions.append(action)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        input_enc = self.tokenizer(
            self.inputs[idx], 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_input_len, 
            return_tensors="pt"
        )

        action_enc = self.tokenizer(
            self.actions[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_output_len,
            return_tensors="pt"
        )

        return {
            "encoder_input_ids": input_enc["input_ids"].squeeze(0),
            "encoder_attention_mask": input_enc["attention_mask"].squeeze(0),
            "decoder_input_ids": action_enc["input_ids"].squeeze(0),
            "decoder_attention_mask": action_enc["attention_mask"].squeeze(0)
        }