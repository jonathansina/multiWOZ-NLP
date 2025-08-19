import sys
sys.path.append("..")

import tqdm
import pickle
from typing import List, Dict, Tuple

from scripts.preprocessing.delexicalization import (
    delexicalize_text, 
    delexicalize_action_label,
    extract_slot_pairs_from_act,
)

from transformers import T5Tokenizer
from torch.utils.data import Dataset
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def get_evaluation_score(predictions: List[str], references: List[str], save: bool = False) -> float:
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

    P, R, F1 = bert_score(
        predictions, 
        references, 
        lang='en', 
        rescale_with_baseline=True, 
        verbose=True, 
        batch_size=256
    )
    
    avg_bert_f1 = F1.mean().item()
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    if save:
        pickle.dump({
            "bleu_score": avg_bleu,
            "bert_f1_score": avg_bert_f1
        }, open(save, "wb"))

    return {
        "bleu_score": avg_bleu,
        "bert_f1_score": avg_bert_f1
    }


def preprocess_text_realization(dialogue: Dict, delex: bool = False) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []

    turns = dialogue["turns"]
    utterances = turns["utterance"]
    speakers = turns["speaker"]
    acts_all = turns["dialogue_acts"]

    last_user_utterrance = ""
    for i, (utt, speaker) in enumerate(zip(utterances, speakers)):
        if speaker == 0:
            last_user_utterrance = utt
        elif speaker == 1 and i < len(acts_all):
            act = acts_all[i]["dialog_act"]
            slot_pairs = extract_slot_pairs_from_act(act)

            if delex:
                input_user = delexicalize_text(last_user_utterrance, slot_pairs)
                input_action = delexicalize_action_label(act)
                output_response = delexicalize_text(utt, slot_pairs)
            else:
                input_user = last_user_utterrance

                parts = []
                for act_type, aslots in zip(act.get("act_type", []), act.get("act_slots", [])):
                    names = aslots.get("slot_name", [])
                    values = aslots.get("slot_value", [])
                    slots = [f"{n.split('-')[-1]}={v}" for n, v in zip(names, values)]
                    parts.append(f"{act_type}({', '.join(slots)})")
                input_action = " | ".join(parts)
                output_response = utt

            input_with_context = f"[USER]: {input_user} [ACTION]: {input_action}"
            samples.append((input_with_context, output_response))

    return samples


class ResponseDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: T5Tokenizer, max_output_len: int = 64, max_input_len: int = 64, delex: bool = False):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        self.actions = []
        self.responses = []

        for dialogue in tqdm.tqdm(data, desc="Processing dialogues"):
            preprocessed_text = preprocess_text_realization(dialogue, delex=delex)

            for (action, response) in preprocessed_text:
                self.actions.append(action)
                self.responses.append(response)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        action_enc = self.tokenizer(
            self.actions[idx], 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_input_len, 
            return_tensors="pt"
        )

        response_enc = self.tokenizer(
            self.responses[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_output_len,
            return_tensors="pt"
        )

        return {
            "encoder_input_ids": action_enc["input_ids"].squeeze(0),
            "encoder_attention_mask": action_enc["attention_mask"].squeeze(0),
            "decoder_input_ids": response_enc["input_ids"].squeeze(0),
            "decoder_attention_mask": response_enc["attention_mask"].squeeze(0)
        }