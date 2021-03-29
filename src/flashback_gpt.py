import sys
import torch
import re
from typing import List
from format import format_thread, format_header, format_thread_post, format_post_item, parse_thread
from transformers import NoBadWordsLogitsProcessor, BeamSearchScorer


def generate_post(model, tokenizer, thread, quote_post_idx=None, by_user=None, context_size=100, 
                  length_penalty=1.0, num_beams=5, **kwargs):
    """
    Generate a post to a forum thread. 
    Optionally quote a previous post.
    Optionally generate from a provided username.
    Optionally specify a length penalty.
    Optionally specify number of previous posts to include as context
    """
    context = thread

    # Take the last context_size posts
    context["posts"] = context["posts"][-context_size:] if context_size > 0 else []

    # Format context
    thread_str = format_thread(context, allow_empty=True)[0]

    # Generate or specify username for this post
    if by_user is not None:
        thread_str += by_user + ":\n"
    else:
        # Generate a user
        thread_str = generate_from_model(model, tokenizer, thread_str, eos_token_ids=tokenizer.encode("\n"), 
                                         bad_words_ids=[[2422]], num_beams=num_beams, num_return_sequences=1)[0]
        # Add a colon if the model forgets it...
        thread_str = re.sub(r":*\n$", ":\n", thread_str)

    # Optionally force quote
    if quote_post_idx is not None and 0 <= quote_post_idx < len(thread["posts"]):
        # Filter text items to quote
        quote_post = thread["posts"][quote_post_idx]
        quote_post["post"] = [post_item for post_item in quote_post["post"] if post_item["type"] == "text"]
        thread_str += format_post_item({
            "type": "quote",
            "username": quote_post["username"],
            "post": quote_post["post"]
        })

    bad_words_ids = [[200],[364]]  # Ban generating "Citat" and tab characters
    generated_texts = generate_from_model(model, tokenizer, thread_str, eos_token_ids=tokenizer.encode("\n\n"), 
                                         min_length=2, bad_words_ids=bad_words_ids, length_penalty=length_penalty*len(context["posts"]), 
                                         num_beams=num_beams, 
                                         num_return_sequences=1, **kwargs)

     # We return only the most likely beam
    generated_text = generated_texts[0]

    # In case generation hit max_length, make sure the post ends with two newlines to make the post valid
    while not generated_text.endswith("\n\n"):
        generated_text += "\n"

    # Parse the full thread and extract the generated post (last one)
    parsed_thread = parse_thread(generated_text)
    post = parsed_thread["posts"][-1]
    
    return post, generated_text


@torch.no_grad()
def generate_from_model(model, tokenizer, input_text, eos_token_ids: List[int]=None, min_length=0, 
        max_length=400, num_beams=6, no_repeat_ngrams=3, bad_words_ids=None, 
        num_return_sequences=1, length_penalty=1.0, **model_kwargs) -> List[str]:

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    input_ids, model_kwargs = model._expand_inputs_for_generation(input_ids, expand_size=num_beams)

    if eos_token_ids is not None and len(eos_token_ids) > 1:
        eos_token_id = tokenizer.eos_token_id
    else:
        eos_token_id = None if eos_token_ids is None else eos_token_ids[0]

    logits_processor = model._get_logits_processor(
        repetition_penalty=None,
        no_repeat_ngram_size=no_repeat_ngrams,
        encoder_no_repeat_ngram_size=None,
        encoder_input_ids=None,
        bad_words_ids=bad_words_ids,
        min_length=input_ids.shape[-1] + min_length,
        eos_token_id=eos_token_id,
        prefix_allowed_tokens_fn=None,
        num_beams=num_beams,
        num_beam_groups=None,
        diversity_penalty=None,
    )

    logits_warper = model._get_logits_warper(
        top_k=None, top_p=None, temperature=None, num_beams=num_beams
    )
    
    if eos_token_ids is not None and len(eos_token_ids) > 1:
        logits_processor.insert(0, MultiTokenEOSLogitsProcessor(eos_token_ids, tokenizer.eos_token_id))
    
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        max_length=max_length,
        num_beams=num_beams,
        device=model.device,
        length_penalty=length_penalty,
        num_beam_hyps_to_keep=num_return_sequences
    )
    
    output_ids = model.beam_sample(
        input_ids,
        beam_scorer,
        logits_processor=logits_processor,
        logits_warper=logits_warper,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_id,
        **model_kwargs,
    )

    decoded_outputs = []
    for i in range(len(output_ids)):
        ids = [token_id for token_id in output_ids[i].tolist() if token_id != tokenizer.pad_token_id]
        # If generation stopped by artificial eos_token, replace it
        if ids[-1] == tokenizer.eos_token_id and eos_token_ids is not None:
            ids[-1] = eos_token_ids[-1]
        decoded_outputs.append(tokenizer.decode(ids))

    return decoded_outputs


class MultiTokenEOSLogitsProcessor(NoBadWordsLogitsProcessor):
    """
    To enable generation stopping not just for single eos_token, but for a specific list of tokens (eos_token_ids).
    This is done by modifying the logits such that the real eos_token is generated if the eos_token_ids are about to be generated
    """
    def __init__(self, eos_token_ids: List[int], real_eos_token: int):
        self.bad_words_ids = [eos_token_ids]
        self.real_eos_token = real_eos_token

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids:  int tensor of previous token ids (batch_size * num_beams, seq_len)
            scores:     float tensor of size (batch_size * num_beams, vocab_size)
        Returns:
            modified_scores: float tensor
        """
        tokens = self._calc_banned_bad_words_ids(input_ids)
        scores = self._move_scores_to_real_eos_token(scores, tokens)
        return scores

    def _move_scores_to_real_eos_token(self, scores: torch.Tensor, banned_tokens: List[List[int]]) -> None:
        """
        Moves the scores of banned_tokens to real_eos_token, and sets the scores of banned_tokens to -inf
        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
        """
        assert all(len(sublist) <= 1 for sublist in banned_tokens), "banned_tokens should at most have one token to be swapped"
        for idx, batch_banned_tokens in enumerate(banned_tokens):
            for token in batch_banned_tokens:
                scores[idx, self.real_eos_token] = scores[idx, token]
                scores[idx, token] = -float("inf")

        return scores
