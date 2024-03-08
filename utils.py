import torch


def tokenize_and_align_labels(tokenizer, examples, label_all_tokens=False, skip_index=-100):
    """
    From assignment 4: use to tokenize and align tokens with labels
    :param tokenizer: tokenizer from huggingface
    :type tokenizer: tokenizer
    :param examples: data sample
    :type examples:
    :param label_all_tokens:
    :type label_all_tokens: bool
    :param skip_index: value for indices that should skipped
    :type skip_index: int
    :return:
    :rtype:
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True,
                                 padding=True)
    labels = []

    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids: list[int] = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(skip_index)

            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else skip_index)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_real_word_indices(word_ids):
    """
    get the starting indices of each word in list of subwords
    :param word_ids:
    :type word_ids:
    :return:
    :rtype:
    """
    indices = []
    prev_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != prev_word_idx:
            indices.append(i)
            prev_word_idx = word_idx
    return indices


def clean_ner_output(ner_output, tokenizer_word_ids):
    """
    return ner labels without subwords and padding tokens
    :param ner_output:
    :type ner_output:
    :param tokenizer_word_ids:
    :type tokenizer_word_ids:
    :return:
    :rtype:
    """
    real_word_indices = get_real_word_indices(tokenizer_word_ids)
    # convert to tensor
    real_word_indices = torch.tensor(real_word_indices, dtype=torch.int64)
    clean_output = torch.index_select(ner_output, 0, real_word_indices)
    return clean_output
