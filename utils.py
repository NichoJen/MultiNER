import torch


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=False, skip_index=-100):
    """
    Adapted from assignment 4: use to tokenize and align tokens with labels
    tokenizer: tokenizer from huggingface
    examples: data sample
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
    """
    real_word_indices = get_real_word_indices(tokenizer_word_ids)
    # convert to tensor
    real_word_indices = torch.tensor(real_word_indices, dtype=torch.int64)
    clean_output = torch.index_select(ner_output, 0, real_word_indices)
    return clean_output


def clean_ner_output_eval(ner_predictions, labels):
    """
    remove subwords and padding from outputs and labels and convert labels to seqeval format
    """
    indices_to_keep = torch.where(labels != -100, 1, 0)  # get indices that are not
    indices_to_keep = list(indices_to_keep)

    ner_predictions = list(ner_predictions)
    labels = list(labels)

    ner_predictions_clean = []
    labels_clean = []

    for pred, lab, indices in zip(ner_predictions, labels, indices_to_keep):
        indices = indices.nonzero().T[0]  # get nonzero indices and convert to vector
        clean_pred = torch.index_select(pred, 0, indices)
        clean_lab = torch.index_select(lab, 0, indices)

        ner_predictions_clean.append(clean_pred)
        labels_clean.append(clean_lab)

    return ner_predictions_clean, labels_clean
