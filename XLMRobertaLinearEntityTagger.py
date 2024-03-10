import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from utils import clean_ner_output_eval
import evaluate


class XLMRobertaLinearEntityTagger(nn.Module):
    def __init__(self, num_classes, xlm_roberta_model):
        super(XLMRobertaLinearEntityTagger, self).__init__()
        self.xlm_roberta = xlm_roberta_model
        self.l1 = nn.Linear(768, num_classes)

        # remove roberta params from backprop
        for param in self.xlm_roberta.parameters():
            param.requires_grad = False

        self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.l1(roberta_output)
        return logits

    def compute_metrics(self, dataloader, device, split=""):
        self.eval()
        correct = 0
        n_real_tokens = 0
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_masks = batch["attention_mask"].to(device)

                predicted_logits = self(inputs, attention_masks).transpose(1, 2)

                loss = self.loss(predicted_logits, labels)

                predicted_labels = torch.argmax(predicted_logits, dim=1)

                batch_correct = torch.eq(predicted_labels, labels)

                n_real_tokens_batch = torch.ne(labels, -100).sum()  # number of tokens with label other than -100

                total_loss += loss.item()
                correct += batch_correct.sum()
                n_real_tokens += n_real_tokens_batch

        total_loss /= len(dataloader)
        accuracy = correct / n_real_tokens

        print(f"{split} loss: {total_loss} Accuracy: {correct}/{n_real_tokens} {(accuracy * 100):.2f}%")

        return total_loss, accuracy


seqeval = evaluate.load('seqeval')
index_to_label_dict = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC',
                       8: 'I-MISC'}
index_to_labels = lambda labels: [index_to_label_dict[label.item()] for label in labels]


def evaluate_model(model, dataloader, device, index_to_label_fn, split=""):
    model.eval()

    correct = 0
    n_real_tokens = 0
    total_loss = 0

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_masks = batch["attention_mask"].to(device)

            predicted_logits = model(inputs, attention_masks).transpose(1, 2)

            loss = model.loss(predicted_logits, labels)

            predicted_labels = torch.argmax(predicted_logits, dim=1)

            batch_correct = torch.eq(predicted_labels, labels)

            n_real_tokens_batch = torch.ne(labels, -100).sum()  # number of tokens with label other than -100

            total_loss += loss.item()
            correct += batch_correct.sum()
            n_real_tokens += n_real_tokens_batch

            # get labels for seqeval
            clean_predictions, clean_labels = clean_ner_output_eval(predicted_labels, labels)
            predictions = list(map(index_to_labels, clean_predictions))
            references = list(map(index_to_labels, clean_labels))
            all_predictions += predictions
            all_references += references

    total_loss /= len(dataloader)
    accuracy = correct / n_real_tokens

    seqeval_results = seqeval.compute(predictions=all_predictions, references=all_references, scheme="IOB2",
                                      mode="strict")

    metrics = {"loss": total_loss, "accuracy": accuracy, "seqeval": seqeval_results}

    return metrics


def train_model(model, lr, epochs, batch_size, train_loader, project_name, device, val_loader=None,
                test_loader=None):
    num_batches = len(train_loader)

    run = wandb.init(
        project=project_name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
        })

    config = wandb.config

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        print("epoch: ", epoch)
        train_loss_per_epoch = 0
        steps = 0
        for step, batch in (pbar := tqdm(enumerate(train_loader), total=num_batches)):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_masks = batch["attention_mask"].to(device)

            predicted_logits = model(inputs, attention_masks).transpose(1, 2)
            loss = loss_fn(predicted_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_metrics = {"train/train_loss_per_step": loss.item()}
            wandb.log(step_metrics)

            train_loss_per_epoch += loss.item()

        train_loss_per_epoch /= num_batches
        metrics = evaluate_model(model, train_loader, device, index_to_label_fn=index_to_labels)

        wandb_metrics = {"train/train_loss_per_epoch": train_loss_per_epoch,
                         "train/accuracy_all_tokens": metrics["accuracy"],
                         "train/overall_precision": metrics["seqeval"]["overall_precision"],
                         "train/overall_recall": metrics["seqeval"]["overall_recall"],
                         "train/overall_f1": metrics["seqeval"]["overall_f1"],
                         "train/overall_accuracy": metrics["seqeval"]["overall_accuracy"],
                         "train/LOC/precision": metrics["seqeval"]["LOC"]["precision"],
                         "train/LOC/recall": metrics["seqeval"]["LOC"]["recall"],
                         "train/LOC/f1": metrics["seqeval"]["LOC"]["f1"],
                         "train/LOC/number": metrics["seqeval"]["LOC"]["number"],
                         "train/MISC/precision": metrics["seqeval"]["MISC"]["precision"],
                         "train/MISC/recall": metrics["seqeval"]["MISC"]["recall"],
                         "train/MISC/f1": metrics["seqeval"]["MISC"]["f1"],
                         "train/MISC/number": metrics["seqeval"]["MISC"]["number"],
                         "train/ORG/precision": metrics["seqeval"]["ORG"]["precision"],
                         "train/ORG/recall": metrics["seqeval"]["ORG"]["recall"],
                         "train/ORG/f1": metrics["seqeval"]["ORG"]["f1"],
                         "train/ORG/number": metrics["seqeval"]["ORG"]["number"],
                         "train/PER/precision": metrics["seqeval"]["PER"]["precision"],
                         "train/PER/recall": metrics["seqeval"]["PER"]["recall"],
                         "train/PER/f1": metrics["seqeval"]["PER"]["f1"],
                         "train/PER/number": metrics["seqeval"]["PER"]["number"],
                         "train/epoch": epoch + 1}

        wandb.log(wandb_metrics)

        print(f"epoch loss: {train_loss_per_epoch}")

        val_metrics = evaluate_model(model, val_loader, device, index_to_label_fn=index_to_labels)

        # Log train and validation metrics to wandb
        val_metrics = {"val/val_loss": val_metrics["loss"],
                       "val/val_accuracy_all_tokens": val_metrics["accuracy"],
                       "overall_precision": val_metrics["seqeval"]["overall_precision"],
                       "val/overall_recall": val_metrics["seqeval"]["overall_recall"],
                       "val/overall_f1": val_metrics["seqeval"]["overall_f1"],
                       "val/overall_accuracy": val_metrics["seqeval"]["overall_accuracy"],
                       "val/LOC/precision": val_metrics["seqeval"]["LOC"]["precision"],
                       "val/LOC/recall": val_metrics["seqeval"]["LOC"]["recall"],
                       "val/LOC/f1": val_metrics["seqeval"]["LOC"]["f1"],
                       "val/LOC/number": val_metrics["seqeval"]["LOC"]["number"],
                       "val/MISC/precision": val_metrics["seqeval"]["MISC"]["precision"],
                       "val/MISC/recall": val_metrics["seqeval"]["MISC"]["recall"],
                       "val/MISC/f1": val_metrics["seqeval"]["MISC"]["f1"],
                       "val/MISC/number": val_metrics["seqeval"]["MISC"]["number"],
                       "val/ORG/precision": val_metrics["seqeval"]["ORG"]["precision"],
                       "val/ORG/recall": val_metrics["seqeval"]["ORG"]["recall"],
                       "val/ORG/f1": val_metrics["seqeval"]["ORG"]["f1"],
                       "val/ORG/number": val_metrics["seqeval"]["ORG"]["number"],
                       "val/PER/precision": val_metrics["seqeval"]["PER"]["precision"],
                       "val/PER/recall": val_metrics["seqeval"]["PER"]["recall"],
                       "val/PER/f1": val_metrics["seqeval"]["PER"]["f1"],
                       "val/PER/number": val_metrics["seqeval"]["PER"]["number"]}
        wandb.log({**metrics, **val_metrics})

        # save model checkpoint
        checkpoint_path = 'models/xlm_roberta_wiki_neural_eng_ep_' + str(epoch) + '.pth'
        torch.save({'epoch': epoch,
                    'model_state_dict': model.l1.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_per_epoch},
                   checkpoint_path)

        # create wandb artifact
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(checkpoint_path)
        run.log_artifact(artifact)

    # log test metrics
    test_metrics = evaluate_model(model, test_loader, device, index_to_label_fn=index_to_labels)

    test_metrics = {"overall_precision": test_metrics["seqeval"]["overall_precision"],
                    "overall_recall": test_metrics["seqeval"]["overall_recall"],
                    "overall_f1": test_metrics["seqeval"]["overall_f1"],
                    "overall_accuracy": test_metrics["seqeval"]["overall_accuracy"],
                    "LOC/precision": test_metrics["seqeval"]["LOC"]["precision"],
                    "LOC/recall": test_metrics["seqeval"]["LOC"]["recall"],
                    "LOC/f1": test_metrics["seqeval"]["LOC"]["f1"],
                    "LOC/number": test_metrics["seqeval"]["LOC"]["number"],
                    "MISC/precision": test_metrics["seqeval"]["MISC"]["precision"],
                    "MISC/recall": test_metrics["seqeval"]["MISC"]["recall"],
                    "MISC/f1": test_metrics["seqeval"]["MISC"]["f1"],
                    "MISC/number": test_metrics["seqeval"]["MISC"]["number"],
                    "ORG/precision": test_metrics["seqeval"]["ORG"]["precision"],
                    "ORG/recall": test_metrics["seqeval"]["ORG"]["recall"],
                    "ORG/f1": test_metrics["seqeval"]["ORG"]["f1"],
                    "ORG/number": test_metrics["seqeval"]["ORG"]["number"],
                    "PER/precision": test_metrics["seqeval"]["PER"]["precision"],
                    "PER/recall": test_metrics["seqeval"]["PER"]["recall"],
                    "PER/f1": test_metrics["seqeval"]["PER"]["f1"],
                    "PER/number": test_metrics["seqeval"]["PER"]["number"]}

    wandb.summary.update(test_metrics)
