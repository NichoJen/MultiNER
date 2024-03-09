import torch
import torch.nn as nn
import wandb
from tqdm import tqdm


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
        _, accuracy = model.compute_metrics(train_loader)

        metrics = {"train/train_loss_per_epoch": train_loss_per_epoch,
                   "train/accuracy": accuracy,
                   "train/epoch": epoch + 1}

        wandb.log(metrics)

        print(f"epoch loss: {train_loss_per_epoch}")

        val_loss, val_accuracy = model.compute_metrics(val_loader)

        # Log train and validation metrics to wandb
        val_metrics = {"val/val_loss": val_loss,
                       "val/val_accuracy": val_accuracy}
        wandb.log({**metrics, **val_metrics})

        # save model checkpoint
        checkpoint_path = 'models/xlm_roberta_wiki_neural_ep_' + str(epoch) + '.pth'
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
    test_loss, test_accuracy = model.compute_metrics(test_loader)
    wandb.summary['test_accuracy'] = test_loss
    wandb.summary['test_accuracy'] = test_accuracy

    wandb.finish()
