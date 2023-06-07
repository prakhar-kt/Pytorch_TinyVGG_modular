import os
import torch
import torch.nn as nn
from torchvision import transforms
import argparse

import data_setup, model_builder, utils, train_test_steps

# Setup Hyperparameters
parser = argparse.ArgumentParser(description="Get some Hyperparamters.")

parser.add_argument("--num_epochs",
                    type=int,
                    default=10,
                    help="Number of epochs to train the model for.")

parser.add_argument("--batch_size",
                    type=int,
                    default=32,
                    help="Number of samples per batch")

parser.add_argument("--hidden_units",
                    type=int,
                    default=10,
                    help="Number of hidden units in the model.")

parser.add_argument("--learning_rate",
                    type=float,
                    default=0.001,
                    help="Learning rate for the optimizer.")

args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate

print(f"Training a model for {NUM_EPOCHS} epochs with a batch size of {BATCH_SIZE} and a learning rate of {LEARNING_RATE}...")





# Define paths
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# define the transforms
transform = transforms.Compose(
    [transforms.Resize((64, 64)),
     transforms.ToTensor()]
)

# create dataloaders using the data_setup

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
                                                        train_dir=train_dir,
                                                        test_dir=test_dir,
                                                        transform=transform,
                                                        batch_size=BATCH_SIZE


)

# create model using the model_builder

model = model_builder.TinyVGG(input_shape=3, 
                              hidden_units=HIDDEN_UNITS, 
                              output_shape=len(class_names)).to(device)



# define the loss function and optimizer

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# train the model using the train_test_steps

results = train_test_steps.train_model(model=model,
                                       train_dataloader=train_dataloader,
                                       test_dataloader=test_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device,
                                       epochs=NUM_EPOCHS)


utils.save_checkpoint(model=model,
                      target_dir="model_checkpoints",
                      model_name="tiny_vgg.pth")


