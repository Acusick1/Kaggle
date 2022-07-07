import os
import torch
import numpy as np
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from src.titanic import prepare_dataset_pandas, load_clean_data
from src.settings import DATA_PATH, DEVICE


class Net(nn.Module):
    def __init__(self, l1=128, l2=64):
        super(Net, self).__init__()
        self.hid1 = nn.Linear(10, l1)
        self.hid2 = nn.Linear(l1, l2)
        self.oupt = nn.Linear(l2, 2)

        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        x = torch.relu(self.hid1(x))
        x = torch.relu(self.hid2(x))
        x = self.oupt(x)  # no softmax: CrossEntropyLoss()
        return x


def train(net, optimizer, criterion, train_loader, val_loader, max_num_epochs=10, device=DEVICE, tuning=False):

    for epoch in range(max_num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        #  for i, data in enumerate(train_subset):
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        if tuning:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), path)

            tune.report(loss=(val_loss / val_steps), accuracy=correct / total)


def get_train_val_loaders(train_data, split=0.8, batch_size=32, num_workers=0):

    test_abs = int(len(train_data) * split)
    train_subset, val_subset = random_split(
        train_data, [test_abs, len(train_data) - test_abs])

    train_loader = DataLoader(
        train_subset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=num_workers
    )

    return train_loader, val_loader


def train_cifar(config, train_data, num_workers=0, max_num_epochs=10, checkpoint_dir=None):

    net = Net(config["l1"], config["l2"])

    # Have to define device from within since running via scheduler
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    else:
        device = "cpu"

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_loader, val_loader = get_train_val_loaders(
        train_data,
        batch_size=config["batch_size"],
        num_workers=num_workers
    )

    train(net, optimizer, criterion, train_loader, val_loader,
          max_num_epochs=max_num_epochs,
          device=device,
          tuning=True)

    print("Finished Training")


def dummy_config_test():

    target = "Survived"
    drop_columns = ["PassengerId", "Cabin", "Ticket", "Name", "Sex"]

    train_data, _ = load_clean_data()
    train_data = prepare_dataset_pandas([train_data], drop=drop_columns)[0]

    train_target = torch.tensor(train_data[target].values).type(torch.LongTensor)
    train_features = torch.tensor(train_data.drop(target, axis=1).values.astype(np.float32))

    dataset = TensorDataset(train_features, train_target)

    config = {
        "l1": 128,
        "l2": 64,
        "lr": 0.1,
        "batch_size": 16
    }

    train_cifar(config, dataset)


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0):

    target = "Survived"
    drop_columns = ["PassengerId", "Cabin", "Ticket", "Name", "Sex"]

    raw_train_data, raw_test_data = load_clean_data()
    train_data, test_data = prepare_dataset_pandas([raw_train_data, raw_test_data], drop=drop_columns)

    train_target = torch.tensor(train_data[target].values).type(torch.LongTensor)
    train_features = torch.tensor(train_data.drop(target, axis=1).values.astype(np.float32))
    test_features = torch.tensor(test_data.drop(target, axis=1).values.astype(np.float32))

    dataset = TensorDataset(train_features, train_target)

    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16, 32])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar, train_data=dataset, max_num_epochs=max_num_epochs),  # , data_dir=data_dir),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])

    if "cuda" in DEVICE and torch.cuda.device_count() > 1:
        best_trained_model = nn.DataParallel(best_trained_model)

    best_trained_model.to(DEVICE)
    test = test_features.to(DEVICE)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    prediction = best_trained_model(test)
    raw_test_data[target] = prediction.cpu().detach().numpy().argmax(axis=1)
    raw_test_data.to_csv(DATA_PATH / "titanic" / "nn_prediction.csv", columns=["PassengerId", target], index=False)

    # test_acc = test_accuracy(best_trained_model, DEVICE)
    # print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":

    # dummy_config_test()
    main(num_samples=100, max_num_epochs=100)
