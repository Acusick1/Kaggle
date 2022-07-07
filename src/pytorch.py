import os
import pandas as pd
import torch
import numpy as np
from functools import partial
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import KFold
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


def cross_validate(train_func, kfold, config, train_data):

    fold_results = []
    for fold, (train_index, test_index) in enumerate(kfold.split(train_data)):
        # Dividing data into folds
        # TODO: this is not the ideal way to do this; redefining an input tensordataset, may have to put input dataframe
        #  instead
        train_split, val_split = train_data[train_index], train_data[test_index]

        train_set = TensorDataset(train_split[0], train_split[1])
        val_set = TensorDataset(val_split[0], val_split[1])

        train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=False)
        val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

        # print("[%d, %2d] cross-validation" % (fold, kfold.n_splits))
        fold_results.append(train_func(train_loader=train_loader, val_loader=val_loader))

    result_df = pd.DataFrame(fold_results, columns=["loss", "accuracy"])
    results = result_df.mean().to_dict()

    idx = int(result_df["loss"].idxmin())

    # TODO: Need to choose model? Just here to satisfy ray tune checkpoints?
    results["epoch"] = fold_results[idx]["epoch"]
    results["net"] = fold_results[idx]["net"]
    results["optimizer"] = fold_results[idx]["optimizer"]

    return results


def train(net, optimizer, criterion, train_loader, val_loader, max_num_epochs=10, device=DEVICE):

    print_every = min(len(train_loader), 1000) - 1
    results = dict()
    best = np.inf

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
            if i > 0 and i % print_every == 0:
                # print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        sum_val_loss = 0.0
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
                sum_val_loss += loss.cpu().numpy()
                val_steps += 1

        val_loss = sum_val_loss / val_steps
        accuracy = correct / total
        # TODO: Better performance definition
        current = val_loss + (1 - accuracy)

        if epoch == 0 or current < best:
            best = current
            results["epoch"] = epoch + 1
            results["loss"] = val_loss
            results["accuracy"] = accuracy
            results["net"] = net.state_dict()
            results["optimizer"] = optimizer.state_dict()

    return results


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


def tune_wrapper(train_func):

    results = train_func()
    # TODO: Is specifying epoch necessary?
    with tune.checkpoint_dir(results["epoch"]) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((results["net"], results["optimizer"]), path)

    tune.report(loss=results["loss"], accuracy=results["accuracy"])


def train_cifar(config, train_data, num_workers=0, max_num_epochs=10, checkpoint_dir=None, cv=False):

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

    if not cv:

        train_loader, val_loader = get_train_val_loaders(
            train_data,
            batch_size=config["batch_size"],
            num_workers=num_workers
        )

        train_func = partial(train,
                             net=net,
                             optimizer=optimizer,
                             criterion=criterion,
                             train_loader=train_loader,
                             val_loader=val_loader,
                             max_num_epochs=max_num_epochs,
                             device=device)
    else:
        train_fold_func = partial(train,
                                  net=net,
                                  optimizer=optimizer,
                                  criterion=criterion,
                                  max_num_epochs=max_num_epochs,
                                  device=device)

        train_func = partial(cross_validate,
                             train_func=train_fold_func,
                             kfold=KFold(n_splits=10, shuffle=True, random_state=0),
                             config=config,
                             train_data=train_data)

    tune_wrapper(train_func)

    print("Finished Training")


def get_example_config():

    config = {
        "l1": 128,
        "l2": 64,
        "lr": 0.1,
        "batch_size": 16,
        "momentum": 0.9
    }

    return config


def run_single_config_cv():

    target = "Survived"
    drop_columns = ["PassengerId", "Cabin", "Ticket", "Name", "Sex"]

    train_data, _ = load_clean_data()
    train_data = prepare_dataset_pandas([train_data], drop=drop_columns)[0]

    train_target = torch.tensor(train_data[target].values).type(torch.LongTensor)
    train_features = torch.tensor(train_data.drop(target, axis=1).values.astype(np.float32))

    dataset = TensorDataset(train_features, train_target)

    config = get_example_config()

    kfold = KFold(n_splits=10, random_state=0, shuffle=True)

    net = Net(config["l1"], config["l2"])
    net.to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"])
    criterion = nn.CrossEntropyLoss()

    train_func = partial(train, net=net, optimizer=optimizer, criterion=criterion)
    results = cross_validate(train_func, kfold, config, dataset)

    return results


def run_single_config():

    target = "Survived"
    drop_columns = ["PassengerId", "Cabin", "Ticket", "Name", "Sex"]

    train_data, _ = load_clean_data()
    train_data = prepare_dataset_pandas([train_data], drop=drop_columns)[0]

    train_target = torch.tensor(train_data[target].values).type(torch.LongTensor)
    train_features = torch.tensor(train_data.drop(target, axis=1).values.astype(np.float32))

    dataset = TensorDataset(train_features, train_target)

    config = get_example_config()

    net = Net(config["l1"], config["l2"])
    net.to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"])
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = get_train_val_loaders(dataset, batch_size=config["batch_size"])
    train(net, optimizer, criterion, train_loader, val_loader)
    # train_cifar(config, dataset)


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0, cv=False):

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
        partial(train_cifar, train_data=dataset, max_num_epochs=max_num_epochs, cv=cv),  # , data_dir=data_dir),
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

    # run_single_config()
    # run_single_config_cv()
    main(num_samples=1000, max_num_epochs=100, cv=True)
