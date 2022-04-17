import numpy as np
import torch
from sklearn import metrics
from torch.autograd.grad_mode import F
from torchvision import models
import torch.nn as nn
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import time, os
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import transforms


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


pretrained = True  # We want to use the pretrained model
feature_extract = True  # We want to free all the model parameters at the begining

net = models.resnet18(pretrained=pretrained)
set_parameter_requires_grad(net, feature_extract)

# (1) Modify the last fc layer for binary classification
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, out_features=2)

# (2) Note that there is only one Linear layer as the classifier in resnet18, so replacing fc is replacing the
#     whole classifier. Consider fine-tuning layer4.
for param in net.layer4.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()


# define train function for one epoch
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / (batch_idx + 1)


# define evaluation function
def evaluate(model, device, eval_loader, criterion):
    model.eval()
    correct_eval, total_eval = 0, 0
    all_targets = torch.randn(0).to(device)
    all_predicted_labels = torch.randn(0).to(device)
    all_predicted_scores = torch.randn(0).to(device)
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(eval_loader):
            print('Processing batch {0}...'.format(batch_idx))
            input, target = input.to(device), target.to(device)
            output = model(input)
            _, predicted_label = torch.max(output.data.detach(), 1)
            predicted_score = F.softmax(output.data.detach(), dim=1)[:, 1]
            total_eval += predicted_label.size(0)
            correct_eval += (predicted_label == target).sum().item()
            all_targets = torch.cat((all_targets, target))
            all_predicted_labels = torch.cat((all_predicted_labels, predicted_label))
            all_predicted_scores = torch.cat((all_predicted_scores, predicted_score))

    all_targets = all_targets.cpu()
    all_predicted_labels = all_predicted_labels.cpu()
    all_predicted_scores = all_predicted_scores.cpu()

    # return metrics
    import sklearn.metrics as metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    ret = {}
    ret['accuracy'] = accuracy_score(all_targets, all_predicted_labels)
    ret['precision'] = precision_score(all_targets, all_predicted_labels, average='binary')
    ret['recall'] = recall_score(all_targets, all_predicted_labels, average='binary')
    ret['f1'] = f1_score(all_targets, all_predicted_labels, average='binary')
    fpr, tpr, threshold = metrics.roc_curve(all_targets, all_predicted_scores)
    ret['fpr'] = fpr
    ret['tpr'] = tpr
    ret['correct'] = correct_eval
    ret['total'] = total_eval
    ret['targets'] = all_targets
    ret['predictions'] = all_predicted_labels
    ret['predicted_scores'] = all_predicted_scores

    return ret


# define prediction function
def predict_seed(seed_img, model, device, criterion):
    if not torch.is_tensor(seed_img) and isinstance(seed_img, np.ndarray):
        # convert seed_img to tensor
        seed_img = transforms.ToTensor()(seed_img).float()
        if len(seed_img.shape) == 2:
            # convert (H,W) to (B,C,H,W)
            seed_img = seed_img.unsqueeze(0).unsqueeze(0)
        elif len(seed_img.shape) == 3:
            # convert (C,H,W) to (B,C,H,W)
            seed_img = seed_img.unsqueeze(0)
    else:
        raise Exception('The input image for classification must be either a Tensor or an Numpy array!')
    # seed_img = test_transforms(seed_img)
    seed_img = seed_img.to(device)

    model.eval()
    with torch.no_grad():
        output = model(seed_img)
        _, predicted_label = torch.max(output.data.detach(), 1)
        predicted_score = F.softmax(output.data.detach(), dim=1)[:, 1]

    return predicted_label, predicted_score


transform_train = transforms.Compose(
    [transforms.Resize([256, 256]),
     transforms.RandomCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor()
     ])
transform_test = transforms.Compose(
    [transforms.Resize([224, 224]),
     # transforms.RandomCrop(224),
     # transforms.RandomHorizontalFlip(),
     transforms.ToTensor()
     ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_train = datasets.ImageFolder(root="DatasetTest/Train",
                                     transform=transform_train)
dataset_validation = datasets.ImageFolder(root="DatasetTest/Validation",
                                          transform=transform_test)
dataset_test = datasets.ImageFolder(root="DatasetTest/Test",
                                    transform=transform_test)

print(len(dataset_train))
print(dataset_train[0][0])
print(dataset_train[0][1])

train_loader = DataLoader(
    dataset_train,
    batch_size=8,
    sampler=RandomSampler(dataset_train),
    num_workers=0
)

val_loader = DataLoader(
    dataset_validation,
    batch_size=8,
    num_workers=0
)

# Write logs
time_of_run = time.strftime('%m-%d %H:%M:%S', time.localtime())
experiment = 'resnet' + '_pretrained_' + str(pretrained) + '_' + time_of_run
logname = os.path.join('runs', experiment)  # create a folder called runs/ before this
logname = logname.replace(":", "_")
writer = SummaryWriter(logname)

best_model = None
best_total_val = 0
best_correct_val = 0
best_model_acc = 0

lr = 0.001
sched_step_size = 5
gamma = 0.9

# http://karpathy.github.io/2019/04/25/recipe/
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=sched_step_size, gamma=gamma)

net = net.to(device)

num_epoch = 0
num_epoch_not_improved = 0
while num_epoch_not_improved < 2:
    num_epoch += 1

    # train
    print(f'Epoch number {num_epoch}')
    train_loss = train(net, device, train_loader, optimizer, criterion)
    print(f'Loss: {train_loss}')
    writer.add_scalar('Train Loss', train_loss, num_epoch)
    scheduler.step()

    # evaluate
    output_val = evaluate(net, device, val_loader, criterion)
    acc = output_val['accuracy']
    correct_val = output_val['correct']
    total_val = output_val['total']
    print('\t{0:.2f}% ({1}/{2})'.format(acc * 100, correct_val, total_val))
    writer.add_scalar('Validation Accuracy', acc, num_epoch)
    writer.add_scalar('Validation Precision', output_val['precision'], num_epoch)
    writer.add_scalar('Validation Recall', output_val['recall'], num_epoch)
    writer.add_scalar('Validation f1 score', output_val['f1'], num_epoch)

    if acc > best_model_acc:
        best_model = copy.deepcopy(net)
        best_model_acc = acc
        best_total_val = total_val
        best_correct_val = correct_val
        num_epoch_not_improved = 0
    else:
        num_epoch_not_improved += 1

    print('Current best model validation:')
    print('\t{0:.2f}% ({1}/{2})'.format(best_model_acc * 100, best_correct_val, best_total_val))
    print()
    writer.add_scalar('Best Model Accuracy', best_model_acc, num_epoch)

test_loader = DataLoader(
    dataset_test,
    batch_size=8,
    num_workers=0
)

# torch.save(best_model.state_dict(), os.path.join('Models', experiment + '_' + 'out.pth'))

print('Evaluate on training set')
output_train = evaluate(best_model, device, train_loader, criterion)
roc_auc_train = metrics.auc(output_train['fpr'], output_train['tpr'])
print('Accuracy: \t{0:.2f}% ({1}/{2})'.format(output_train['accuracy'] * 100, output_train['correct'],
                                              output_train['total']))
print('Precision: \t{0:.2f}, Recall: \t{1:.2f}, F1: \t{2:.2f}, AUC: \t{3:.2f}'.format(output_train['precision'],
                                                                                      output_train['recall'],
                                                                                      output_train['f1'],
                                                                                      roc_auc_train))

print('Evaluate on test set')
output_test = evaluate(best_model, device, test_loader, criterion)
roc_auc_test = metrics.auc(output_test['fpr'], output_test['tpr'])
print('Accuracy: \t{0:.2f}% ({1}/{2})'.format(output_test['accuracy'] * 100, output_test['correct'],
                                              output_test['total']))
print('Precision: \t{0:.2f}, Recall: \t{1:.2f}, F1: \t{2:.2f}, AUC: \t{3:.2f}'.format(output_test['precision'],
                                                                                      output_test['recall'],
                                                                                      output_test['f1'],
                                                                                      roc_auc_test))

plt.title('Receiver Operating Characteristic')
plt.plot(output_test['fpr'], output_test['tpr'], 'b', label='AUC = %0.2f' % roc_auc_test)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# add evaluation metrics in the writer as well
writer.add_pr_curve('Test pr_curve', output_test['targets'], output_test['predicted_scores'], 0)
writer.add_scalar('Test accuracy', output_test['accuracy'], 0)
writer.add_scalar('Test precision', output_test['precision'], 0)
writer.add_scalar('Test recall', output_test['recall'], 0)
writer.add_scalar('Test f1 score', output_test['f1'], 0)
writer.add_scalar('Test AUC', roc_auc_test, 0)
writer.close()
