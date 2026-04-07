from models.wide_resnet import wide_resnet_28_10
from models.resnet import resnet18


def get_classifier(P, n_classes=10):
    if P.model == 'pre_resnet18':
        if P.dataset == 'tinyimagenet':
            classifier = resnet18(num_classes=n_classes, stride=2)
        else:
            classifier = resnet18(num_classes=n_classes)
    elif P.model == 'wrn3410':
        if P.dataset == 'tinyimagenet':
            classifier = wide_resnet_28_10(num_classes=n_classes, stride=2)
        else:
            classifier = wide_resnet_28_10(num_classes=n_classes)
    else:
        raise NotImplementedError()

    return classifier
