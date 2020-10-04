from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from metrics import *
from datasets import *
from collections import OrderedDict


class CNN(nn.Module):
    """CNN."""

    def __init__(self, model_arch="resnet50", n_classes=2, include_top=False, pretrained=False, lower_features=False):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.include_top = include_top
        self.pretrained = pretrained
        self.lower_features = lower_features

        self.gradients = None
        self.classifier = None

        if (model_arch == "resnet50"):
            self.model = models.resnet50(pretrained=True)

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, n_classes)

            self.features = nn.Sequential(*list(self.model.children())[:-1])
            #print(self.features)
            self.features_dict = OrderedDict()

        elif (model_arch == "resnet101"):
            self.model = models.resnet101(pretrained=True)
            #print(self.model)

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, n_classes)

            self.features_dict = OrderedDict()
            if (lower_features == True):
                self.model = nn.Sequential(*list(self.model.children())[:5])
            else:
                self.model = nn.Sequential(*list(self.model.children())[:-2])

        elif (model_arch == "squeezenet"):
            self.model = models.squeezenet1_1(pretrained=True)
            #print(self.model)
            #self.classifier = self.model.classifier

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            #num_ftrs = self.model.fc.in_features
            #self.model.fc = torch.nn.Linear(num_ftrs, n_classes)
            #num_ftrs = 512
            #self.model.classifier[-1] = torch.nn.Linear(num_ftrs, n_classes)

            self.features_dict = OrderedDict()
            if (lower_features == True):
                self.model = nn.Sequential(self.model.features[:6])
            else:
                self.model = nn.Sequential(self.model.features)

            #print(self.model)
            #exit()
        elif (model_arch == "densenet121"):
            self.model = models.densenet121(pretrained=True)

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(num_ftrs, n_classes)
            self.features = nn.Sequential(*list(self.model.children())[:-1])
            print(self.model)

        elif (model_arch == "vgg19"):
            self.model = models.vgg19(pretrained=True)

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.classifier[0].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, n_classes)

            self.features = nn.Sequential(*list(self.model.children())[:-1])
            #print(self.features)
            print(self.model)

        elif (model_arch == "vgg16"):
            self.model = models.vgg16(pretrained=True);

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.classifier[0].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, n_classes)

            if(lower_features == True):
                self.model = nn.Sequential(self.model.features[:5])
            else:
                self.model = nn.Sequential(*list(self.model.children())[:-2])

            #print(self.features)
            #print(self.model)
            #exit()
            print(self.model)
            self.features_dict = OrderedDict()

        elif (model_arch == "mobilenet"):
            self.model = models.mobilenet_v2(pretrained=True);

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, n_classes)

            if(lower_features == True):
                #self.model = nn.Sequential(self.model.features[:5])
                self.model = nn.Sequential(*list(self.model.features)[:5])
            else:
                #self.model = nn.Sequential(*list(self.model.children())[:-1])
                self.model = nn.Sequential(*list(self.model.features))

            self.features_dict = OrderedDict()

        elif (model_arch == "alexnet"):
            self.model = models.alexnet(pretrained=True)

            for params in self.model.parameters():
                params.requires_grad = self.pretrained

            num_ftrs = self.model.classifier[0].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, n_classes)

            self.features = nn.Sequential(*list(self.model.children())[:-1])
            #print(self.features)
            print(self.model)

        else:
            self.model_arch = None
            print("No valid backbone cnn network selected!")

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def forward(self, x):
        """Perform forward."""
        if(self.include_top == False):
            # extract features
            x = self.model(x)
            self.features_dict['out'] = x
            self.features_dict['aux'] = x
            return self.features_dict

        elif(self.include_top == True):
            #print(x.size())
            x = self.model(x)

            # flatten
            x = x.view(x.size(0), -1)

            x = self.classifier(x)
            self.features_dict['out'] = x

            return self.features_dict

        return x

def loadModel(model_arch="", classes=None, pre_trained_path=None, expType=None, trainable_backbone_flag=False, lower_features=False):
    print("Load model architecture ... ")

    if (model_arch == "deeplabv3_resnet101_orig"):
        print("deeplab_resnet architecture selected ...")
        model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)

        for params in model.parameters():
            params.requires_grad = trainable_backbone_flag

        model.classifier[-1] = torch.nn.Conv2d(256, len(classes), kernel_size=(1, 1))
        model.aux_classifier[-1] = torch.nn.Conv2d(256, len(classes), kernel_size=(1, 1))
        features = model.backbone

        if (pre_trained_path != None):
            print("load pre-trained-weights ... ")
            model_dict_state = torch.load(pre_trained_path) # + "/best_model.pth")
            model.load_state_dict(model_dict_state['net'])
        return model, features

    elif (model_arch == "fcn_resnet101_orig"):
        print("deeplab_resnet architecture selected ...")
        model = models.segmentation.fcn_resnet101(pretrained=True, progress=True)

        for params in model.parameters():
            params.requires_grad = trainable_backbone_flag

        model.classifier[-1] = torch.nn.Conv2d(512, len(classes), kernel_size=(1, 1))
        model.aux_classifier[-1] = torch.nn.Conv2d(256, len(classes), kernel_size=(1, 1))
        features = model.backbone

        if (pre_trained_path != None):
            print("load pre-trained-weights ... ")
            model_dict_state = torch.load(pre_trained_path)# + "/best_model.pth")
            model.load_state_dict(model_dict_state['net'])

        return model, features

    elif (model_arch == "deeplabv3_resnet101"):
        print("deeplabv3_resnet101 architecture selected ...")
        backbone_net = CNN(model_arch="resnet101", n_classes=len(classes), include_top=False, pretrained=trainable_backbone_flag, lower_features=lower_features)

        if (lower_features == True):
            classifier = nn.Sequential(
                DeepLabHead(256, len(classes)),
                # nn.Softmax()
            )
        else:
            classifier = nn.Sequential(
                DeepLabHead(2048, len(classes)),
                # nn.Softmax()
            )

        features = backbone_net
        model = models.segmentation.DeepLabV3(backbone=backbone_net, classifier=classifier, aux_classifier=None)

        if (pre_trained_path != None):
            print("load pre-trained-weights ... ")
            model_dict_state = torch.load(pre_trained_path)# + "/best_model.pth")
            model.load_state_dict(model_dict_state['net'])
        return model, features

    elif (model_arch == "deeplabv3_vgg16"):
        print("deeplabv3_vgg architecture selected ...")
        # backbone_net = CNN(model_arch="resnet101", n_classes=len(classes), include_top=False)
        backbone_net = CNN(model_arch="vgg16", n_classes=len(classes), include_top=False, pretrained=trainable_backbone_flag, lower_features=lower_features)

        if (lower_features == True):
            classifier = nn.Sequential(
                DeepLabHead(64, len(classes)),
                # nn.Softmax()
            )
        else:
            classifier = nn.Sequential(
                DeepLabHead(512, len(classes)),
                # nn.Softmax()
            )

        features = backbone_net
        model = models.segmentation.DeepLabV3(backbone=backbone_net, classifier=classifier, aux_classifier=None)
        #print(model)
        #exit()
        if (pre_trained_path != None):
            print("load pre-trained-weights ... ")
            model_dict_state = torch.load(pre_trained_path)  # + "/best_model.pth")
            model.load_state_dict(model_dict_state['net'])

        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print("total_params:" + str(total_params))
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print("total_trainable_params: " + str(total_trainable_params))
        #exit()

        return model, features

    elif (model_arch == "deeplabv3_mobilenet"):
        print("deeplabv3_mobilenet architecture selected ...")
        backbone_net = CNN(model_arch="mobilenet", n_classes=len(classes), include_top=False, pretrained=trainable_backbone_flag, lower_features=lower_features)

        if (lower_features == True):
            classifier = nn.Sequential(
                DeepLabHead(32, len(classes)),
                # nn.Softmax()
            )
        else:
            classifier = nn.Sequential(
                DeepLabHead(1280, len(classes)),
                # nn.Softmax()
            )

        features = backbone_net
        model = models.segmentation.DeepLabV3(backbone=backbone_net, classifier=classifier, aux_classifier=None)

        if (pre_trained_path != None):
            print("load pre-trained-weights ... ")
            model_dict_state = torch.load(pre_trained_path)
            model.load_state_dict(model_dict_state['net'])

        return model, features

    elif (model_arch == "deeplabv3_squeezenet"):
        print("deeplabv3_mobilenet architecture selected ...")
        backbone_net = CNN(model_arch="squeezenet", n_classes=len(classes), include_top=False, pretrained=trainable_backbone_flag, lower_features=lower_features)

        if (lower_features == True):
            classifier = nn.Sequential(
                DeepLabHead(128, len(classes)),
                # nn.Softmax()
            )
        else:
            classifier = nn.Sequential(
                DeepLabHead(512, len(classes)),
                # nn.Softmax()
            )

        features = backbone_net
        model = models.segmentation.DeepLabV3(backbone=backbone_net, classifier=classifier, aux_classifier=None)

        if (pre_trained_path != None):
            print("load pre-trained-weights ... ")
            model_dict_state = torch.load(pre_trained_path)# + "/best_model.pth")
            model.load_state_dict(model_dict_state['net'])

        return model, features

    elif (model_arch == "fcn_vgg16"):
        print("fcn_vgg16 architecture selected ...")
        backbone_net = CNN(model_arch="vgg16", n_classes=len(classes), include_top=False, pretrained=trainable_backbone_flag, lower_features=lower_features)

        if(lower_features == True):
            classifier = nn.Sequential(
                FCNHead(64, len(classes)),
                # nn.Softmax()
            )
        else:
            classifier = nn.Sequential(
                FCNHead(512, len(classes)),
                # nn.Softmax()
            )
        features = backbone_net
        model = models.segmentation.FCN(backbone=backbone_net, classifier=classifier, aux_classifier=None)
        # print(model)

        if (pre_trained_path != None):
            print("load pre-trained-weights ... ")
            model_dict_state = torch.load(pre_trained_path)# + "/best_model.pth")
            model.load_state_dict(model_dict_state['net'])

        return model, features
    elif (model_arch == "fcn_resnet101"):
        print("fcn_resnet101 architecture selected ...")
        backbone_net = CNN(model_arch="resnet101", n_classes=len(classes), include_top=False, pretrained=trainable_backbone_flag, lower_features=lower_features)

        if (lower_features == True):
            classifier = nn.Sequential(
                FCNHead(256, len(classes)),
                # nn.Softmax()
            )
        else:
            classifier = nn.Sequential(
                FCNHead(2048, len(classes)),
                # nn.Softmax()
            )

        features = backbone_net
        model = models.segmentation.FCN(backbone=backbone_net, classifier=classifier, aux_classifier=None)

        if (pre_trained_path != None):
            print("load pre-trained-weights ... ")
            model_dict_state = torch.load(pre_trained_path)  # + "/best_model.pth")
            model.load_state_dict(model_dict_state['net'])

        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print("total_params:" + str(total_params))
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print("total_trainable_params: " + str(total_trainable_params))
        #exit()

        return model, features

    elif (model_arch == "fcn_squeezenet"):
        print("deeplabv3_squeezenet architecture selected ...")
        backbone_net = CNN(model_arch="squeezenet", n_classes=len(classes), include_top=False, pretrained=trainable_backbone_flag, lower_features=lower_features)

        if (lower_features == True):
            classifier = nn.Sequential(
                FCNHead(128, len(classes)),
                # nn.Softmax()
            )
        else:
            classifier = nn.Sequential(
                FCNHead(512, len(classes)),
                # nn.Softmax()
            )

        features = backbone_net
        model = models.segmentation.FCN(backbone=backbone_net, classifier=classifier, aux_classifier=None)

        if (pre_trained_path != None):
            print("load pre-trained-weights ... ")
            model_dict_state = torch.load(pre_trained_path)# + "/best_model.pth")
            model.load_state_dict(model_dict_state['net'])

        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print("total_params:" + str(total_params))
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print("total_trainable_params: " + str(total_trainable_params))
        # exit()
        return model, features

    elif (model_arch == "fcn_mobilenet"):
        print("deeplabv3_mobilenet architecture selected ...")
        backbone_net = CNN(model_arch="mobilenet", n_classes=len(classes), include_top=False, pretrained=trainable_backbone_flag, lower_features=lower_features)

        if (lower_features == True):
            classifier = nn.Sequential(
                FCNHead(32, len(classes)),
                # nn.Softmax()
            )
        else:
            classifier = nn.Sequential(
                FCNHead(1280, len(classes)),
                # nn.Softmax()
            )

        features = backbone_net
        model = models.segmentation.FCN(backbone=backbone_net, classifier=classifier, aux_classifier=None)

        if (pre_trained_path != None):
            print("load pre-trained-weights ... ")
            model_dict_state = torch.load(pre_trained_path)# + "/best_model.pth")
            model.load_state_dict(model_dict_state['net'])

        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print("total_params:" + str(total_params))
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print("total_trainable_params: " + str(total_trainable_params))
        # exit()
        return model, features

    else:
        print("ERROR: select valid model architecture!")
        exit()
