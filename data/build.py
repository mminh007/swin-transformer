# function to create data (CIFAR 10)
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch
import torchvision


def build_dataloader(args):
    """
    """
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.imgsz, args.imgsz), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.imgsz, interpolation=InterpolationMode.BILINEAR),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)


    return trainloader, testloader


def build_dataset():
    # function build custom dataset
    pass

