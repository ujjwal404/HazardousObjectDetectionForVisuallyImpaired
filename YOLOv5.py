import yaml
from yolov5.models import YOLOv5
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import check_dataset, check_file, check_img_size, \
    check_requirements, check_yaml
from yolov5.utils.plots import plot_results
from yolov5.utils.metrics import fitness
from yolov5.utils.datasets import create_dataloader
from yolov5.utils import google_utils
import torch


def train(data_yaml='data.yaml', weights='yolov5s.pt', batch_size=16, img_size=[640, 640], epochs=300, device='', cache_images=False, hyp='data/hyp.scratch.yaml', resume=False, nosave=False):
    # Initialize
    with open(data_yaml) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_yaml(data_dict)  # validate yaml
    device = select_device(device)
    train_path = data_dict['train']
    nc = int(data_dict['nc'])  # number of classes
    names = data_dict['names']  # class names
    assert len(names) == nc, 'names and nc are inconsistent'
    imgsz, imgsz_test = [int(x) for x in img_size]

    # Model
    model = YOLOv5(n_classes=nc).to(device)

    # Optimizer
    hyp_dict = google_utils.parse_hyp(hyp)
    optimizer = torch.optim.SGD(model.parameters(), lr=hyp_dict['lr0'], momentum=hyp_dict['momentum'], nesterov=True, weight_decay=hyp_dict['weight_decay'])

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if resume:
        checkpoint = torch.load(weights, map_location=device)  # load checkpoint
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_fitness = checkpoint['best_fitness']
        if hyp != checkpoint['hyp']:  # check hyps
            print('\nHyperparameters from "hyp.scratch.yaml" do not match the checkpoint. Using hyps from checkpoint.\n')
            hyp_dict = checkpoint['hyp']

    # Dataset
    train_loader = create_dataloader(train_path, imgsz, batch_size, cache_images, hyp_dict, augment=True, rect=False, local_rank=device)

    # Train loop
    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(4, device=device)  # mean losses
        for i, (imgs, targets, paths, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            loss, _, _ = model(imgs, targets)  # forward
            loss.backward()  # backward
            optimizer.step()  # update weights
            optimizer.zero_grad()  # reset gradients
            mloss = (mloss * i + loss.detach().cpu()) / (i + 1)  # update mean losses

        # Log results
        if not nosave:
            log_dict = {'epoch': epoch, 'losses': mloss}
            plot_results(save_dir='runs/train', log_dict=log_dict, plot_types=('loss',), img_size=imgsz)[0].show()

        # Update best fitness
        fitness_val = fitness(model)
        if fitness_val > best_fitness:
            best_fitness = fitness_val

        # Save checkpoint
        if not nosave:
            # Save last checkpoint
            checkpoint = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'hyp': hyp_dict,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'runs/train/exp/weights/last.pt')

            # Save best checkpoint
            if best_fitness == fitness_val:
                torch.save(checkpoint, f'runs/train/exp/weights/best.pt')

    # Finish
    if not nosave:
        plot_results(save_dir='runs/train', plot_types=('loss',), img_size=imgsz)[0].show()
        print(f'\nTraining complete. Best fitness = {best_fitness:.3f}.\n')