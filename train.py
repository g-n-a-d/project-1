import torch
from yaml import dump

def train(model, lossFunction, dataloader_train, dataloader_val=None, num_epochs=50, lr=1e-5, milestones_lr=[36], gamma=0.1, device=torch.device('cpu'), display=False, log=False):
    criterion = lossFunction
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_lr, gamma=gamma)
    model.train()
    for epoch in range(num_epochs):
        log_loss = []
        for i, (r, l) in enumerate(dataloader_train):
            r = r.to(device)
            l = l.to(device)
            pred = model(r)
            loss = criterion(pred, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if display:
                print('epoch {}/{}: [{}/{}] -------> loss: {}'.format(
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    len(dataloader_train),
                    loss.item()))
               if dataloader_dev != None:
                    cm = confusionMatrix(model, dataloader_val, device)
                    print('confusion matrix:')
                    print(cm)
                    print('accuracy: {}'.format((torch.trace(cm)/cm.sum()).item()))
            log_loss.append(loss.item())
        scheduler.step()
        if log:
            with open('log_loss_{}.yaml'.format(type(model).__name__), 'a') as f:
                dump({'epoch_{}'.format(epoch + 1) : log_loss}, f, default_flow_style=False)
