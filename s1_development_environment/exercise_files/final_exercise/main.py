import click
import torch
from model import MyAwesomeModel

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=5, help="number of epochs to train for")
def train(lr, epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr, epochs)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        running_loss = 0
        totalAcc = 0
        count = 0
        for images, labels in train_set():
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=0)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))  
            totalAcc = totalAcc + accuracy.item()
            count = count+1
        assert count != 0
        print(f'Accuracy: {totalAcc/count*100}%')
        print(f'Loss: {running_loss/count}')
    
    checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}
    torch.save(checkpoint, "mymodel.pth")




@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    checkpoint = torch.load(model_checkpoint)
    model = MyAwesomeModel(checkpoint['input_size'],
                             checkpoint['hidden_layers'][0],
                             checkpoint['hidden_layers'][1],
                             checkpoint['hidden_layers'][2],
                             checkpoint['output_size'],
                             )
    model.load_state_dict(checkpoint['state_dict'])
    _, test_set = mnist()
    criterion = torch.nn.NLLLoss()

    with torch.no_grad():
            count=0
            totalAcc = 0
            running_loss = 0
            for images, labels in test_set():
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                running_loss += loss.item()
                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=0)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))  
                totalAcc = totalAcc + accuracy.item()
                count = count+1
    print(f'Accuracy: {totalAcc/count*100}%')
    print(f'Loss: {running_loss/count}')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
