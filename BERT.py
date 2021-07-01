import nuclio
from mlrun import code_to_function, mount_v3io, run_local, NewTask
import os
import pandas as pd
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertModel
import torch
import torch.nn as nn
from torch.utils import data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from collections import defaultdict
from mlrun.artifacts import PlotArtifact, ChartArtifact, TableArtifact
from mlrun.datastore import DataItem
from mlrun import MLClientCtx

# BertSentimentClassifier adopted by mlRun stock-analysis demo. Using this classifier to
#compare results from our pretrained sentiment.py to the BERT Classifier to test accuracy

class BertSentimentClassifier(nn.Module):
    def __init__(self, pretrained_model, n_classes):
        super(BertSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(p=0.2)
        self.out_linear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, pooled_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        out = self.dropout(pooled_out)
        out = self.out_linear(out)
        return self.softmax(out)

class ReviewsDataset(data.Dataset):
    def __init__(self, review, target, tokenizer, max_len):
        self.review = review
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.review)
    
    def __getitem__(self, item):
        review = str(self.review[item])
        enc = self.tokenizer.encode_plus(
            review,
            max_length=self.max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
            truncation=True)
        
        return {'input_ids': enc['input_ids'].squeeze(0), 
                'attention_mask': enc['attention_mask'].squeeze(0),
                'targets': torch.tensor(self.target[item], dtype=torch.long)}

def score_to_sents(score):
    if score <= 2:
        return 0
    if score == 3:
        return 1
    return 2


def create_data_loader(df, tokenizer, max_len, batch_size):
    dataset = ReviewsDataset(
        review=df.content.to_numpy(),
        target=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len)
    
    return data.DataLoader(dataset, batch_size=batch_size, num_workers=4)

def train_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    scheduler,
    n_examples,
    device
):
    model.train()
    losses = []
    correct_preds = 0
    
    for i, d in enumerate(data_loader):
        if i % 50 == 0:
            print(f'batch {i + 1}/ {len(data_loader)}')
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, pred = torch.max(outputs, dim=1)
        
        loss = criterion(outputs, targets)
        correct_preds += torch.sum(pred == targets)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return (correct_preds.double() / n_examples).detach().cpu().numpy(), np.mean(losses)

def eval_model(
    model,
    data_loader,
    criterion,
    n_examples,
    device
):
    print('evaluation')
    model = model.eval()
    correct_preds = 0
    losses = []
    
    with torch.no_grad():
        for i, d in enumerate(data_loader):
            if i % 50 == 0:
                print(f'batch {i + 1}/ {len(data_loader)}')
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, pred = torch.max(outputs, dim=1)

            loss = criterion(outputs, targets)
            correct_preds += torch.sum(pred == targets)
            losses.append(loss.item())
    return (correct_preds.double() / n_examples).detach().cpu().numpy(), np.mean(losses)


def eval_on_test(model_path, data_loader, device, n_examples, pretrained_model, n_classes):
    model = BertSentimentClassifier(pretrained_model, n_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct_preds = 0

    with torch.no_grad():
        for i, d in enumerate(data_loader):
            if i % 50 == 0:
                print(f'batch {i + 1}/ {len(data_loader)}')

            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, pred = torch.max(outputs, dim=1)
            correct_preds += torch.sum(pred == targets)
    return correct_preds.double() / n_examples


def train_sentiment_analysis_model(context: MLClientCtx, 
                                   reviews_dataset: DataItem,
                                   pretrained_model: str = 'sentiment.py', 
                                   models_dir: str = 'medity',
                                   model_filename: str = 'sentiment.py',
                                   n_classes: int = 3,
                                   MAX_LEN: int = 128,
                                   BATCH_SIZE: int = 16,
                                   EPOCHS: int = 50,
                                   random_state: int = 42):

    # Check for CPU or GPU 
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    base_path = os.path.abspath(context.artifact_path)
    plots_path = os.path.join(base_path, 'plots')
    data_path = os.path.join(base_path, 'data')
    context.logger.info(f'Using {device}')
    
    models_basepath = os.path.join(context.artifact_path, models_dir)
    os.makedirs(models_basepath, exist_ok=True)
    model_filepath = os.path.join(models_basepath, model_filename)
    
    # Get dataset
    df = reviews_dataset.as_df()
    
    # Save score plot
    df = df[['content', 'score']]
    sns.distplot(df.score)
    reviews_scores_artifact = context.log_artifact(PlotArtifact(f"reviews-scores", body=plt.gcf()),
                                                   target_path=f"{plots_path}/reviews-scores.html")
    
    # Turn scores to sentiment label
    df['sentiment'] = df['score'].apply(score_to_sents)
    
    # Load bert tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    # Tokenize reviews
    lens = [len(tokenizer.encode(df.loc[review]['content'])) for review in df.index]
    max_length = max(lens)
    context.logger.info(f'longest review: {max_length}')
    plt.clf()
    sns.distplot(lens)
    reviews_lengths_artifact = context.log_artifact(PlotArtifact(f"reviews-lengths", body=plt.gcf()),
                                                    target_path=f"{plots_path}/reviews-lengths.html")
    
    # Create training and validation datasets
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state)
    df_dev, df_test = train_test_split(df_test, test_size = 0.5, random_state=random_state)
    
    # Create dataloaders for all datasets
    train_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    dev_loader = create_data_loader(df_dev, tokenizer, MAX_LEN, BATCH_SIZE)
    test_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    
    # Load the bert sentiment classifier base
    model = BertSentimentClassifier(pretrained_model, n_classes=n_classes).to(device)
    
    # training
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss().to(device)
    
    history = defaultdict(list)
    best_acc = train_acc = train_loss = dev_acc = dev_loss = 0

    context.logger.info('Started training the model')
    for epoch in range(EPOCHS):
        train_acc, train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            len(df_train),
            device
        )
        
        dev_acc, dev_loss = eval_model(
            model,
            dev_loader,
            criterion,
            len(df_dev),
            device
        )

        # Append results to history
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['dev_acc'].append(dev_acc)
        history['dev_loss'].append(dev_loss)
        context.logger.info(f'Epoch: {epoch + 1}/{EPOCHS}: Train loss: {train_loss}, accuracy: {train_acc} Val loss: {dev_loss}, accuracy: {dev_acc}')

        if dev_acc > best_acc:
            torch.save(model.state_dict(), model_filepath)
            context.logger.info(f'Updating model, Current models is better then the previous one ({best_acc} vs. {dev_acc}).')
            best_acc = dev_acc
    
    context.logger.info('Finished training, testing and logging results')
    chart = ChartArtifact('summary')
    chart.header = ['epoch', 'accuracy', 'val_accuracy', 'loss', 'val_loss']
    for i in range(len(history['train_acc'])):
        chart.add_row([i + 1, history['train_acc'][i],
                       history['train_loss'][i],
                       history['dev_acc'][i],
                       history['dev_loss'][i]])
    summary = context.log_artifact(chart, local_path=os.path.join('plots', 'summary.html'))

    history_df = pd.DataFrame(history)
    history_table = TableArtifact('history', df=history_df)
    history_artifact = context.log_artifact(history_table, target_path=os.path.join(data_path, 'history.csv'))

    test_acc = eval_on_test(model_filepath, test_loader, device, len(df_test), pretrained_model, n_classes)
    context.logger.info(f'Received {test_acc} on test dataset')
    results = {'train_accuracy': train_acc,
               'train_loss': train_loss,
               'best_acccuracy': best_acc,
               'validation_accuracy': dev_acc,
               'validation_loss': dev_loss}
    context.log_results(results)
    context.log_model(key='sentiment.py',
                      model_file=model_filename,
                      model_dir=models_dir,
                      artifact_path=context.artifact_path,
                      upload=False,
                      labels={'framework': 'pytorch',
                              'category': 'nlp',
                              'action': 'sentiment_analysis'},
                      metrics=context.results,
                      parameters={'pretrained_model': pretrained_model,
                                  'MAX_LEN': MAX_LEN,
                                  'BATCH_SIZE': BATCH_SIZE,
                                  'EPOCHS': EPOCHS,
                                  'random_state': random_state},
                      extra_data={'reviews_scores': reviews_scores_artifact,
                                  'reviews_length': reviews_lengths_artifact,
                                  'training_history': history_artifact})


reviews_datafile = os.path.join(os.path.abspath('..'), 'data', 'tweet_data.csv')

#feeding in model that we created
pretrained_model = 'sentiment.py'

task = NewTask(params={'pretrained_model': pretrained_model,
                       'EPOCHS': 1},
               inputs={'reviews_dataset': reviews_datafile})
lrun = run_local(task, handler=train_sentiment_analysis_model)