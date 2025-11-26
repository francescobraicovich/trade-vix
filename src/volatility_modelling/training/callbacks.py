import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, mode='min', min_delta=0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'min' and score > self.best_score - self.min_delta) or              (self.mode == 'max' and score < self.best_score + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(self, model, score):
        if self.save_best_only:
            if (self.mode == 'min' and score < self.best_score) or                (self.mode == 'max' and score > self.best_score):
                self.best_score = score
                torch.save(model.state_dict(), self.filepath)
        else:
            torch.save(model.state_dict(), self.filepath)
