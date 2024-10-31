import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from loader import *
from spectrumtransformer import *
from utils import *

input_dim = 4096
param_dim = 3
output_dim = 4096
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_pp_500_11750 = SpectrumTransformer(input_dim = input_dim, param_dim = param_dim, output_dim = output_dim, 
                                         nhead = 16, num_encoder_layers = 6)
model_pp_11750_20000 = SpectrumTransformer(input_dim=input_dim, param_dim = param_dim, output_dim = output_dim, 
                                           nhead = 16, num_encoder_layers = 6)

for file_index in range(1, 8):
    print(f"Processing Dataset {file_index}/7")
    
    if file_index <= 4:
        model = model_pp_500_11750
        initial_lr = 1e-5 * 0.75 ** (file_index - 1)
        batch_size = 16
        num_epochs = 75
    else:
        model = model_pp_11750_20000
        initial_lr = 1e-5 * 0.75 ** (file_index - 5)
        batch_size = 8
        num_epochs = 100

    criterion = nn.SmoothL1Loss
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay = 1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, patience = 3, verbose = True)

    train_data, val_data, test_data, scalers = load_data(file_index)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

    train_model(model, file_index, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, patience=8)
    avg_test_loss, _, _ = evaluate_model(model, test_loader, criterion, scalers)
    print(f"Dataset {file_index} Test Loss: {avg_test_loss:.4f}")