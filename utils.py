import torch
import numpy as np
import matplotlib as plt

def train_model(model, index, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for parameters, wavelengths, spectra in train_loader:
            parameters, wavelengths, spectra = parameters.to(device), wavelengths.to(device), spectra.to(device)
            inputs = wavelengths.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs, parameters)
            loss = criterion(outputs, spectra)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for parameters, wavelengths, spectra in val_loader:
                parameters, wavelengths, spectra = parameters.to(device), wavelengths.to(device), spectra.to(device)
                inputs = wavelengths.unsqueeze(1)
                outputs = model(inputs, parameters)
                loss = criterion(outputs, spectra)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.8f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            if index <= 4:
                torch.save(model.state_dict(), 'model_pp_500_11750.pth')
            else:
                torch.save(model.state_dict(), 'model_pp_11750_20000.pth')
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss. Early stopping counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

def plot_comparison(predictions, targets, parameters, wavelengths, num_samples = 1):
    indices = np.random.choice(len(predictions), num_samples, replace = False)
    for i, idx in enumerate(indices):
        plt.figure(figsize = (9, 6))
        plt.plot(wavelengths[idx], targets[idx], label = 'True Spectrum', color = 'blue')
        plt.plot(wavelengths[idx], predictions[idx], label = 'Predicted Spectrum', color = 'red', 
                 linestyle = 'dashed')
        
        params_text = f"Pp: {parameters[idx][0]:.2f}\nλ0: {parameters[idx][1]:.2f}\nFWHM: {parameters[idx][2]:.2f}"
        plt.text(0.05, 0.95, params_text, transform = plt.gca().transAxes, 
             fontsize = 10, verticalalignment = 'top', bbox = dict(facecolor = 'white', alpha = 0.5))
        
        plt.xlabel("Wavelengths (nm)")
        plt.ylabel("Spectral Intensity (dB)")
        plt.legend()
        plt.show()

def evaluate_model(model, test_loader, criterion, scalers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loss = 0.0
    all_predictions = []
    all_targets = []
    all_parameters = []
    all_wavelengths = []

    with torch.no_grad():
        for parameters, wavelengths, spectra in test_loader:
            parameters, wavelengths, spectra = parameters.to(device), wavelengths.to(device), spectra.to(device)
            inputs = wavelengths.unsqueeze(1)
            outputs = model(inputs, parameters)
            loss = criterion(outputs, spectra)
            test_loss += loss.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(spectra.cpu().numpy())
            all_parameters.append(parameters.cpu().numpy())
            all_wavelengths.append(wavelengths.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.4f}")

    all_predictions = np.concatenate(all_predictions, axis = 0)
    all_targets = np.concatenate(all_targets, axis = 0)
    all_parameters = np.concatenate(all_parameters, axis = 0)
    all_wavelengths = np.concatenate(all_wavelengths, axis = 0)

    parameters_og = scalers['parameters'].inverse_transform(all_parameters)
    wavelengths_og = scalers['wavelengths'].inverse_transform(all_wavelengths)
    targets_og = scalers['spectra'].inverse_transform(all_targets)
    predictions_og = scalers['spectra'].inverse_transform(all_predictions)

    plot_comparison(predictions_og, targets_og, parameters_og, wavelengths_og)

    return avg_test_loss, all_predictions, all_targets