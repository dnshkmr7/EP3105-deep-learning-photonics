import pandas as pd

parameters = pd.read_csv('/parameters.csv')
wavelength = pd.read_csv('/wavelength.csv')
spectra = pd.read_csv('/spectra.csv')

num_splits = 7
rows_per_split = len(parameters) // num_splits

for i in range(num_splits):
    start_row = i * rows_per_split
    end_row = (i + 1) * rows_per_split
    
    parameters_split = parameters.iloc[start_row:end_row]
    wavelength_split = wavelength.iloc[start_row:end_row]
    spectra_split = spectra.iloc[start_row:end_row]
    
    parameters_split.to_csv(f'parameters_split_{i+1}.csv', index = False)
    wavelength_split.to_csv(f'wavelength_split_{i+1}.csv', index = False)
    spectra_split.to_csv(f'spectra_split_{i+1}.csv', index = False)