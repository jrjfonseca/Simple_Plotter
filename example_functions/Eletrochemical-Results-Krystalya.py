#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from datetime import datetime

# Setup paths
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent  # The Experiments directory (two levels up from Electrochemestry)
lib_dir = parent_dir / "Notebooks" / "lib"  # lib is in Notebooks/lib, not in Electrochemestry

# Add the lib directory to the path for importing
sys.path.append(str(lib_dir))

# Import our electrochemistry module
import electrochemistry as ec

# Define data and output paths
data_dir = parent_dir / "Raw_Data" / "Eletrochemestry"
output_base_dir = parent_dir / "Processed_Data" / "Eletrochemestry"

########################################################P148########################################################
# Define the file to analyze
file_path = data_dir / "P148_CC-C20_CV-C40_Channel_8_Wb_1.xlsx"
    
# Set cell metadata for plotting
cell_metadata = {
    'cell_id': 'P148',
    'cathode_type': 'LFO-C',
    'anode_type': 'Graphite',
    'cathode_composition': '75:20:5 (AM:C:Binder)',
    'cathode_mass': '20.3 mg',      # Cathode mass information
    'anode_mass': '15.1 mg',        # Added anode mass information
    'anode_composition': '90:5:5 (AM:C:Binder)',  # Added anode composition
    'anode_mixing_method': 'Dry mixing',          # Added anode mixing method
    'electrolyte': '1M LiPF6 in EC:EMC (1:1) + 5% VC',
    'electrolyte_quantity': '200 µL',  # Added electrolyte quantity
    'channel': '8 VLC',              # Added channel information
    'voltage_range': '1.5-3.5 V',    # Added voltage range
    'c_rate': '0.05C',
    'cathode_mixing_method': 'Vortex',  # Changed from mixing_method to cathode_mixing_method
    'date': "2024-08-16",  # Current date as default
    'pressure': '2.5 ton 40 min'  # Example pressure value
}

# Load and normalize data
print("Loading and normalizing data...")
df_normalized = ec.load_data(
    file_path,
    total_mass=20.3,      # Total electrode mass in mg
    am_percentage=75,     # Active material percentage (75%)
    collector_mass=0      # Current collector mass in mg
)

# Create output directory
output_dir = output_base_dir / "P148"
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating plots...")

# Generate all plots
plots = {
    'combined_performance': ec.plot_combined_performance(
        df_normalized,
        cycles_to_plot=[1, 2, 3, 5, 10, 25, 50],
        cell_metadata=cell_metadata,
        voltage_range=[1.5, 3.5],
        capacity_range=[200, 320]
    ),
    'charge_discharge': ec.plot_charge_discharge(
        df_normalized, 
        cycles_to_plot=[1, 2, 3, 5, 10, 25, 50], 
        cell_metadata=cell_metadata,
        y_range=[1.5, 3.5]  # Set voltage range to match expected range (1.5-3.5V)
    ),
    'capacity': ec.plot_capacity_vs_cycle(
        df_normalized,
        cell_metadata=cell_metadata,
        y_range=[200, 320]  # Set capacity range from 200 to 350 mAh/g
    ),
    'soh': ec.plot_state_of_health(
        df_normalized,
        cell_metadata=cell_metadata,
        y_range=[50, 105]  # Display SOH between 50% and 105%
    ),
    'coulombic_efficiency': ec.plot_coulombic_efficiency(
        df_normalized,
        cell_metadata=cell_metadata,
        y_range=[80, 101]  # Display CE between 80% and 101%
    ),
    'voltage_time': ec.plot_voltage_time(
        df_normalized,
        cell_metadata=cell_metadata,
        y_range=[1.5, 3.5]  # Set voltage range to match expected range (1.5-3.5V)
    ),
    'differential_capacity': ec.plot_differential_capacity(
        df_normalized,
        cycles=[1],  # Plot multiple cycles
        cell_metadata=cell_metadata,
        voltage_range=[1.5, 4.0],  # Set voltage range
        polynomial_spline=3,
        s_spline=1e-5,
        polyorder_1=5,
        window_size_1=101,
        polyorder_2=5,
        window_size_2=1001,
        final_smooth=True
    ),
}

# Save all plots with high quality settings
print("Saving plots...")
for name, fig in plots.items():
    # Create filename with cell ID prefix
    file_name = f"{cell_metadata['cell_id']}_{name}"
    output_path = output_dir / file_name
    ec.save_plot(
        fig,
        output_path,
        save_png=True,
        width=1000,
        height=700,
        dpi=300
    )
    print(f"Saved {file_name} plot")

print(f"\nAll plots have been saved to: {output_dir}")
print("The following plots were generated:")
for name in plots.keys():
    file_name = f"{cell_metadata['cell_id']}_{name}"
    print(f"  - {file_name}.html and {file_name}.png")
#######################################################################################################################

########################################################P172########################################################
# Define the file to analyze
file_path = data_dir / "P172_LFO-C_GR_250312_Channel_4_Wb_1.xlsx"
    
# Set cell metadata for plotting
cell_metadata = {
    'cell_id': 'P172',
    'cathode_type': 'LFO-C',
    'anode_type': 'Li metal',
    'cathode_composition': '75:20:5 (AM:C:Binder)',
    'cathode_mass': '58.8 mg',
    'anode_mass': '---',
    'anode_composition': 'Pure Li metal',  # Added anode composition
    'anode_mixing_method': 'N/A',          # Not applicable for Li metal
    'electrolyte': '1M LiPF6 in EC:EMC (1:1) + 5% VC',
    'electrolyte_quantity': '300 µL',    # Added electrolyte quantity
    'channel': '4 VLC',                  # Added channel information
    'voltage_range': '1.5-3.5 V',        # Added voltage range
    'c_rate': '0.05C',
    'cathode_mixing_method': 'Vortex',  # Changed from mixing_method to cathode_mixing_method
    'date': "2025-03-12",
    'pressure': '2.5 ton 5 min'
}

# Load and normalize data
print("\nLoading and normalizing data for P172...")
df_normalized = ec.load_data(
    file_path,
    total_mass=58.8,      # Total electrode mass in mg
    am_percentage=75,     # Active material percentage (75%)
    collector_mass=0      # Current collector mass in mg
)

# Create output directory
output_dir = output_base_dir / "P172"
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating charge-discharge plot...")

# Generate charge-discharge plot
cd_plot = ec.plot_charge_discharge(
    df_normalized, 
    cycles_to_plot=[1, 2, 3, 5, 10, 25, 50], 
    cell_metadata=cell_metadata,
    y_range=[1.5, 3.5]  # Set voltage range to match expected range (1.5-3.5V)
)

# Save the plot
file_name = f"{cell_metadata['cell_id']}_charge_discharge"
output_path = output_dir / file_name
ec.save_plot(
    cd_plot,
    output_path,
    save_png=True,
    width=1000,
    height=700,
    dpi=300
)
print(f"Saved {file_name} plot for P172")
#######################################################################################################################

########################################################P173########################################################
# Define the file to analyze
file_path = data_dir / "P173_LFO-C_BM323_Gr_12_03_2025_Channel_5_Wb_1.xlsx"
    
# Set cell metadata for plotting
cell_metadata = {
    'cell_id': 'P173',
    'cathode_type': 'LFO-C',
    'anode_type': 'Li metal',
    'cathode_composition': '75:20:5 (AM:C:Binder)',
    'cathode_mass': '59 mg',
    'anode_mass': '---',
    'anode_composition': 'Pure Li metal',    # Added anode composition
    'anode_mixing_method': 'N/A',            # Not applicable for Li metal
    'electrolyte': '1M LiPF6 in EC:EMC (1:1) + 5% VC',
    'electrolyte_quantity': '250 µL',        # Added electrolyte quantity
    'channel': '5 VLC',                      # Added channel information
    'voltage_range': '1.5-3.5 V',            # Added voltage range
    'c_rate': '0.05C',
    'cathode_mixing_method': 'Vortex',  # Changed from mixing_method to cathode_mixing_method
    'date': "2025-03-12",
    'pressure': '2.5 ton 5 min'
}

# Load and normalize data
print("\nLoading and normalizing data for P173...")
df_normalized = ec.load_data(
    file_path,
    total_mass=59,      # Total electrode mass in mg
    am_percentage=75,     # Active material percentage (75%)
    collector_mass=0      # Current collector mass in mg
)

# Create output directory
output_dir = output_base_dir / "P173"
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating charge-discharge plot...")

# Generate charge-discharge plot
cd_plot = ec.plot_charge_discharge(
    df_normalized, 
    cycles_to_plot=[1, 2], 
    cell_metadata=cell_metadata,
    y_range=[1.5, 3.5]  # Set voltage range to match expected range (1.5-3.5V)
)

# Save the plot
file_name = f"{cell_metadata['cell_id']}_charge_discharge"
output_path = output_dir / file_name
ec.save_plot(
    cd_plot,
    output_path,
    save_png=True,
    width=1000,
    height=700,
    dpi=300
)
print(f"Saved {file_name} plot for P173")
#######################################################################################################################

########################################################P174########################################################
# Define the file to analyze
file_path = data_dir / "P174_BM323-LFO_C_Li_Channel_7_Wb_1.xlsx"
    
# Set cell metadata for plotting
cell_metadata = {
    'cell_id': 'P174',
    'cathode_type': 'LFO-C',
    'anode_type': 'Li metal',
    'cathode_composition': '75:20:5 (AM:C:Binder)',
    'cathode_mass': '27.6 mg',          # Not applicable for Li metal
    'electrolyte': '1M LiPF6 in EC:EMC (1:1) + 5% VC',    # Added electrolyte quantity
    'channel': '7 VLC',                      # Added channel information from filename
    'voltage_range': '1.5-3.5 V',            # Added voltage range
    'c_rate': '0.05C',
    'cathode_mixing_method': 'Vortex',  # Changed from mixing_method to cathode_mixing_method
    'date': "2025-03-15",
    'pressure': '2.5 ton 5 min'
}

# Load and normalize data
print("\nLoading and normalizing data for P174...")
df_normalized = ec.load_data(
    file_path,
    total_mass=27.6,      # Total electrode mass in mg
    am_percentage=75,     # Active material percentage (75%)
    collector_mass=0      # Current collector mass in mg
)

# Create output directory
output_dir = output_base_dir / "P174"
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating plots...")

# Generate plots
plots = {
    'charge_discharge': ec.plot_charge_discharge(
        df_normalized, 
        cycles_to_plot=[1, 2, 3, 4, 5, 6, 7, 8], 
        cell_metadata=cell_metadata,
        y_range=[1.5, 3.5]  # Set voltage range to match expected range (1.5-3.5V)
    ),
    'combined_performance': ec.plot_combined_performance(
        df_normalized,
        cycles_to_plot=[1, 2, 3, 5, 8],
        cell_metadata=cell_metadata,
        voltage_range=[1.5, 3.5],
        capacity_range=[100, 350]
    )
}

# Save all plots with high quality settings
print("Saving plots...")
for name, fig in plots.items():
    # Create filename with cell ID prefix
    file_name = f"{cell_metadata['cell_id']}_{name}"
    output_path = output_dir / file_name
    ec.save_plot(
        fig,
        output_path,
        save_png=True,
        width=1000,
        height=700,
        dpi=300
    )
    print(f"Saved {file_name} plot")

print(f"\nAll plots have been saved to: {output_dir}")
print("The following plots were generated:")
for name in plots.keys():
    file_name = f"{cell_metadata['cell_id']}_{name}"
    print(f"  - {file_name}.html and {file_name}.png")
#######################################################################################################################

########################################################P175########################################################
# Define the file to analyze
file_path = data_dir / "P175_BM323-LFO-C_Li_Channel_8_Wb_1.xlsx"
    
# Set cell metadata for plotting
cell_metadata = {
    'cell_id': 'P175',
    'cathode_type': 'LFO-C',
    'anode_type': 'Li metal',
    'cathode_composition': '75:20:5 (AM:C:Binder)',
    'cathode_mass': '27 mg',
    'anode_composition': 'Pure Li metal',  # Added anode composition
    'anode_mixing_method': 'N/A',          # Not applicable for Li metal
    'electrolyte': '1M LiPF6 in EC:EMC (1:1) + 5% VC',
    'channel': '8 VLC',                  # Added channel information
    'voltage_range': '1.5-3.5 V',        # Added voltage range
    'c_rate': '0.05C',
    'cathode_mixing_method': 'Vortex',  # Changed from mixing_method to cathode_mixing_method
    'date': "2025-03-15",
    'pressure': '2.5 ton 5 min'
}

# Load and normalize data
print("\nLoading and normalizing data for P175...")
df_normalized = ec.load_data(
    file_path,
    total_mass=27,      # Total electrode mass in mg
    am_percentage=75,     # Active material percentage (75%)
    collector_mass=0      # Current collector mass in mg
)

# Create output directory
output_dir = output_base_dir / "P175"
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating charge-discharge plot...")

# Generate charge-discharge plot
cd_plot = ec.plot_charge_discharge(
    df_normalized, 
    cycles_to_plot=[1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15], 
    cell_metadata=cell_metadata,
    y_range=[1.5, 3.6]  # Set voltage range to match expected range (1.5-3.5V)
)

# Generate differential capacity plot
dqdv_plot = ec.plot_differential_capacity(
    df_normalized,
    cycles=[1, 2],  # Plot cycles 1 and 2
    cell_metadata=cell_metadata,
    voltage_range=[1.5, 3.6],  # Set voltage range
    polynomial_spline=3,
    s_spline=1e-5,
    polyorder_1=5,
    window_size_1=101,
    polyorder_2=5,
    window_size_2=1001,
    final_smooth=True
)

# Generate capacity vs cycle plot
capacity_plot = ec.plot_capacity_vs_cycle(
    df_normalized,
    cell_metadata=cell_metadata,
    y_range=[150, 350]  # Set capacity range from 150 to 350 mAh/g
)

# Save all plots with high quality settings
plots = {
    'charge_discharge': cd_plot,
    'differential_capacity': dqdv_plot,
    'capacity_vs_cycle': capacity_plot,
    'combined_performance': ec.plot_combined_performance(
        df_normalized,
        cycles_to_plot=[1, 2, 3, 5, 10,14,15],
        cell_metadata=cell_metadata,
        voltage_range=[1.5, 3.6],
        capacity_range=[150, 350]
    )
}

# Save all plots with high quality settings
print("Saving plots...")
for name, fig in plots.items():
    # Create filename with cell ID prefix
    file_name = f"{cell_metadata['cell_id']}_{name}"
    output_path = output_dir / file_name
    ec.save_plot(
        fig,
        output_path,
        save_png=True,
        width=1000,
        height=700,
        dpi=300
    )
    print(f"Saved {file_name} plot")

print(f"\nAll plots have been saved to: {output_dir}")
print("The following plots were generated:")
for name in plots.keys():
    file_name = f"{cell_metadata['cell_id']}_{name}"
    print(f"  - {file_name}.html and {file_name}.png")
#######################################################################################################################

########################################################P180########################################################
# Define the file to analyze
file_path = data_dir / "P180_BM324-Mescla-BM_vsLi_Channel_7_Wb_1.xlsx"
    
# Set cell metadata for plotting
cell_metadata = {
    'cell_id': 'P180',
    'cathode_type': 'LFO-C',
    'anode_type': 'Li metal',
    'cathode_composition': '75:20:5 (AM:C:Binder)',
    'cathode_mass': '18.9 mg',
    'anode_composition': 'Pure Li metal',
    'electrolyte': '1M LiPF6 in EC:EMC (1:1) + 5% VC',
    'channel': '7 VLC',
    'voltage_range': '1.5-3.5 V',
    'c_rate': '0.05C',
    'cathode_mixing_method': 'Ball Milling',
    'date': "2025-03-31",
    'pressure': '1 ton 2 min'
}

# Load and normalize data
print("\nLoading and normalizing data for P180...")
df_normalized = ec.load_data(
    file_path,
    total_mass=18.9,      # Total electrode mass in mg
    am_percentage=75,     # Active material percentage (75%)
    collector_mass=0      # Current collector mass in mg
)

# Create output directory
output_dir = output_base_dir / "P180"
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating plots...")

# Generate all plots
plots = {
    'combined_performance': ec.plot_combined_performance(
        df_normalized,
        cycles_to_plot=[1, 2, 3,4, 5],
        cell_metadata=cell_metadata,
        voltage_range=[1.5, 3.5],
        capacity_range=[150, 350]
    ),
    'charge_discharge': ec.plot_charge_discharge(
        df_normalized, 
        cycles_to_plot=[1, 2,3,4,5], 
        cell_metadata=cell_metadata,
        y_range=[1.5, 3.5]
    ),
    'capacity': ec.plot_capacity_vs_cycle(
        df_normalized,
        cell_metadata=cell_metadata,
        y_range=[150, 350]
    ),
    'differential_capacity': ec.plot_differential_capacity(
        df_normalized,
        cycles=[1],
        cell_metadata=cell_metadata,
        voltage_range=[1.5, 3.5],
        polynomial_spline=3,
        s_spline=1e-5,
        polyorder_1=5,
        window_size_1=101,
        polyorder_2=5,
        window_size_2=1001,
        final_smooth=True
    ),
}

# Save all plots with high quality settings
print("Saving plots...")
for name, fig in plots.items():
    # Create filename with cell ID prefix
    file_name = f"{cell_metadata['cell_id']}_{name}"
    output_path = output_dir / file_name
    ec.save_plot(
        fig,
        output_path,
        save_png=True,
        width=1000,
        height=700,
        dpi=300
    )
    print(f"Saved {file_name} plot")

print(f"\nAll plots have been saved to: {output_dir}")
print("The following plots were generated:")
for name in plots.keys():
    file_name = f"{cell_metadata['cell_id']}_{name}"
    print(f"  - {file_name}.html and {file_name}.png")
#######################################################################################################################

########################################################P181########################################################
# Define the file to analyze
file_path = data_dir / "P181_LFO-C_BM324_MesclaBM_vsLi_Channel_2_Wb_1.xlsx"
    
# Set cell metadata for plotting
cell_metadata = {
    'cell_id': 'P181',
    'cathode_type': 'LFO-C',
    'anode_type': 'Li metal',
    'cathode_composition': '75:20:5 (AM:C:Binder)',
    'cathode_mass': '10.8 mg',
    'electrolyte': '1M LiPF6 in EC:EMC (1:1) + 5% VC',
    'channel': '2 VLC',
    'voltage_range': '1.5-3.5 V',
    'c_rate': '0.05C',
    'cathode_mixing_method': 'Ball Milling',
    'date': "2025-03-31",
    'pressure': '1 ton 2 min'
}

# Load and normalize data
print("\nLoading and normalizing data for P181...")
df_normalized = ec.load_data(
    file_path,
    total_mass=10.8,      # Total electrode mass in mg
    am_percentage=75,     # Active material percentage (75%)
    collector_mass=0      # Current collector mass in mg
)

# Create output directory
output_dir = output_base_dir / "P181"
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating plots...")

# Generate all plots
plots = {
    'combined_performance': ec.plot_combined_performance(
        df_normalized,
        cycles_to_plot=[1, 2 ],
        cell_metadata=cell_metadata,
        voltage_range=[1.5, 3.5],
        capacity_range=[150, 350]
    ),
    'charge_discharge': ec.plot_charge_discharge(
        df_normalized, 
        cycles_to_plot=[1, 2], 
        cell_metadata=cell_metadata,
        y_range=[1.5, 3.5]
    ),
    'capacity': ec.plot_capacity_vs_cycle(
        df_normalized,
        cell_metadata=cell_metadata,
        y_range=[150, 350]
    ),
    'differential_capacity': ec.plot_differential_capacity(
        df_normalized,
        cycles=[1],
        cell_metadata=cell_metadata,
        voltage_range=[1.5, 3.5],
        polynomial_spline=3,
        s_spline=1e-5,
        polyorder_1=5,
        window_size_1=101,
        polyorder_2=5,
        window_size_2=1001,
        final_smooth=True
    ),
}

# Save all plots with high quality settings
print("Saving plots...")
for name, fig in plots.items():
    # Create filename with cell ID prefix
    file_name = f"{cell_metadata['cell_id']}_{name}"
    output_path = output_dir / file_name
    ec.save_plot(
        fig,
        output_path,
        save_png=True,
        width=1000,
        height=700,
        dpi=300
    )
    print(f"Saved {file_name} plot")

print(f"\nAll plots have been saved to: {output_dir}")
print("The following plots were generated:")
for name in plots.keys():
    file_name = f"{cell_metadata['cell_id']}_{name}"
    print(f"  - {file_name}.html and {file_name}.png")
#######################################################################################################################

########################################################P182########################################################
# Define the file to analyze
file_path = data_dir / "P182_BM324_MesclaBM_vsLi_Channel_4_Wb_1.xlsx"
    
# Set cell metadata for plotting
cell_metadata = {
    'cell_id': 'P182',
    'cathode_type': 'LFO-C',
    'anode_type': 'Li metal',
    'cathode_composition': '75:20:5 (AM:C:Binder)',
    'cathode_mass': '13.4 mg',
    'electrolyte': '1M LiPF6 in EC:EMC (1:1) + 5% VC',
    'channel': '4 VLC',
    'voltage_range': '1.5-3.5 V',
    'c_rate': '0.05C',
    'cathode_mixing_method': 'Ball Milling',
    'date': "2025-03-31",
    'pressure': '1 ton 2 min'
}

# Load and normalize data
print("\nLoading and normalizing data for P182...")
df_normalized = ec.load_data(
    file_path,
    total_mass=13.4,      # Total electrode mass in mg
    am_percentage=75,   13,4  # Active material percentage (75%)
    collector_mass=0      # Current collector mass in mg
)

# Create output directory
output_dir = output_base_dir / "P182"
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating plots...")

# Generate all plots
plots = {
    'combined_performance': ec.plot_combined_performance(
        df_normalized,
        cycles_to_plot=[1, 2,3],
        cell_metadata=cell_metadata,
        voltage_range=[1.5, 3.5],
        capacity_range=[150, 350]
    ),
    'charge_discharge': ec.plot_charge_discharge(
        df_normalized, 
        cycles_to_plot=[1, 2,3], 
        cell_metadata=cell_metadata,
        y_range=[1.5, 3.5]
    ),
    'capacity': ec.plot_capacity_vs_cycle(
        df_normalized,
        cell_metadata=cell_metadata,
        y_range=[150, 350]
    ),
    'differential_capacity': ec.plot_differential_capacity(
        df_normalized,
        cycles=[1],
        cell_metadata=cell_metadata,
        voltage_range=[1.5, 3.5],
        polynomial_spline=3,
        s_spline=1e-5,
        polyorder_1=5,
        window_size_1=101,
        polyorder_2=5,
        window_size_2=1001,
        final_smooth=True
    ),
}

# Save all plots with high quality settings
print("Saving plots...")
for name, fig in plots.items():
    # Create filename with cell ID prefix
    file_name = f"{cell_metadata['cell_id']}_{name}"
    output_path = output_dir / file_name
    ec.save_plot(
        fig,
        output_path,
        save_png=True,
        width=1000,
        height=700,
        dpi=300
    )
    print(f"Saved {file_name} plot")

print(f"\nAll plots have been saved to: {output_dir}")
print("The following plots were generated:")
for name in plots.keys():
    file_name = f"{cell_metadata['cell_id']}_{name}"
    print(f"  - {file_name}.html and {file_name}.png")
#######################################################################################################################