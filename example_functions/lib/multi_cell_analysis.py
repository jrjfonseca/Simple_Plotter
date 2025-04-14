import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import plotly.colors as pc
import os
import sys
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter

# Import our electrochemistry module
from lib import electrochemistry as ec

def load_multiple_cells(file_paths: list, cell_metadata_list: list, 
                        masses: list = None, am_percentages: list = None, collector_masses: list = None) -> dict:
    """Load data for multiple cells and return a dictionary with normalized data.
    
    Args:
        file_paths: List of paths to cell data files
        cell_metadata_list: List of dictionaries with cell metadata
        masses: List of total electrode masses in mg
        am_percentages: List of active material percentages (0-100)
        collector_masses: List of current collector masses in mg
        
    Returns:
        Dictionary with cell_ids as keys and normalized dataframes as values
    """
    if len(file_paths) != len(cell_metadata_list):
        raise ValueError("Number of file paths must match number of cell metadata dictionaries")
    
    if masses and (len(masses) != len(file_paths)):
        raise ValueError("If provided, number of masses must match number of file paths")
        
    if am_percentages and (len(am_percentages) != len(file_paths)):
        raise ValueError("If provided, number of am_percentages must match number of file paths")
        
    if collector_masses and (len(collector_masses) != len(file_paths)):
        raise ValueError("If provided, number of collector_masses must match number of file paths")
    
    cell_data = {}
    
    for i, file_path in enumerate(file_paths):
        metadata = cell_metadata_list[i]
        cell_id = metadata.get('cell_id', f'Cell_{i+1}')
        
        # Extract mass parameters from metadata if not provided explicitly
        total_mass = masses[i] if masses else float(metadata.get('cathode_mass', '0').split()[0]) if 'cathode_mass' in metadata and metadata['cathode_mass'] != 'Not measured' else 0
        am_percentage = am_percentages[i] if am_percentages else float(metadata.get('cathode_composition', '').split(':')[0]) if 'cathode_composition' in metadata else 75
        collector_mass = collector_masses[i] if collector_masses else 0
        
        print(f"Loading data for {cell_id}...")
        df_normalized = ec.load_data(
            file_path,
            total_mass=total_mass,
            am_percentage=am_percentage,
            collector_mass=collector_mass
        )
        
        # Add cell_id as a column to the dataframe for identification
        df_normalized['cell_id'] = cell_id
        
        cell_data[cell_id] = {
            'data': df_normalized,
            'metadata': metadata
        }
    
    return cell_data

def load_paper_data(file_path: Path, metadata: dict) -> dict:
    """Load data from a paper's Excel file.
    
    Args:
        file_path: Path to the Excel file containing paper data
        metadata: Dictionary containing cell metadata
        
    Returns:
        Dictionary with the same structure as load_multiple_cells
    """
    import pandas as pd
    
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Print column names and their types for debugging
    print("\nAvailable columns in Excel file:", df.columns.tolist())
    print("\nColumn names with their exact representation:")
    for col in df.columns:
        print(f"'{col}' (length: {len(col)}, ascii: {[ord(c) for c in col]})")
    
    # Clean column names by removing any non-standard whitespace and normalizing
    df.columns = [col.strip().replace('\u202f', ' ').replace('\xa0', ' ') for col in df.columns]
    
    try:
        # Create a normalized dataframe with the same structure as our experimental data
        df_normalized = pd.DataFrame({
            'Cycle_Index': df['Cycle index'.strip()],  # Remove any trailing spaces
            'Capacity': df['Discharge (mAh/g)'],  # This column name is correct
            'Step_Type': 'Discharge'  # All data is discharge capacity
        })
        
        return {
            'data': df_normalized,
            'metadata': metadata
        }
    except KeyError as e:
        print(f"\nError accessing column: {e}")
        print("Make sure the Excel file has exactly these column names:")
        print("- 'Cycle index'")
        print("- 'Discharge (mAh/g)'")
        print("\nActual column names in file:")
        for col in df.columns:
            print(f"- '{col}'")
        raise

def compare_capacity_vs_cycle(cell_data: dict, paper_data: dict = None, x_range: list = None, y_range: list = None, 
                              specific_cycles: list = None, title: str = "Capacity Comparison",
                              exclude_last_cycle: bool = False) -> go.Figure:
    """Plot discharge capacity vs cycle number for multiple cells.
    
    Args:
        cell_data: Dictionary with cell_ids as keys and dictionaries containing data and metadata as values
        paper_data: Optional dictionary with paper data
        x_range: Optional list specifying [min, max] for cycle number axis
        y_range: Optional list specifying [min, max] for capacity axis
        specific_cycles: Optional list of specific cycles to highlight with markers
        title: Title for the plot
        exclude_last_cycle: If True, excludes the last cycle from each cell's data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Create color scale for cells
    colors = pc.qualitative.Plotly + pc.qualitative.Light24 + pc.qualitative.Dark24
    
    # Process each cell
    for i, (cell_id, cell_info) in enumerate(cell_data.items()):
        df = cell_info['data']
        metadata = cell_info['metadata']
        
        # Calculate discharge capacity per cycle
        discharge_capacity = df[df['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max()
        
        # Skip cells with no discharge data
        if discharge_capacity.empty:
            print(f"Warning: No discharge data found for {cell_id}, skipping capacity plot.")
            continue
            
        # Exclude last cycle if requested
        if exclude_last_cycle and len(discharge_capacity) > 1:
            discharge_capacity = discharge_capacity.iloc[:-1]
            
        # Plot the capacity curve as scatter
        fig.add_trace(
            go.Scatter(
                x=discharge_capacity.index,
                y=discharge_capacity.values,
                name=metadata.get('label', cell_id),  # Use custom label if available, fallback to cell_id
                mode='markers',
                marker=dict(
                    color=colors[i % len(colors)],
                    size=12,
                    line=dict(color='white', width=1)
                ),
            )
        )
    
    # Add paper data if provided
    if paper_data:
        for i, (paper_id, paper_info) in enumerate(paper_data.items()):
            df = paper_info['data']
            metadata = paper_info['metadata']
            
            # Plot paper data with a different style
            fig.add_trace(
                go.Scatter(
                    x=df['Cycle_Index'],
                    y=df['Capacity'],
                    name=metadata.get('label', paper_id),
                    mode='markers',  # Change from 'lines' to 'markers'
                    marker=dict(
                        color=colors[(i + len(cell_data)) % len(colors)],
                        size=12,
                        symbol='diamond',  # Use diamond markers to distinguish from experimental data
                        line=dict(color='white', width=1)
                    ),
                )
            )
    
    # Create title
    title_text = f"<b>{title}</b><br>"
    
    # Add metadata for each cell
    for cell_id, cell_info in cell_data.items():
        metadata = cell_info['metadata']
        # First line: Basic cell information
        title_text += f"<b>{cell_id}</b>: Cathode {metadata.get('cathode_type', 'Unknown')} ({metadata.get('cathode_composition', 'Unknown')}), "
        title_text += f"Anode: {metadata.get('anode_type', 'Unknown')}, "
        title_text += f"Cathode Mass = {metadata.get('cathode_mass', 'Unknown')}, "
        title_text += f"C-rate: {metadata.get('c_rate', 'Unknown')}, "
        title_text += f"Voltage: {metadata.get('voltage_range', 'Unknown')}<br>"
        # Second line: Electrolyte
        title_text += f"Electrolyte: {metadata.get('electrolyte', 'Unknown')}<br>"
    
    # Add paper data metadata if present (on the same line)
    if paper_data:
        for paper_id, paper_info in paper_data.items():
            metadata = paper_info['metadata']
            title_text += f"<b>Literature data:</b> {metadata.get('reference', 'Unknown')}"
    
    # Update layout with better axis visibility
    fig.update_layout(
        title={
            'text': title_text,
            'y': 0.98,  # Move title closer to top
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 14}
        },
        xaxis_title="Cycle Number",
        yaxis_title="Discharge Capacity (mAh/g)",
        legend=dict(
            orientation="v",  # Vertical orientation
            yanchor="top",  # Anchor to top
            y=0.95,  # Position at top of plot area
            xanchor="right",  # Anchor to right
            x=0.99,  # Position at right edge of plot area
            bgcolor='rgba(255, 255, 255, 0.8)',  # Semi-transparent white background
            bordercolor='rgba(0, 0, 0, 0.2)',  # Light border
            borderwidth=1
        ),
        margin=dict(t=250, b=80, l=80, r=80),  # Increased margins
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            range=x_range if x_range else None,
            showline=True,  # Ensure x-axis line is visible
            linewidth=2,
            linecolor='black',
            mirror=True  # Show axis on both sides
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            range=y_range if y_range else None,
            showline=True,  # Ensure y-axis line is visible
            linewidth=2,
            linecolor='black',
            mirror=True  # Show axis on both sides
        )
    )
    
    return fig

def compare_coulombic_efficiency(cell_data: dict, y_range: list = None, 
                                title: str = "Coulombic Efficiency Comparison") -> go.Figure:
    """Plot coulombic efficiency vs cycle number for multiple cells.
    
    Args:
        cell_data: Dictionary with cell_ids as keys and dictionaries containing data and metadata as values
        y_range: Optional list specifying [min, max] for efficiency axis
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Create color scale for cells
    colors = pc.qualitative.Plotly + pc.qualitative.Light24 + pc.qualitative.Dark24
    
    # Process each cell
    for i, (cell_id, cell_info) in enumerate(cell_data.items()):
        df = cell_info['data']
        metadata = cell_info['metadata']
        
        # Calculate coulombic efficiency per cycle
        discharge_capacity = df[df['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max()
        charge_capacity = df[df['Step_Type'] == 'Charge'].groupby('Cycle_Index')['Capacity'].max()
        
        # Skip cells with insufficient data
        if discharge_capacity.empty or charge_capacity.empty:
            print(f"Warning: Missing charge or discharge data for {cell_id}, skipping coulombic efficiency calculation.")
            continue
            
        # Ensure we only use cycles that have both charge and discharge data
        common_cycles = discharge_capacity.index.intersection(charge_capacity.index)
        if len(common_cycles) == 0:
            print(f"Warning: No cycles with both charge and discharge data for {cell_id}, skipping.")
            continue
            
        discharge_capacity = discharge_capacity.loc[common_cycles]
        charge_capacity = charge_capacity.loc[common_cycles]
        
        coulombic_efficiency = (discharge_capacity / charge_capacity) * 100
        
        # Plot the coulombic efficiency curve
        fig.add_trace(
            go.Scatter(
                x=coulombic_efficiency.index,
                y=coulombic_efficiency.values,
                name=cell_id,
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(
                    size=6,
                    line=dict(width=1, color='white')
                )
            )
        )
    
    # Create title
    title_text = f"<b>{title}</b><br>"
    # Add a subtitle with info about the cells
    subtitle = f"<span style='font-size:16px'>Comparing {len(cell_data)} cells: "
    subtitle += ", ".join([f"{cell_id} ({info['metadata'].get('cathode_type', 'Unknown')}/{info['metadata'].get('anode_type', 'Unknown')})" 
                          for cell_id, info in cell_data.items()])
    subtitle += "</span>"
    title_text += subtitle
    
    # Update layout with better axis visibility
    layout_update = {
        "title": {
            "text": title_text,
            "font": {"size": 22},
            "x": 0.0,  # Left align
            "xanchor": "left",
            "y": 0.95,  # Lower position to be closer to the plot area
            "yanchor": "top"
        },
        "xaxis": {
            "title": "Cycle Number",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "yaxis": {
            "title": "Coulombic Efficiency (%)",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "showlegend": True,
        "legend": {
            "title": None,  # Remove legend title
            "bordercolor": "rgba(0,0,0,0)",  # Make border transparent
            "borderwidth": 0,  # Remove border
            "bgcolor": "rgba(255, 255, 255, 0.8)",
            "orientation": "v",  # Vertical orientation
            "yanchor": "top",  # Anchor to top
            "y": 0.95,  # Position at top
            "xanchor": "right",  # Anchor to right
            "x": 0.99  # Position at right
        },
        "margin": {"t": 180, "r": 30, "l": 80, "b": 80}
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    else:
        # Default range for coulombic efficiency
        layout_update["yaxis"]["range"] = [90, 101]
    
    fig.update_layout(**layout_update)
    
    return fig

def compare_state_of_health(cell_data: dict, y_range: list = None, 
                          title: str = "State of Health Comparison") -> go.Figure:
    """Plot state of health vs cycle number for multiple cells.
    
    Args:
        cell_data: Dictionary with cell_ids as keys and dictionaries containing data and metadata as values
        y_range: Optional list specifying [min, max] for SOH axis
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Create color scale for cells
    colors = pc.qualitative.Plotly + pc.qualitative.Light24 + pc.qualitative.Dark24
    
    # Process each cell
    for i, (cell_id, cell_info) in enumerate(cell_data.items()):
        df = cell_info['data']
        metadata = cell_info['metadata']
        
        # Calculate state of health per cycle
        discharge_capacity = df[df['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max()
        
        # Skip cells with no discharge data
        if discharge_capacity.empty:
            print(f"Warning: No discharge data found for {cell_id}, skipping SOH calculation.")
            continue
            
        initial_capacity = discharge_capacity.iloc[0]
        soh = (discharge_capacity / initial_capacity) * 100
        
        # Plot the SOH curve
        fig.add_trace(
            go.Scatter(
                x=soh.index,
                y=soh.values,
                name=cell_id,
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(
                    size=6,
                    line=dict(width=1, color='white')
                )
            )
        )
    
    # Create title
    title_text = f"<b>{title}</b><br>"
    # Add a subtitle with info about the cells
    subtitle = f"<span style='font-size:16px'>Comparing {len(cell_data)} cells: "
    subtitle += ", ".join([f"{cell_id} ({info['metadata'].get('cathode_type', 'Unknown')}/{info['metadata'].get('anode_type', 'Unknown')})" 
                          for cell_id, info in cell_data.items()])
    subtitle += "</span>"
    title_text += subtitle
    
    # Update layout with better axis visibility
    layout_update = {
        "title": {
            "text": title_text,
            "font": {"size": 22},
            "x": 0.0,  # Left align
            "xanchor": "left",
            "y": 0.95,  # Lower position to be closer to the plot area
            "yanchor": "top"
        },
        "xaxis": {
            "title": "Cycle Number",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "yaxis": {
            "title": "State of Health (%)",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "showlegend": True,
        "legend": {
            "title": None,  # Remove legend title
            "bordercolor": "rgba(0,0,0,0)",  # Make border transparent
            "borderwidth": 0,  # Remove border
            "bgcolor": "rgba(255, 255, 255, 0.8)",
            "orientation": "v",  # Vertical orientation
            "yanchor": "top",  # Anchor to top
            "y": 0.95,  # Position at top
            "xanchor": "right",  # Anchor to right
            "x": 0.99  # Position at right
        },
        "margin": {"t": 180, "r": 30, "l": 80, "b": 80}
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    else:
        # Default range for SOH
        layout_update["yaxis"]["range"] = [50, 105]
    
    fig.update_layout(**layout_update)
    
    return fig 