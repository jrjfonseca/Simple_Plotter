import streamlit as st
import pandas as pd
import plotly.io as pio
from pathlib import Path
import sys
import tempfile
import os
from datetime import datetime
import logging
import base64
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
import io

# Configure page
st.set_page_config(
    page_title="Battery Lab - Electrochemical Test Data Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the example_functions/lib directory to the path for importing electrochemistry
current_dir = Path(__file__).resolve().parent
example_functions_lib = current_dir / "example_functions" / "lib"
sys.path.append(str(example_functions_lib))

# Import the electrochemistry module
import electrochemistry as ec

# Try to use SciencePlots if available
SCIENCEPLOTS_AVAILABLE = False
try:
    plt.style.use(['science'])
    SCIENCEPLOTS_AVAILABLE = True
    logger.info("SciencePlots styles loaded successfully")
except (OSError, ImportError) as e:
    logger.warning(f"SciencePlots styles not available: {e}")
    logger.warning("Publication plots will use default matplotlib style")

# Function to convert Plotly figure to publication-quality matplotlib figure
def generate_publication_plot(fig, title=None, xlabel=None, ylabel=None, x_range=None, y_range=None, is_cycle_plot=False):
    """
    Convert Plotly figure data to a publication-quality matplotlib figure
    Uses SciencePlots if available, otherwise falls back to default style
    
    Args:
        fig: Plotly figure object
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        x_range: Optional [min, max] for x-axis
        y_range: Optional [min, max] for y-axis
        is_cycle_plot: Whether this is a cycle-related plot (for integer x-ticks)
        
    Returns:
        tuple: (matplotlib figure, BytesIO buffer) containing the figure and PNG image
    """
    # Use context manager only if SciencePlots is available
    if SCIENCEPLOTS_AVAILABLE:
        plt_context = plt.style.context(['science'])
    else:
        # Create a no-op context manager as fallback
        from contextlib import nullcontext
        plt_context = nullcontext()
    
    with plt_context:
        # Create matplotlib figure
        mpl_fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        
        # Get the Plotly qualitative colors to match the original plots
        try:
            import plotly.colors as pc
            plotly_colors = pc.qualitative.Plotly
        except ImportError:
            # Default colors if plotly.colors is not available
            plotly_colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
        
        # Process trace names to extract cycle information
        cycles_info = {}
        legend_entries = []
        
        # Track x-axis data to determine if it's cycle numbers
        all_x_values = []
        
        # First pass: identify cycles and their positions
        cycle_positions = {}
        for i, trace in enumerate(fig.data):
            name = trace.name if hasattr(trace, 'name') else f"Trace {i}"
            
            # Collect x values to determine tick formatting
            if hasattr(trace, 'x') and trace.x is not None:
                all_x_values.extend(list(trace.x))
            
            # For charge-discharge plots, extract cycle number
            if "Cycle" in name:
                parts = name.split()
                if len(parts) >= 2 and parts[0] == "Cycle":
                    try:
                        cycle_num = int(parts[1])
                        if cycle_num not in cycle_positions:
                            # Use the position of this cycle in the sequence to determine color
                            cycle_positions[cycle_num] = len(cycle_positions)
                    except ValueError:
                        pass
        
        # Second pass: plot the data with consistent colors based on cycle position
        for i, trace in enumerate(fig.data):
            x = trace.x
            y = trace.y
            name = trace.name if hasattr(trace, 'name') else f"Trace {i}"
            
            # Determine plot mode (lines, markers, or both)
            plot_mode = 'lines'  # Default mode
            if hasattr(trace, 'mode') and trace.mode is not None:
                # Convert to string to safely use 'in' operator
                trace_mode_str = str(trace.mode)
                if 'markers' in trace_mode_str and 'lines' not in trace_mode_str:
                    plot_mode = 'markers'
                elif 'markers' in trace_mode_str and 'lines' in trace_mode_str:
                    plot_mode = 'lines+markers'
            
            marker_size = 8  # Default marker size
            marker_symbol = 'o'  # Default marker symbol
            if hasattr(trace, 'marker'):
                if hasattr(trace.marker, 'size'):
                    marker_size = trace.marker.size
                if hasattr(trace.marker, 'symbol'):
                    # Simple mapping for common marker symbols
                    symbol_map = {'circle': 'o', 'square': 's', 'diamond': 'D', 'cross': 'x', 'x': 'x'}
                    marker_symbol = symbol_map.get(trace.marker.symbol, 'o')
            
            cycle_num = None
            is_charge = True  # Default to solid line
            
            # Extract cycle information and determine if charge/discharge
            if "Cycle" in name:
                parts = name.split()
                if len(parts) >= 2 and parts[0] == "Cycle":
                    try:
                        cycle_num = int(parts[1])
                        is_charge = "Discharge" not in name  # Determine charge/discharge
                        
                        # Create cleaned name for legend
                        cycle_type = "Charge" if "Charge" in name else "Discharge" if "Discharge" in name else ""
                        clean_name = f"Cycle {cycle_num}" + (f" ({cycle_type})" if cycle_type else "")
                    except ValueError:
                        clean_name = name
                else:
                    clean_name = name
            else:
                clean_name = name
            
            # Determine color and line style
            if cycle_num is not None and cycle_num in cycle_positions:
                # Use cycle position to get color (matching Plotly's behavior)
                color_idx = cycle_positions[cycle_num] % len(plotly_colors)
                color = plotly_colors[color_idx]
                linestyle = '-' if is_charge else '--'  # Solid for charge, dashed for discharge
                
                # Only show in legend once per cycle
                if f"Cycle {cycle_num}" in legend_entries:
                    label = '_nolegend_'
                else:
                    label = f"Cycle {cycle_num}"
                    legend_entries.append(f"Cycle {cycle_num}")
            else:
                # For non-cycle traces, use sequential coloring
                color_idx = i % len(plotly_colors)
                color = plotly_colors[color_idx]
                linestyle = '-'  # Default linestyle
                
                if clean_name in legend_entries:
                    label = '_nolegend_'
                else:
                    label = clean_name
                    legend_entries.append(clean_name)
            
            # Plot according to mode
            if plot_mode == 'markers':
                ax.scatter(x, y, color=color, s=marker_size**2, marker=marker_symbol, label=label)
            elif plot_mode == 'lines+markers':
                ax.plot(x, y, color=color, linestyle=linestyle, label=label, marker=marker_symbol, markersize=marker_size)
            else:  # 'lines' is the default
                ax.plot(x, y, color=color, linestyle=linestyle, label=label)
        
        # Set integer ticks for cycle plots
        if is_cycle_plot or ("cycle" in xlabel.lower() if xlabel else False):
            # Determine appropriate tick marks
            if all_x_values:
                try:
                    # Get unique x values sorted, handle potential type conversion issues
                    unique_x = []
                    for x in all_x_values:
                        try:
                            if x is not None:
                                unique_x.append(int(float(x)))
                        except (ValueError, TypeError):
                            pass  # Skip values that can't be converted to int
                    
                    unique_x = sorted(list(set(unique_x)))
                    
                    # If reasonable number of cycles, show all of them
                    if unique_x and len(unique_x) <= 15:
                        ax.set_xticks(unique_x)
                        ax.set_xticklabels([str(x) for x in unique_x])
                except Exception as e:
                    logger.warning(f"Error setting integer ticks: {e}")
                    # Fallback to matplotlib's integer locator
                    import matplotlib.ticker as ticker
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        # Set labels and title
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
            
        # Set axis ranges if provided
        if x_range and len(x_range) == 2:
            ax.set_xlim(x_range)
        if y_range and len(y_range) == 2:
            ax.set_ylim(y_range)
        
        # Add legend if we have named traces
        if legend_entries:
            ax.legend(frameon=True)
            
        # Adjust layout
        plt.tight_layout()
        
        # Save to BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Note: we don't close the figure here so it can be displayed
        
        return mpl_fig, buf

# Set default Plotly template for better appearance
pio.templates.default = "plotly_white"

# Initialize session state for storing processed data
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
    st.session_state.has_data = False
    st.session_state.form_submitted = False
    st.session_state.ready_for_new_cell = False
    
# Initialize multi-cell session state
if 'cells' not in st.session_state:
    st.session_state.cells = {}  # Dictionary to store multiple cells with their metadata
    st.session_state.selected_cells = []  # List to track selected cells for comparison
    st.session_state.has_multiple_cells = False
    st.session_state.batch_files = []  # Store multiple uploaded files
    st.session_state.batch_metadata = {}  # Store metadata for multiple files
    st.session_state.batch_mode = False  # Toggle between single and batch mode

# Function to enhance plot appearance
def enhance_plot(fig):
    """Apply common styling improvements to plotly figures"""
    # Improve font sizes and family
    fig.update_layout(
        font=dict(
            family="Arial, sans-serif",
            size=14,
        ),
        title=dict(
            font=dict(size=18, family="Arial, sans-serif"),
        ),
        legend=dict(
            font=dict(size=12),
            bordercolor="LightGrey",
            borderwidth=1,
        ),
        plot_bgcolor='white',
        height=700,  # Fixed height for better appearance
        margin=dict(l=10, r=10, t=180, b=80),
    )
    
    # Improve axis appearance
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#E5ECF6',
        showline=True,
        linecolor='black',
        mirror=True,
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#E5ECF6',
        showline=True,
        linecolor='black',
        mirror=True,
    )
    
    return fig

# Display banner with a reliable placeholder image
st.image("https://via.placeholder.com/1200x400/0e1117/ffffff?text=Battery+Lab+-+Electrochemical+Test+Data+Analyzer", use_container_width=True)

# Main title
st.title("Battery Lab - Electrochemical Test Data Analyzer")
st.markdown("Upload your Excel file containing coin cell test data to visualize performance")

# Main content - Create two-column layout
col_main, col_sidebar = st.columns([5, 1])

# Hide default sidebar on the right
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        display: none;
    }
    .main-column {
        border-right: 1px solid #ddd;
        padding-right: 2rem;
    }
    .upload-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
    }
    .metadata-card {
        background-color: #f1f8ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with col_main:
    st.title("Battery Lab - Electrochemical Analyzer")
    
    # Create tabs for single cell vs. batch processing
    mode_tab1, mode_tab2 = st.tabs(["📊 Single Cell Analysis", "📚 Batch Processing"])
    
    # Single Cell Mode
    with mode_tab1:
        st.markdown('<div class="main-column">', unsafe_allow_html=True)
        
        # Display file uploader and metadata form
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"], key="single_upload")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            st.markdown('<div class="metadata-card">', unsafe_allow_html=True)
            st.subheader("Cell Metadata")
            
            with st.form(key="metadata_form"):
                # Basic Info
                col1, col2 = st.columns(2)
                with col1:
                    cell_id = st.text_input("Cell ID")
                with col2:
                    date = st.date_input("Test Date", value=datetime.now())
                
                # Cathode Parameters
                st.subheader("Cathode Parameters")
                col1, col2 = st.columns(2)
                with col1:
                    cathode_type = st.text_input("Cathode Type")
                    cathode_mass = st.number_input("Cathode Mass (mg)", min_value=0.001, format="%.3f", value=20.0)
                    active_material_percentage = st.number_input("Active Material (%)", min_value=0.1, max_value=100.0, value=75.0)
                with col2:
                    cathode_composition = st.text_input("Cathode Composition", placeholder="e.g. 75:20:5 (AM:C:Binder)")
                    collector_mass = st.number_input("Collector Mass (mg)", min_value=0.0, format="%.3f", value=0.0)
                    cathode_mixing_method = st.text_input("Cathode Mixing Method", placeholder="e.g. Vortex, Ball Milling")
                
                # Anode Parameters
                st.subheader("Anode Parameters")
                col1, col2 = st.columns(2)
                with col1:
                    anode_type = st.text_input("Anode Type")
                    anode_mass = st.text_input("Anode Mass (mg)", placeholder="e.g. 15.1 mg")
                with col2:
                    anode_composition = st.text_input("Anode Composition", placeholder="e.g. 90:5:5 (AM:C:Binder)")
                    anode_mixing_method = st.text_input("Anode Mixing Method", placeholder="e.g. Dry mixing")
                
                # Cell Parameters
                st.subheader("Cell Parameters")
                col1, col2 = st.columns(2)
                with col1:
                    electrolyte = st.text_input("Electrolyte", placeholder="e.g. 1M LiPF6 in EC:EMC (1:1)")
                    channel = st.text_input("Channel", placeholder="e.g. 8 VLC")
                    c_rate = st.text_input("C-Rate", placeholder="e.g. 0.05C")
                with col2:
                    electrolyte_quantity = st.text_input("Electrolyte Quantity", placeholder="e.g. 200 µL")
                    voltage_range = st.text_input("Voltage Range", placeholder="e.g. 1.5-3.5 V")
                    pressure = st.text_input("Pressure", placeholder="e.g. 2.5 ton 40 min")
                    
                # Submit button - must be inside the form
                submitted = st.form_submit_button(label="Process Data")
                if submitted:
                    st.session_state.form_submitted = True
                    # Validate required fields
                    if not cell_id or not cathode_type or not anode_type:
                        st.error("Please fill in all required fields: Cell ID, Cathode Type, and Anode Type")
                        st.session_state.form_submitted = False
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Handle the form submission
            if st.session_state.form_submitted:
                # Reset form_submitted for next use
                st.session_state.form_submitted = False
                
                with st.spinner("Processing data..."):
                    try:
                        # Save uploaded file to temp location
                        temp_file = Path(tempfile.gettempdir()) / uploaded_file.name
                        with open(temp_file, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        # Create cell metadata dictionary
                        cell_metadata = {
                            'cell_id': cell_id,
                            'cathode_type': cathode_type,
                            'anode_type': anode_type,
                            'cathode_mass': f"{cathode_mass} mg",
                            'collector_mass': f"{collector_mass} mg",
                            'active_material_percentage': active_material_percentage,
                            'date': date.strftime("%Y-%m-%d")
                        }
                        
                        # Add optional metadata
                        if cathode_composition:
                            cell_metadata['cathode_composition'] = cathode_composition
                        if cathode_mixing_method:
                            cell_metadata['cathode_mixing_method'] = cathode_mixing_method
                        if anode_composition:
                            cell_metadata['anode_composition'] = anode_composition
                        if anode_mass:
                            cell_metadata['anode_mass'] = anode_mass
                        if anode_mixing_method:
                            cell_metadata['anode_mixing_method'] = anode_mixing_method
                        if electrolyte:
                            cell_metadata['electrolyte'] = electrolyte
                        if electrolyte_quantity:
                            cell_metadata['electrolyte_quantity'] = electrolyte_quantity
                        if channel:
                            cell_metadata['channel'] = channel
                        if voltage_range:
                            cell_metadata['voltage_range'] = voltage_range
                        if c_rate:
                            cell_metadata['c_rate'] = c_rate
                        if pressure:
                            cell_metadata['pressure'] = pressure
                        
                        # Process data
                        df_normalized = ec.load_data(
                            temp_file,
                            total_mass=cathode_mass,
                            am_percentage=active_material_percentage,
                            collector_mass=collector_mass
                        )
                        
                        # Store in session state
                        st.session_state.processed_data = {
                            'df': df_normalized,
                            'metadata': cell_metadata,
                            'filename': uploaded_file.name
                        }
                        st.session_state.has_data = True
                        
                        # Clean up
                        if temp_file.exists():
                            os.unlink(temp_file)
                        
                        # Add to multi-cell collection automatically
                        cell_key = f"{cell_id}_{datetime.now().strftime('%H%M%S')}"
                        st.session_state.cells[cell_key] = {
                            'df': df_normalized.copy(),
                            'metadata': cell_metadata.copy(),
                            'filename': uploaded_file.name,
                            'display_name': cell_id
                        }
                        st.session_state.selected_cells.append(cell_key)
                        st.session_state.has_multiple_cells = len(st.session_state.cells) > 1
                        
                        st.success("Data processed successfully and added to comparison!")
                    
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Batch Processing Mode
    with mode_tab2:
        st.markdown("""
        ## Batch Cell Data Processing
        Upload multiple Excel files at once and set metadata for each cell.
        """)
        
        # Multi-file uploader
        uploaded_files = st.file_uploader("Upload Multiple Excel Files", type=["xlsx", "xls"], accept_multiple_files=True, key="batch_upload")
        
        if uploaded_files:
            # Store the batch of files in session state if it's a new selection
            current_filenames = [f.name for f in uploaded_files]
            previous_filenames = [f.name for f in st.session_state.batch_files]
            
            if set(current_filenames) != set(previous_filenames):
                st.session_state.batch_files = uploaded_files
                st.session_state.batch_metadata = {}
                
                # Initialize metadata for each file
                for idx, file in enumerate(uploaded_files):
                    cell_id = f"Cell_{idx+1}"
                    st.session_state.batch_metadata[file.name] = {
                        'cell_id': cell_id,
                        'cathode_type': '',
                        'anode_type': 'Li',
                        'cathode_mass': 20.0,
                        'collector_mass': 0.0,
                        'active_material_percentage': 75.0,
                        'date': datetime.now().strftime("%Y-%m-%d"),
                        'display_name': cell_id
                    }
            
            # Display metadata forms for each file
            with st.form(key="batch_metadata_form"):
                st.subheader("Batch Cell Metadata")
                
                # Common parameters (will be applied to all cells)
                st.markdown("### Common Parameters (Applied to All Cells)")
                col1, col2 = st.columns(2)
                with col1:
                    common_cathode_type = st.text_input("Common Cathode Type", placeholder="Leave empty to set individually")
                    common_anode_type = st.text_input("Common Anode Type", value="Li")
                with col2:
                    common_active_material = st.number_input("Common Active Material %", min_value=0.1, max_value=100.0, value=75.0)
                    common_collector_mass = st.number_input("Common Collector Mass (mg)", min_value=0.0, value=0.0, format="%.3f")
                
                # Individual cell metadata
                st.markdown("### Individual Cell Parameters")
                
                # Use an expander for each file to save space
                for idx, file in enumerate(uploaded_files):
                    with st.expander(f"Cell {idx+1}: {file.name}", expanded=idx == 0):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            cell_id = st.text_input(f"Cell ID", value=st.session_state.batch_metadata[file.name]['cell_id'], key=f"id_{idx}")
                            cathode_type = st.text_input(f"Cathode Type", value=st.session_state.batch_metadata[file.name]['cathode_type'], key=f"cathode_{idx}")
                            cathode_mass = st.number_input(f"Cathode Mass (mg)", min_value=0.001, value=st.session_state.batch_metadata[file.name]['cathode_mass'], format="%.3f", key=f"mass_{idx}")
                        
                        with col2:
                            anode_type = st.text_input(f"Anode Type", value=st.session_state.batch_metadata[file.name]['anode_type'], key=f"anode_{idx}")
                            c_rate = st.text_input(f"C-Rate", placeholder="e.g. 0.05C", key=f"crate_{idx}")
                            
                        # Store in session state to persist between reruns
                        st.session_state.batch_metadata[file.name]['cell_id'] = cell_id
                        st.session_state.batch_metadata[file.name]['cathode_type'] = cathode_type if cathode_type else common_cathode_type
                        st.session_state.batch_metadata[file.name]['anode_type'] = anode_type if anode_type else common_anode_type
                        st.session_state.batch_metadata[file.name]['cathode_mass'] = cathode_mass
                        st.session_state.batch_metadata[file.name]['collector_mass'] = common_collector_mass
                        st.session_state.batch_metadata[file.name]['active_material_percentage'] = common_active_material
                        st.session_state.batch_metadata[file.name]['c_rate'] = c_rate
                        st.session_state.batch_metadata[file.name]['display_name'] = cell_id
                
                # Submit button for batch processing
                batch_submitted = st.form_submit_button(label="Process All Cells")
                
            # Process all files if submitted
            if batch_submitted:
                with st.spinner("Processing batch of cells..."):
                    processed_count = 0
                    error_count = 0
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, file in enumerate(uploaded_files):
                        try:
                            status_text.text(f"Processing {file.name}...")
                            
                            # Get metadata for this file
                            metadata = st.session_state.batch_metadata[file.name]
                            
                            # Validate required fields
                            if not metadata['cell_id'] or not metadata['cathode_type'] or not metadata['anode_type']:
                                st.warning(f"Skipping {file.name}: Missing required metadata")
                                error_count += 1
                                continue
                            
                            # Save uploaded file to temp location
                            temp_file = Path(tempfile.gettempdir()) / file.name
                            with open(temp_file, "wb") as f:
                                f.write(file.getvalue())
                            
                            # Create complete metadata dictionary
                            cell_metadata = {
                                'cell_id': metadata['cell_id'],
                                'cathode_type': metadata['cathode_type'],
                                'anode_type': metadata['anode_type'],
                                'cathode_mass': f"{metadata['cathode_mass']} mg",
                                'collector_mass': f"{metadata['collector_mass']} mg",
                                'active_material_percentage': metadata['active_material_percentage'],
                                'date': metadata['date']
                            }
                            
                            # Add optional metadata
                            if 'c_rate' in metadata and metadata['c_rate']:
                                cell_metadata['c_rate'] = metadata['c_rate']
                            
                            # Process data
                            df_normalized = ec.load_data(
                                temp_file,
                                total_mass=float(metadata['cathode_mass']),
                                am_percentage=float(metadata['active_material_percentage']),
                                collector_mass=float(metadata['collector_mass'])
                            )
                            
                            # Add to cell collection
                            cell_key = f"{metadata['cell_id']}_{datetime.now().strftime('%H%M%S')}"
                            st.session_state.cells[cell_key] = {
                                'df': df_normalized,
                                'metadata': cell_metadata,
                                'filename': file.name,
                                'display_name': metadata['cell_id']
                            }
                            st.session_state.selected_cells.append(cell_key)
                            
                            # Clean up
                            if temp_file.exists():
                                os.unlink(temp_file)
                                
                            processed_count += 1
                            
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                            error_count += 1
                        
                        # Update progress
                        progress = (idx + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                    
                    # Update multi-cell state
                    st.session_state.has_multiple_cells = len(st.session_state.cells) > 1
                    
                    # Show summary
                    status_text.text(f"Batch processing complete: {processed_count} cells processed, {error_count} errors")
                    
                    if processed_count > 0:
                        st.success(f"Successfully processed {processed_count} cells. Ready for comparison!")
                        # Automatically go to comparison view
                        st.session_state.batch_mode = True

# Cell Manager Panel
with col_sidebar:
    st.header("Cell Manager")
    
    # Display cell count
    if len(st.session_state.cells) > 0:
        st.success(f"📊 {len(st.session_state.cells)} cells available")
        st.markdown("### Available Cells")
        
        # Display available cells with improved formatting
        for cell_key, cell_data in st.session_state.cells.items():
            cell_container = st.container()
            cell_container.markdown(f"""
            <div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;'>
                <strong>{cell_data['display_name']}</strong><br>
                <small>{cell_data['metadata']['cathode_type']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Checkbox to select/deselect for comparison
                is_selected = st.checkbox(
                    "Select", 
                    value=cell_key in st.session_state.selected_cells,
                    key=f"select_{cell_key}"
                )
                
                if is_selected and cell_key not in st.session_state.selected_cells:
                    st.session_state.selected_cells.append(cell_key)
                elif not is_selected and cell_key in st.session_state.selected_cells:
                    st.session_state.selected_cells.remove(cell_key)
            
            with col2:
                # Button to remove cell
                if st.button("🗑️", key=f"remove_{cell_key}"):
                    # Remove from cells dictionary
                    del st.session_state.cells[cell_key]
                    # Remove from selected cells if present
                    if cell_key in st.session_state.selected_cells:
                        st.session_state.selected_cells.remove(cell_key)
                    st.rerun()
        
        # Button to view multi-cell comparison
        if len(st.session_state.selected_cells) >= 2:
            st.success("✅ Ready for comparison")
        else:
            st.warning("Select at least 2 cells")
    else:
        st.info("No cells available yet. Upload and process cell data to begin.")

# Main content area - show visualization options if data is processed
if st.session_state.has_data:
    st.header("Data Visualization")
    
    # Metadata display
    with st.expander("Cell Metadata", expanded=False):
        st.json(st.session_state.processed_data['metadata'])
    
    # Tabs for different plot types
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Charge-Discharge", "Capacity vs Cycle", "State of Health", 
        "Coulombic Efficiency", "Voltage vs Time", "Differential Capacity", 
        "Combined Performance"
    ])
    
    # Tab 1: Charge-Discharge
    with tab1:
        st.subheader("Charge-Discharge Curves")
        
        # Input for cycle selection
        cd_cycles = st.text_input("Cycles to display (comma-separated)", value="1,2,3", key="cd_cycles")
        
        # Add option for publication quality
        pub_quality = st.checkbox("Publication Quality Plot", key="pub_cd", value=False)
        
        # Add publication quality controls
        if pub_quality:
            with st.expander("Publication Plot Settings", expanded=False):
                pub_title = st.text_input("Custom Plot Title", value="", key="pub_title_cd")
                col1, col2 = st.columns(2)
                with col1:
                    x_min = st.number_input("X-axis Min", value=None, key="x_min_cd")
                    y_min = st.number_input("Y-axis Min", value=None, key="y_min_cd")
                with col2:
                    x_max = st.number_input("X-axis Max", value=None, key="x_max_cd")
                    y_max = st.number_input("Y-axis Max", value=None, key="y_max_cd")
        
        if st.button("Generate Plot", key="btn_cd"):
            with st.spinner("Generating plot..."):
                # Parse cycles
                cycles_to_plot = [int(c.strip()) for c in cd_cycles.split(",")]
                
                # Generate plot
                fig = ec.plot_charge_discharge(
                    st.session_state.processed_data['df'], 
                    cycles_to_plot=cycles_to_plot,
                    cell_metadata=st.session_state.processed_data['metadata']
                )
                
                # Enhance plot
                fig = enhance_plot(fig)
                
                # Handle publication quality option
                if pub_quality:
                    # Build custom title or use default
                    custom_title = pub_title if pub_title else f"Charge-Discharge Curves - {st.session_state.processed_data['metadata']['cell_id']}"
                    
                    # Build axis ranges if provided
                    x_range = None
                    if x_min is not None and x_max is not None:
                        x_range = [x_min, x_max]
                    
                    y_range = None
                    if y_min is not None and y_max is not None:
                        y_range = [y_min, y_max]
                    
                    # Generate publication quality plot
                    mpl_fig, buf = generate_publication_plot(
                        fig, 
                        title=custom_title,
                        xlabel="Capacity (mAh/g)",
                        ylabel="Voltage (V)",
                        x_range=x_range,
                        y_range=y_range
                    )
                    
                    # Display the matplotlib figure directly in Streamlit
                    st.pyplot(mpl_fig)
                    plt.close(mpl_fig)  # Close the figure to free memory
                    
                    # Add download button for the publication quality figure
                    st.download_button(
                        label="Download Publication Quality PNG",
                        data=buf,
                        file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_charge_discharge_pub.png",
                        mime="image/png",
                    )
                else:
                    # Display the regular plotly chart if not using publication quality
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download options
                csv = st.session_state.processed_data['df'][st.session_state.processed_data['df']['Step_Type'] == 'Discharge'][['Cycle_Index', 'Voltage', 'Capacity']]
                
                st.download_button(
                    label="Download Data as CSV",
                    data=csv.to_csv(index=False).encode('utf-8'),
                    file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_charge_discharge_data.csv",
                    mime='text/csv',
                )
    
    # Tab 2: Capacity vs Cycle
    with tab2:
        st.subheader("Capacity vs Cycle")
        
        # Add option for publication quality
        pub_quality = st.checkbox("Publication Quality Plot", key="pub_capacity", value=False)
        
        # Add publication quality controls
        if pub_quality:
            with st.expander("Publication Plot Settings", expanded=False):
                pub_title = st.text_input("Custom Plot Title", value="", key="pub_title_capacity")
                col1, col2 = st.columns(2)
                with col1:
                    x_min = st.number_input("X-axis Min", value=None, key="x_min_capacity")
                    y_min = st.number_input("Y-axis Min", value=None, key="y_min_capacity")
                with col2:
                    x_max = st.number_input("X-axis Max", value=None, key="x_max_capacity")
                    y_max = st.number_input("Y-axis Max", value=None, key="y_max_capacity")
        
        if st.button("Generate Plot", key="btn_capacity"):
            with st.spinner("Generating plot..."):
                # Generate plot
                fig = ec.plot_capacity_vs_cycle(
                    st.session_state.processed_data['df'],
                    cell_metadata=st.session_state.processed_data['metadata']
                )
                
                # Enhance plot
                fig = enhance_plot(fig)
                
                # Handle publication quality option
                if pub_quality:
                    # Build custom title or use default
                    custom_title = pub_title if pub_title else f"Capacity vs Cycle - {st.session_state.processed_data['metadata']['cell_id']}"
                    
                    # Build axis ranges if provided
                    x_range = None
                    if x_min is not None and x_max is not None:
                        x_range = [x_min, x_max]
                    
                    y_range = None
                    if y_min is not None and y_max is not None:
                        y_range = [y_min, y_max]
                    
                    # Generate publication quality plot
                    mpl_fig, buf = generate_publication_plot(
                        fig, 
                        title=custom_title,
                        xlabel="Cycle Number",
                        ylabel="Capacity (mAh/g)",
                        x_range=x_range,
                        y_range=y_range,
                        is_cycle_plot=True
                    )
                    
                    # Display the matplotlib figure directly in Streamlit
                    st.pyplot(mpl_fig)
                    plt.close(mpl_fig)  # Close the figure to free memory
                    
                    # Add download button for the publication quality figure
                    st.download_button(
                        label="Download Publication Quality PNG",
                        data=buf,
                        file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_capacity_pub.png",
                        mime="image/png",
                    )
                else:
                    # Display the regular plotly chart if not using publication quality
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download options
                capacity_data = st.session_state.processed_data['df'][st.session_state.processed_data['df']['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max().reset_index()
                
                st.download_button(
                    label="Download Data as CSV",
                    data=capacity_data.to_csv(index=False).encode('utf-8'),
                    file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_capacity_data.csv",
                    mime='text/csv',
                )
    
    # Tab 3: State of Health
    with tab3:
        st.subheader("State of Health")
        
        # Add option for publication quality
        pub_quality = st.checkbox("Publication Quality Plot", key="pub_soh", value=False)
        
        # Add publication quality controls
        if pub_quality:
            with st.expander("Publication Plot Settings", expanded=False):
                pub_title = st.text_input("Custom Plot Title", value="", key="pub_title_soh")
                col1, col2 = st.columns(2)
                with col1:
                    x_min = st.number_input("X-axis Min", value=None, key="x_min_soh")
                    y_min = st.number_input("Y-axis Min", value=None, key="y_min_soh")
                with col2:
                    x_max = st.number_input("X-axis Max", value=None, key="x_max_soh")
                    y_max = st.number_input("Y-axis Max", value=None, key="y_max_soh")
        
        if st.button("Generate Plot", key="btn_soh"):
            with st.spinner("Generating plot..."):
                # Generate plot
                fig = ec.plot_state_of_health(
                    st.session_state.processed_data['df'],
                    cell_metadata=st.session_state.processed_data['metadata']
                )
                
                # Enhance plot
                fig = enhance_plot(fig)
                
                # Handle publication quality option
                if pub_quality:
                    # Build custom title or use default
                    custom_title = pub_title if pub_title else f"State of Health - {st.session_state.processed_data['metadata']['cell_id']}"
                    
                    # Build axis ranges if provided
                    x_range = None
                    if x_min is not None and x_max is not None:
                        x_range = [x_min, x_max]
                    
                    y_range = None
                    if y_min is not None and y_max is not None:
                        y_range = [y_min, y_max]
                    
                    # Generate publication quality plot
                    mpl_fig, buf = generate_publication_plot(
                        fig, 
                        title=custom_title,
                        xlabel="Cycle Number",
                        ylabel="State of Health (%)",
                        x_range=x_range,
                        y_range=y_range,
                        is_cycle_plot=True
                    )
                    
                    # Display the matplotlib figure directly in Streamlit
                    st.pyplot(mpl_fig)
                    plt.close(mpl_fig)  # Close the figure to free memory
                    
                    # Add download button for the publication quality figure
                    st.download_button(
                        label="Download Publication Quality PNG",
                        data=buf,
                        file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_soh_pub.png",
                        mime="image/png",
                    )
                else:
                    # Display the regular plotly chart if not using publication quality
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download options
                discharge_capacity = st.session_state.processed_data['df'][st.session_state.processed_data['df']['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max()
                initial_capacity = discharge_capacity.iloc[0]
                soh = (discharge_capacity / initial_capacity) * 100
                soh_df = pd.DataFrame({'Cycle_Index': soh.index, 'State_of_Health': soh.values})
                
                st.download_button(
                    label="Download Data as CSV",
                    data=soh_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_soh_data.csv",
                    mime='text/csv',
                )
    
    # Tab 4: Coulombic Efficiency
    with tab4:
        st.subheader("Coulombic Efficiency")
        
        # Add option for publication quality
        pub_quality = st.checkbox("Publication Quality Plot", key="pub_ce", value=False)
        
        # Add publication quality controls
        if pub_quality:
            with st.expander("Publication Plot Settings", expanded=False):
                pub_title = st.text_input("Custom Plot Title", value="", key="pub_title_ce")
                col1, col2 = st.columns(2)
                with col1:
                    x_min = st.number_input("X-axis Min", value=None, key="x_min_ce")
                    y_min = st.number_input("Y-axis Min", value=None, key="y_min_ce")
                with col2:
                    x_max = st.number_input("X-axis Max", value=None, key="x_max_ce")
                    y_max = st.number_input("Y-axis Max", value=None, key="y_max_ce")
        
        if st.button("Generate Plot", key="btn_ce"):
            with st.spinner("Generating plot..."):
                # Generate plot
                fig = ec.plot_coulombic_efficiency(
                    st.session_state.processed_data['df'],
                    cell_metadata=st.session_state.processed_data['metadata']
                )
                
                # Enhance plot
                fig = enhance_plot(fig)
                
                # Handle publication quality option
                if pub_quality:
                    # Build custom title or use default
                    custom_title = pub_title if pub_title else f"Coulombic Efficiency - {st.session_state.processed_data['metadata']['cell_id']}"
                    
                    # Build axis ranges if provided
                    x_range = None
                    if x_min is not None and x_max is not None:
                        x_range = [x_min, x_max]
                    
                    y_range = None
                    if y_min is not None and y_max is not None:
                        y_range = [y_min, y_max]
                    
                    # Generate publication quality plot
                    mpl_fig, buf = generate_publication_plot(
                        fig, 
                        title=custom_title,
                        xlabel="Cycle Number",
                        ylabel="Coulombic Efficiency (%)",
                        x_range=x_range,
                        y_range=y_range,
                        is_cycle_plot=True
                    )
                    
                    # Display the matplotlib figure directly in Streamlit
                    st.pyplot(mpl_fig)
                    plt.close(mpl_fig)  # Close the figure to free memory
                    
                    # Add download button for the publication quality figure
                    st.download_button(
                        label="Download Publication Quality PNG",
                        data=buf,
                        file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_ce_pub.png",
                        mime="image/png",
                    )
                else:
                    # Display the regular plotly chart if not using publication quality
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download options
                discharge_capacity = st.session_state.processed_data['df'][st.session_state.processed_data['df']['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max()
                charge_capacity = st.session_state.processed_data['df'][st.session_state.processed_data['df']['Step_Type'] == 'Charge'].groupby('Cycle_Index')['Capacity'].max()
                ce = (discharge_capacity / charge_capacity) * 100
                ce_df = pd.DataFrame({'Cycle_Index': ce.index, 'Coulombic_Efficiency': ce.values})
                
                st.download_button(
                    label="Download Data as CSV",
                    data=ce_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_coulombic_efficiency_data.csv",
                    mime='text/csv',
                )
    
    # Tab 5: Voltage vs Time
    with tab5:
        st.subheader("Voltage vs Time")
        
        # Add option for publication quality
        pub_quality = st.checkbox("Publication Quality Plot", key="pub_vt", value=False)
        
        # Add publication quality controls
        if pub_quality:
            with st.expander("Publication Plot Settings", expanded=False):
                pub_title = st.text_input("Custom Plot Title", value="", key="pub_title_vt")
                col1, col2 = st.columns(2)
                with col1:
                    x_min = st.number_input("X-axis Min", value=None, key="x_min_vt")
                    y_min = st.number_input("Y-axis Min", value=None, key="y_min_vt")
                with col2:
                    x_max = st.number_input("X-axis Max", value=None, key="x_max_vt")
                    y_max = st.number_input("Y-axis Max", value=None, key="y_max_vt")
        
        if st.button("Generate Plot", key="btn_vt"):
            with st.spinner("Generating plot..."):
                # Generate plot
                fig = ec.plot_voltage_time(
                    st.session_state.processed_data['df'],
                    cell_metadata=st.session_state.processed_data['metadata']
                )
                
                # Enhance plot
                fig = enhance_plot(fig)
                
                # Handle publication quality option
                if pub_quality:
                    # Build custom title or use default
                    custom_title = pub_title if pub_title else f"Voltage vs Time - {st.session_state.processed_data['metadata']['cell_id']}"
                    
                    # Build axis ranges if provided
                    x_range = None
                    if x_min is not None and x_max is not None:
                        x_range = [x_min, x_max]
                    
                    y_range = None
                    if y_min is not None and y_max is not None:
                        y_range = [y_min, y_max]
                    
                    # Generate publication quality plot
                    mpl_fig, buf = generate_publication_plot(
                        fig, 
                        title=custom_title,
                        xlabel="Time (h)",
                        ylabel="Voltage (V)",
                        x_range=x_range,
                        y_range=y_range
                    )
                    
                    # Display the matplotlib figure directly in Streamlit
                    st.pyplot(mpl_fig)
                    plt.close(mpl_fig)  # Close the figure to free memory
                    
                    # Add download button for the publication quality figure
                    st.download_button(
                        label="Download Publication Quality PNG",
                        data=buf,
                        file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_voltage_time_pub.png",
                        mime="image/png",
                    )
                else:
                    # Display the regular plotly chart if not using publication quality
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download options
                voltage_time = st.session_state.processed_data['df'][['Test_Time', 'Voltage']]
                
                st.download_button(
                    label="Download Data as CSV",
                    data=voltage_time.to_csv(index=False).encode('utf-8'),
                    file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_voltage_time_data.csv",
                    mime='text/csv',
                )
    
    # Tab 6: Differential Capacity
    with tab6:
        st.subheader("Differential Capacity (dQ/dV)")
        
        # Input for cycle selection
        dqdv_cycles = st.text_input("Cycles to display (comma-separated)", value="1,2", key="dqdv_cycles")
        
        # Add option for publication quality
        pub_quality = st.checkbox("Publication Quality Plot", key="pub_dqdv", value=False)
        
        # Add publication quality controls
        if pub_quality:
            with st.expander("Publication Plot Settings", expanded=False):
                pub_title = st.text_input("Custom Plot Title", value="", key="pub_title_dqdv")
                col1, col2 = st.columns(2)
                with col1:
                    x_min = st.number_input("X-axis Min", value=None, key="x_min_dqdv")
                    y_min = st.number_input("Y-axis Min", value=None, key="y_min_dqdv")
                with col2:
                    x_max = st.number_input("X-axis Max", value=None, key="x_max_dqdv")
                    y_max = st.number_input("Y-axis Max", value=None, key="y_max_dqdv")
        
        if st.button("Generate Plot", key="btn_dqdv"):
            with st.spinner("Generating plot..."):
                # Parse cycles
                cycles_to_plot = [int(c.strip()) for c in dqdv_cycles.split(",")]
                
                # Generate plot
                fig = ec.plot_differential_capacity(
                    st.session_state.processed_data['df'],
                    cycles=cycles_to_plot,
                    cell_metadata=st.session_state.processed_data['metadata']
                )
                
                # Enhance plot
                fig = enhance_plot(fig)
                
                # Handle publication quality option
                if pub_quality:
                    # Build custom title or use default
                    custom_title = pub_title if pub_title else f"Differential Capacity - {st.session_state.processed_data['metadata']['cell_id']}"
                    
                    # Build axis ranges if provided
                    x_range = None
                    if x_min is not None and x_max is not None:
                        x_range = [x_min, x_max]
                    
                    y_range = None
                    if y_min is not None and y_max is not None:
                        y_range = [y_min, y_max]
                    
                    # Generate publication quality plot
                    mpl_fig, buf = generate_publication_plot(
                        fig, 
                        title=custom_title,
                        xlabel="Voltage (V)",
                        ylabel="dQ/dV (mAh/g·V)",
                        x_range=x_range,
                        y_range=y_range
                    )
                    
                    # Display the matplotlib figure directly in Streamlit
                    st.pyplot(mpl_fig)
                    plt.close(mpl_fig)  # Close the figure to free memory
                    
                    # Add download button for the publication quality figure
                    st.download_button(
                        label="Download Publication Quality PNG",
                        data=buf,
                        file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_dqdv_pub.png",
                        mime="image/png",
                    )
                else:
                    # Display the regular plotly chart if not using publication quality
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 7: Combined Performance
    with tab7:
        st.subheader("Combined Performance")
        
        # Input for cycle selection
        cp_cycles = st.text_input("Cycles to display (comma-separated)", value="1,2,3,5", key="cp_cycles")
        
        # Add option for publication quality
        pub_quality = st.checkbox("Publication Quality Plot", key="pub_cp", value=False)
        
        # Add publication quality controls
        if pub_quality:
            with st.expander("Publication Plot Settings", expanded=False):
                pub_title = st.text_input("Custom Plot Title", value="", key="pub_title_cp")
                col1, col2 = st.columns(2)
                with col1:
                    x_min = st.number_input("X-axis Min", value=None, key="x_min_cp")
                    y_min = st.number_input("Y-axis Min", value=None, key="y_min_cp")
                with col2:
                    x_max = st.number_input("X-axis Max", value=None, key="x_max_cp")
                    y_max = st.number_input("Y-axis Max", value=None, key="y_max_cp")
        
        if st.button("Generate Plot", key="btn_cp"):
            with st.spinner("Generating plot..."):
                # Parse cycles
                cycles_to_plot = [int(c.strip()) for c in cp_cycles.split(",")]
                
                # Generate plot
                fig = ec.plot_combined_performance(
                    st.session_state.processed_data['df'],
                    cycles_to_plot=cycles_to_plot,
                    cell_metadata=st.session_state.processed_data['metadata']
                )
                
                # Enhance plot
                fig = enhance_plot(fig)
                
                # Handle publication quality option
                if pub_quality:
                    # Build custom title or use default
                    custom_title = pub_title if pub_title else f"Combined Performance - {st.session_state.processed_data['metadata']['cell_id']}"
                    
                    # Build axis ranges if provided
                    x_range = None
                    if x_min is not None and x_max is not None:
                        x_range = [x_min, x_max]
                    
                    y_range = None
                    if y_min is not None and y_max is not None:
                        y_range = [y_min, y_max]
                    
                    # Generate publication quality plot
                    mpl_fig, buf = generate_publication_plot(
                        fig, 
                        title=custom_title,
                        xlabel="Cycle Number",
                        ylabel="Performance Metrics",
                        x_range=x_range,
                        y_range=y_range,
                        is_cycle_plot=True
                    )
                    
                    # Display the matplotlib figure directly in Streamlit
                    st.pyplot(mpl_fig)
                    plt.close(mpl_fig)  # Close the figure to free memory
                    
                    # Add download button for the publication quality figure
                    st.download_button(
                        label="Download Publication Quality PNG",
                        data=buf,
                        file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_combined_perf_pub.png",
                        mime="image/png",
                    )
                else:
                    # Display the regular plotly chart if not using publication quality
                    st.plotly_chart(fig, use_container_width=True)
    
    # Option to download complete dataset
    st.subheader("Download Raw Data")
    st.download_button(
        label="Download Complete Dataset as CSV",
        data=st.session_state.processed_data['df'].to_csv(index=False).encode('utf-8'),
        file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_raw_data.csv",
        mime='text/csv',
    )

# Add Multi-Cell Comparison tabs if multiple cells are available
if st.session_state.has_multiple_cells:
    st.markdown("---")
    st.header("🔬 Multi-Cell Comparison")
    
    # Show selected cells information
    selected_cells_info = ", ".join([st.session_state.cells[key]['display_name'] for key in st.session_state.selected_cells])
    st.markdown(f"**Comparing:** {selected_cells_info}")
    
    # Tabs for different comparison plot types
    mc_tab1, mc_tab2, mc_tab3 = st.tabs([
        "Capacity vs Cycle", 
        "Charge-Discharge", 
        "Cell Information"
    ])
    
    # Tab 1: Multi-Cell Capacity
    with mc_tab1:
        st.subheader("Capacity vs Cycle Comparison")
        
        # Add option for publication quality
        pub_quality = st.checkbox("Publication Quality Plot", key="pub_mc_capacity", value=False)
        
        # Add publication quality controls
        if pub_quality:
            with st.expander("Publication Plot Settings", expanded=False):
                pub_title = st.text_input("Custom Plot Title", value="", key="pub_title_mc_capacity")
                col1, col2 = st.columns(2)
                with col1:
                    x_min = st.number_input("X-axis Min", value=None, key="x_min_mc_capacity")
                    y_min = st.number_input("Y-axis Min", value=None, key="y_min_mc_capacity")
                with col2:
                    x_max = st.number_input("X-axis Max", value=None, key="x_max_mc_capacity")
                    y_max = st.number_input("Y-axis Max", value=None, key="y_max_mc_capacity")
        
        if st.button("Generate Comparison Plot", key="btn_mc_capacity"):
            with st.spinner("Generating multi-cell capacity comparison..."):
                fig = go.Figure()
                
                # Plot each selected cell
                for cell_key in st.session_state.selected_cells:
                    cell_data = st.session_state.cells[cell_key]
                    
                    # Get discharge capacity data for each cycle
                    df = cell_data['df']
                    discharge_df = df[df['Step_Type'] == 'Discharge']
                    capacity_data = discharge_df.groupby('Cycle_Index')['Capacity'].max().reset_index()
                    
                    # Add trace for this cell
                    fig.add_trace(go.Scatter(
                        x=capacity_data['Cycle_Index'],
                        y=capacity_data['Capacity'],
                        mode='lines+markers',
                        name=f"{cell_data['display_name']} ({cell_data['metadata']['cathode_type']})"
                    ))
                
                # Update layout
                fig.update_layout(
                    title="Capacity vs Cycle Comparison",
                    xaxis_title="Cycle Number",
                    yaxis_title="Capacity (mAh/g)",
                    legend_title="Cells"
                )
                
                # Enhance plot
                fig = enhance_plot(fig)
                
                # Handle publication quality option
                if pub_quality:
                    # Build custom title or use default
                    custom_title = pub_title if pub_title else "Capacity vs Cycle Comparison"
                    
                    # Build axis ranges if provided
                    x_range = None
                    if x_min is not None and x_max is not None:
                        x_range = [x_min, x_max]
                    
                    y_range = None
                    if y_min is not None and y_max is not None:
                        y_range = [y_min, y_max]
                    
                    # Generate publication quality plot
                    mpl_fig, buf = generate_publication_plot(
                        fig, 
                        title=custom_title,
                        xlabel="Cycle Number",
                        ylabel="Capacity (mAh/g)",
                        x_range=x_range,
                        y_range=y_range
                    )
                    
                    # Display the matplotlib figure directly in Streamlit
                    st.pyplot(mpl_fig)
                    plt.close(mpl_fig)  # Close the figure to free memory
                    
                    # Add download button for the publication quality figure
                    st.download_button(
                        label="Download Publication Quality PNG",
                        data=buf,
                        file_name="multi_cell_capacity_comparison_pub.png",
                        mime="image/png",
                    )
                else:
                    # Display the regular plotly chart if not using publication quality
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create download option for comparison data
                comparison_data = pd.DataFrame()
                
                for cell_key in st.session_state.selected_cells:
                    cell_data = st.session_state.cells[cell_key]
                    df = cell_data['df']
                    discharge_df = df[df['Step_Type'] == 'Discharge']
                    capacity_series = discharge_df.groupby('Cycle_Index')['Capacity'].max()
                    
                    # Add to comparison dataframe
                    comparison_data[f"{cell_data['display_name']}"] = pd.Series(capacity_series)
                
                # Reset index to make Cycle_Index a column
                comparison_data = comparison_data.reset_index().rename(columns={'index': 'Cycle_Index'})
                
                st.download_button(
                    label="Download Comparison Data as CSV",
                    data=comparison_data.to_csv(index=False).encode('utf-8'),
                    file_name="multi_cell_capacity_comparison.csv",
                    mime='text/csv',
                )
    
    # Tab 2: Multi-Cell Charge-Discharge
    with mc_tab2:
        st.subheader("Charge-Discharge Curves Comparison")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Input for cycle selection
            compare_cycle = st.number_input("Cycle to compare:", min_value=1, value=1)
            highlight_cell = st.selectbox(
                "Highlight cell:", 
                ["None"] + [st.session_state.cells[key]['display_name'] for key in st.session_state.selected_cells]
            )
            # Add option to show charge, discharge, or both
            curve_display = st.radio("Display curves:", ["Both", "Discharge Only", "Charge Only"])
            # Add option for publication quality
            pub_quality = st.checkbox("Publication Quality Plot", key="pub_mc_discharge", value=False)
            
            # Add publication quality controls
            if pub_quality:
                with st.expander("Publication Plot Settings", expanded=False):
                    pub_title = st.text_input("Custom Plot Title", value="", key="pub_title_mc_discharge")
                    x_min = st.number_input("X-axis Min", value=None, key="x_min_mc_discharge")
                    x_max = st.number_input("X-axis Max", value=None, key="x_max_mc_discharge")
                    y_min = st.number_input("Y-axis Min", value=None, key="y_min_mc_discharge")
                    y_max = st.number_input("Y-axis Max", value=None, key="y_max_mc_discharge")
        
        if st.button("Generate Comparison Plot", key="btn_mc_cd"):
            with st.spinner("Generating multi-cell charge-discharge comparison..."):
                fig = go.Figure()
                
                # Plot charge and/or discharge curve for selected cycle for each cell
                for cell_key in st.session_state.selected_cells:
                    cell_data = st.session_state.cells[cell_key]
                    cell_name = cell_data['display_name']
                    
                    # Get data for the selected cycle
                    df = cell_data['df']
                    cycle_df = df[df['Cycle_Index'] == compare_cycle]
                    
                    # Set line width based on highlight selection
                    line_width = 3 if highlight_cell == cell_name else 1.5
                    opacity = 1.0 if highlight_cell == "None" or highlight_cell == cell_name else 0.7
                    
                    # Determine which curves to display
                    if curve_display == "Both" or curve_display == "Discharge Only":
                        # Get discharge data for this cycle
                        discharge_df = cycle_df[cycle_df['Step_Type'] == 'Discharge']
                        
                        # Sort by voltage (descending for discharge)
                        discharge_df = discharge_df.sort_values('Voltage', ascending=False)
                        
                        # Add discharge trace
                        if not discharge_df.empty:
                            fig.add_trace(go.Scatter(
                                x=discharge_df['Capacity'],
                                y=discharge_df['Voltage'],
                                mode='lines',
                                line=dict(width=line_width),
                                opacity=opacity,
                                name=f"{cell_name} ({cell_data['metadata']['cathode_type']})",
                                legendgroup=cell_name,
                                showlegend=True
                            ))
                    
                    if curve_display == "Both" or curve_display == "Charge Only":
                        # Get charge data for this cycle
                        charge_df = cycle_df[cycle_df['Step_Type'] == 'Charge']
                        
                        # Sort by voltage (ascending for charge)
                        charge_df = charge_df.sort_values('Voltage', ascending=True)
                        
                        # Add charge trace
                        if not charge_df.empty:
                            fig.add_trace(go.Scatter(
                                x=charge_df['Capacity'],
                                y=charge_df['Voltage'],
                                mode='lines',
                                line=dict(width=line_width, dash='dash'),
                                opacity=opacity,
                                name=f"{cell_name} ({cell_data['metadata']['cathode_type']})",
                                legendgroup=cell_name,
                                showlegend=(curve_display == "Charge Only")
                            ))
                
                # Update layout
                title_suffix = ""
                if curve_display == "Discharge Only":
                    title_suffix = " - Discharge"
                elif curve_display == "Charge Only":
                    title_suffix = " - Charge"
                else:
                    title_suffix = " (solid=discharge, dash=charge)"
                
                fig.update_layout(
                    title=f"Charge-Discharge Curve Comparison - Cycle {compare_cycle}{title_suffix}",
                    xaxis_title="Capacity (mAh/g)",
                    yaxis_title="Voltage (V)",
                    legend_title="Cells"
                )
                
                # Enhance plot
                fig = enhance_plot(fig)
                
                # Handle publication quality option
                if pub_quality:
                    # Build custom title or use default
                    custom_title = pub_title if pub_title else f"Charge-Discharge Curve Comparison - Cycle {compare_cycle}{title_suffix}"
                    
                    # Build axis ranges if provided
                    x_range = None
                    if x_min is not None and x_max is not None:
                        x_range = [x_min, x_max]
                    
                    y_range = None
                    if y_min is not None and y_max is not None:
                        y_range = [y_min, y_max]
                    
                    # Generate publication quality plot
                    mpl_fig, buf = generate_publication_plot(
                        fig, 
                        title=custom_title,
                        xlabel="Capacity (mAh/g)",
                        ylabel="Voltage (V)",
                        x_range=x_range,
                        y_range=y_range
                    )
                    
                    # Display the matplotlib figure directly in Streamlit
                    st.pyplot(mpl_fig)
                    plt.close(mpl_fig)  # Close the figure to free memory
                    
                    # Add download button for the publication quality figure
                    st.download_button(
                        label="Download Publication Quality PNG",
                        data=buf,
                        file_name=f"charge_discharge_comparison_cycle_{compare_cycle}_pub.png",
                        mime="image/png",
                    )
                else:
                    # Display the regular plotly chart if not using publication quality
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create composite dataframe for download
                all_data = pd.DataFrame()
                
                for cell_key in st.session_state.selected_cells:
                    cell_data = st.session_state.cells[cell_key]
                    cell_name = cell_data['display_name']
                    
                    # Get and process data
                    df = cell_data['df']
                    cycle_df = df[df['Cycle_Index'] == compare_cycle]
                    
                    if curve_display == "Both" or curve_display == "Discharge Only":
                        discharge_df = cycle_df[cycle_df['Step_Type'] == 'Discharge'].sort_values('Voltage', ascending=False)
                        if not discharge_df.empty:
                            # Create cell-specific columns for discharge
                            cell_discharge = pd.DataFrame({
                                f'Capacity_Discharge_{cell_name}': discharge_df['Capacity'],
                                f'Voltage_Discharge_{cell_name}': discharge_df['Voltage']
                            })
                            
                            # Add to composite dataframe
                            if all_data.empty:
                                all_data = cell_discharge
                            else:
                                all_data = pd.concat([all_data, cell_discharge], axis=1)
                    
                    if curve_display == "Both" or curve_display == "Charge Only":
                        charge_df = cycle_df[cycle_df['Step_Type'] == 'Charge'].sort_values('Voltage', ascending=True)
                        if not charge_df.empty:
                            # Create cell-specific columns for charge
                            cell_charge = pd.DataFrame({
                                f'Capacity_Charge_{cell_name}': charge_df['Capacity'],
                                f'Voltage_Charge_{cell_name}': charge_df['Voltage']
                            })
                            
                            # Add to composite dataframe
                            if all_data.empty:
                                all_data = cell_charge
                            else:
                                all_data = pd.concat([all_data, cell_charge], axis=1)
                
                # Download options
                st.download_button(
                    label="Download Comparison Data as CSV",
                    data=all_data.to_csv(index=False).encode('utf-8'),
                    file_name=f"charge_discharge_comparison_cycle_{compare_cycle}.csv",
                    mime='text/csv',
                )
    
    # Tab 3: Cell Information
    with mc_tab3:
        st.subheader("Cell Information Comparison")
        
        # Create a comparison table of key metadata
        comparison_table = []
        
        # Define metadata fields to display
        metadata_fields = [
            "cell_id", "cathode_type", "anode_type", 
            "cathode_mass", "active_material_percentage",
            "date", "electrolyte", "c_rate"
        ]
        
        # Field display names
        field_names = {
            "cell_id": "Cell ID",
            "cathode_type": "Cathode Type",
            "anode_type": "Anode Type",
            "cathode_mass": "Cathode Mass",
            "active_material_percentage": "Active Material %",
            "date": "Test Date",
            "electrolyte": "Electrolyte",
            "c_rate": "C-Rate"
        }
        
        # Build table rows
        for field in metadata_fields:
            row = {"Field": field_names.get(field, field)}
            
            for cell_key in st.session_state.selected_cells:
                cell_data = st.session_state.cells[cell_key]
                cell_name = cell_data['display_name']
                metadata = cell_data['metadata']
                
                # Add value if exists
                row[cell_name] = metadata.get(field, "N/A")
            
            comparison_table.append(row)
        
        # Convert to DataFrame for display
        comparison_df = pd.DataFrame(comparison_table)
        st.dataframe(comparison_df, use_container_width=True)

# Welcome info when no data is loaded
else:
    st.markdown("""
    ## Instructions
    1. Upload your Excel file with electrochemical test data using the sidebar
    2. Fill in the required metadata fields:
       - **Cell ID** (mandatory)
       - **Cathode Type** (mandatory)
       - **Anode Type** (mandatory)
       - Other fields are optional but recommended for comprehensive analysis
    3. Click "Process Data" to analyze your file
    4. Choose visualization options or add to multi-cell comparison:
        - Use the single-cell visualization tabs to view detailed analysis
        - Click "Add to Comparison" to include in multi-cell analysis
        - Click "Load New Cell" to process another cell
    5. Compare multiple cells:
        - Process and add multiple cells to comparison
        - Select cells in the sidebar for comparison
        - Use the multi-cell comparison tabs to view comparative analysis
    6. Download data in CSV format for further analysis
    
    This application analyzes electrochemical test data from coin cells to evaluate battery performance.
    """)
    
    # Example plot image would go here
    st.info("Upload data to get started!")

# Footer
st.markdown("---")
st.markdown("Battery Lab - Electrochemical Test Data Analyzer | © 2025") 