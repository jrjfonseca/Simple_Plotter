import streamlit as st
import pandas as pd
import plotly.io as pio
from pathlib import Path
import sys
import tempfile
import os
from datetime import datetime
import logging

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

# Set default Plotly template for better appearance
pio.templates.default = "plotly_white"

# Initialize session state for storing processed data
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
    st.session_state.has_data = False
    st.session_state.form_submitted = False

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

# Main title
st.title("Battery Lab - Electrochemical Test Data Analyzer")
st.markdown("Upload your Excel file containing coin cell test data to visualize performance")

# Sidebar for inputs
with st.sidebar:
    st.header("🔋 Load Your Cell Data")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Create a form for metadata
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
        
        # Handle the form submission - this runs after the form based on the session state
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
                    
                    st.success("Data processed successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

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
                
                # Display plot
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
        
        if st.button("Generate Plot", key="btn_capacity"):
            with st.spinner("Generating plot..."):
                # Generate plot
                fig = ec.plot_capacity_vs_cycle(
                    st.session_state.processed_data['df'],
                    cell_metadata=st.session_state.processed_data['metadata']
                )
                
                # Enhance plot
                fig = enhance_plot(fig)
                
                # Display plot
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
        
        if st.button("Generate Plot", key="btn_soh"):
            with st.spinner("Generating plot..."):
                # Generate plot
                fig = ec.plot_state_of_health(
                    st.session_state.processed_data['df'],
                    cell_metadata=st.session_state.processed_data['metadata']
                )
                
                # Enhance plot
                fig = enhance_plot(fig)
                
                # Display plot
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
        
        if st.button("Generate Plot", key="btn_ce"):
            with st.spinner("Generating plot..."):
                # Generate plot
                fig = ec.plot_coulombic_efficiency(
                    st.session_state.processed_data['df'],
                    cell_metadata=st.session_state.processed_data['metadata']
                )
                
                # Enhance plot
                fig = enhance_plot(fig)
                
                # Display plot
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
        
        if st.button("Generate Plot", key="btn_vt"):
            with st.spinner("Generating plot..."):
                # Generate plot
                fig = ec.plot_voltage_time(
                    st.session_state.processed_data['df'],
                    cell_metadata=st.session_state.processed_data['metadata']
                )
                
                # Enhance plot
                fig = enhance_plot(fig)
                
                # Display plot
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
                
                # Display plot
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 7: Combined Performance
    with tab7:
        st.subheader("Combined Performance")
        
        # Input for cycle selection
        cp_cycles = st.text_input("Cycles to display (comma-separated)", value="1,2,3,5", key="cp_cycles")
        
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
                
                # Display plot
                st.plotly_chart(fig, use_container_width=True)
    
    # Option to download complete dataset
    st.subheader("Download Raw Data")
    st.download_button(
        label="Download Complete Dataset as CSV",
        data=st.session_state.processed_data['df'].to_csv(index=False).encode('utf-8'),
        file_name=f"{st.session_state.processed_data['metadata']['cell_id']}_raw_data.csv",
        mime='text/csv',
    )

else:
    # Welcome info when no data is loaded
    st.markdown("""
    ## Instructions
    1. Upload your Excel file with electrochemical test data using the sidebar
    2. Fill in the required metadata fields
    3. Click "Process Data" to analyze your file
    4. Choose from different visualization options:
        - Charge-Discharge Curves
        - Capacity vs Cycle
        - State of Health
        - Coulombic Efficiency
        - Voltage vs Time
        - Differential Capacity (dQ/dV)
        - Combined Performance
    5. Download the raw data or specific plot data in CSV format
    
    This application analyzes electrochemical test data from coin cells to evaluate battery performance.
    """)
    
    # Example plot image would go here
    st.info("Upload data to get started!")

# Footer
st.markdown("---")
st.markdown("Battery Lab - Electrochemical Test Data Analyzer | © 2025") 