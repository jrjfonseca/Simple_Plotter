from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import Optional, List
import os
import io
import json
import pandas as pd
import plotly.io as pio
from pathlib import Path
import sys
import tempfile
import shutil
import logging
from datetime import datetime

# Configure logging
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

# Create FastAPI app
app = FastAPI(title="Battery Lab Data Analyzer")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Global variable to store processed data
processed_data = {}

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

# Define route for homepage
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define route for file upload
@app.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    excel_file: UploadFile = File(...),
    cell_id: str = Form(...),
    cathode_type: str = Form(...),
    anode_type: str = Form(...),
    cathode_mass: float = Form(...),
    collector_mass: float = Form(...),
    active_material_percentage: float = Form(...),
    # New metadata fields - all optional
    date: Optional[str] = Form(None),
    cathode_composition: Optional[str] = Form(None),
    cathode_mixing_method: Optional[str] = Form(None),
    anode_composition: Optional[str] = Form(None),
    anode_mass: Optional[str] = Form(None),
    anode_mixing_method: Optional[str] = Form(None),
    electrolyte: Optional[str] = Form(None),
    electrolyte_quantity: Optional[str] = Form(None),
    channel: Optional[str] = Form(None),
    voltage_range: Optional[str] = Form(None),
    c_rate: Optional[str] = Form(None),
    pressure: Optional[str] = Form(None)
):
    global processed_data
    
    logger.info(f"Upload started for file: {excel_file.filename}, cell_id: {cell_id}")
    
    # Create a temporary file to store the uploaded excel file
    temp_file = Path(tempfile.gettempdir()) / excel_file.filename
    
    try:
        # Save the uploaded file
        logger.info(f"Saving uploaded file to: {temp_file}")
        with open(temp_file, "wb") as f:
            content = await excel_file.read()
            f.write(content)
        
        logger.info("File saved successfully")
        
        # Create cell metadata dictionary with all fields
        cell_metadata = {
            'cell_id': cell_id,
            'cathode_type': cathode_type,
            'anode_type': anode_type,
            'cathode_mass': f"{cathode_mass} mg",
            'collector_mass': f"{collector_mass} mg",
            'active_material_percentage': active_material_percentage
        }
        
        # Add optional metadata if provided
        if date:
            cell_metadata['date'] = date
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
            
        # Set current date if not provided
        if not date:
            current_date = datetime.now().strftime("%Y-%m-%d")
            cell_metadata['date'] = current_date
            
        logger.info(f"Metadata collected: {cell_metadata}")
        
        # Load and process data
        logger.info(f"Processing data with electrochemistry module. Parameters: total_mass={cathode_mass}, am_percentage={active_material_percentage}, collector_mass={collector_mass}")
        df_normalized = ec.load_data(
            temp_file,
            total_mass=cathode_mass,
            am_percentage=active_material_percentage,
            collector_mass=collector_mass
        )
        
        logger.info(f"Data processed successfully. DataFrame shape: {df_normalized.shape}")
        
        # Store the processed data and metadata in the global variable
        processed_data = {
            'df': df_normalized,
            'metadata': cell_metadata,
            'filename': excel_file.filename
        }
        
        logger.info("Redirecting to results page")
        # Return the results page which will show plot options
        return templates.TemplateResponse(
            "results.html", 
            {
                "request": request,
                "cell_id": cell_id,
                "filename": excel_file.filename,
                "has_data": True
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        # Return the error message
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "error": f"Error processing file: {str(e)}"
            }
        )
    finally:
        # Clean up the temporary file
        logger.info(f"Cleaning up temporary file: {temp_file}")
        if temp_file.exists():
            temp_file.unlink()

# Generate charge-discharge plot
@app.get("/plot/charge_discharge", response_class=HTMLResponse)
async def plot_charge_discharge(request: Request, cycles: str = "1,10,50,100"):
    global processed_data
    
    if not processed_data or 'df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data has been processed. Please upload a file first.")
    
    # Parse cycles to plot
    cycles_to_plot = [int(c) for c in cycles.split(',')]
    
    # Generate plot
    fig = ec.plot_charge_discharge(
        processed_data['df'],
        cycles_to_plot=cycles_to_plot,
        cell_metadata=processed_data['metadata']
    )
    
    # Enhance plot appearance
    fig = enhance_plot(fig)
    
    # Convert plot to HTML with responsive settings
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
    
    return templates.TemplateResponse(
        "plot.html", 
        {
            "request": request,
            "plot_html": plot_html,
            "title": "Charge-Discharge Curves",
            "cell_id": processed_data['metadata']['cell_id'],
            "plot_type": "charge_discharge",
            "cycles": cycles
        }
    )

# Generate capacity vs cycle plot
@app.get("/plot/capacity", response_class=HTMLResponse)
async def plot_capacity(request: Request):
    global processed_data
    
    if not processed_data or 'df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data has been processed. Please upload a file first.")
    
    # Generate plot
    fig = ec.plot_capacity_vs_cycle(
        processed_data['df'],
        cell_metadata=processed_data['metadata']
    )
    
    # Enhance plot appearance
    fig = enhance_plot(fig)
    
    # Convert plot to HTML with responsive settings
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
    
    return templates.TemplateResponse(
        "plot.html", 
        {
            "request": request,
            "plot_html": plot_html,
            "title": "Capacity vs Cycle",
            "cell_id": processed_data['metadata']['cell_id'],
            "plot_type": "capacity"
        }
    )

# Generate state of health plot
@app.get("/plot/soh", response_class=HTMLResponse)
async def plot_soh(request: Request):
    global processed_data
    
    if not processed_data or 'df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data has been processed. Please upload a file first.")
    
    # Generate plot
    fig = ec.plot_state_of_health(
        processed_data['df'],
        cell_metadata=processed_data['metadata']
    )
    
    # Enhance plot appearance
    fig = enhance_plot(fig)
    
    # Convert plot to HTML with responsive settings
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
    
    return templates.TemplateResponse(
        "plot.html", 
        {
            "request": request,
            "plot_html": plot_html,
            "title": "State of Health vs Cycle",
            "cell_id": processed_data['metadata']['cell_id'],
            "plot_type": "soh"
        }
    )

# Generate coulombic efficiency plot
@app.get("/plot/coulombic_efficiency", response_class=HTMLResponse)
async def plot_coulombic_efficiency(request: Request):
    global processed_data
    
    if not processed_data or 'df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data has been processed. Please upload a file first.")
    
    # Generate plot
    fig = ec.plot_coulombic_efficiency(
        processed_data['df'],
        cell_metadata=processed_data['metadata']
    )
    
    # Enhance plot appearance
    fig = enhance_plot(fig)
    
    # Convert plot to HTML with responsive settings
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
    
    return templates.TemplateResponse(
        "plot.html", 
        {
            "request": request,
            "plot_html": plot_html,
            "title": "Coulombic Efficiency vs Cycle",
            "cell_id": processed_data['metadata']['cell_id'],
            "plot_type": "coulombic_efficiency"
        }
    )

# Generate voltage vs time plot
@app.get("/plot/voltage_time", response_class=HTMLResponse)
async def plot_voltage_time(request: Request):
    global processed_data
    
    if not processed_data or 'df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data has been processed. Please upload a file first.")
    
    # Generate plot
    fig = ec.plot_voltage_time(
        processed_data['df'],
        cell_metadata=processed_data['metadata']
    )
    
    # Enhance plot appearance
    fig = enhance_plot(fig)
    
    # Convert plot to HTML with responsive settings
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
    
    return templates.TemplateResponse(
        "plot.html", 
        {
            "request": request,
            "plot_html": plot_html,
            "title": "Voltage vs Time",
            "cell_id": processed_data['metadata']['cell_id'],
            "plot_type": "voltage_time"
        }
    )

# Generate differential capacity plot
@app.get("/plot/differential_capacity", response_class=HTMLResponse)
async def plot_differential_capacity(request: Request, cycles: str = "1,2"):
    global processed_data
    
    if not processed_data or 'df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data has been processed. Please upload a file first.")
    
    # Parse cycles to plot
    cycles_to_plot = [int(c) for c in cycles.split(',')]
    
    # Generate plot
    fig = ec.plot_differential_capacity(
        processed_data['df'],
        cycles=cycles_to_plot,
        cell_metadata=processed_data['metadata']
    )
    
    # Enhance plot appearance
    fig = enhance_plot(fig)
    
    # Convert plot to HTML with responsive settings
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
    
    return templates.TemplateResponse(
        "plot.html", 
        {
            "request": request,
            "plot_html": plot_html,
            "title": "Differential Capacity (dQ/dV)",
            "cell_id": processed_data['metadata']['cell_id'],
            "plot_type": "differential_capacity",
            "cycles": cycles
        }
    )

# Generate combined performance plot
@app.get("/plot/combined_performance", response_class=HTMLResponse)
async def plot_combined_performance(request: Request, cycles: str = "1,10,50,100"):
    global processed_data
    
    if not processed_data or 'df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data has been processed. Please upload a file first.")
    
    # Parse cycles to plot
    cycles_to_plot = [int(c) for c in cycles.split(',')]
    
    # Generate plot
    fig = ec.plot_combined_performance(
        processed_data['df'],
        cycles_to_plot=cycles_to_plot,
        cell_metadata=processed_data['metadata']
    )
    
    # Enhance plot appearance
    fig = enhance_plot(fig)
    
    # Convert plot to HTML with responsive settings
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
    
    return templates.TemplateResponse(
        "plot.html", 
        {
            "request": request,
            "plot_html": plot_html,
            "title": "Combined Performance",
            "cell_id": processed_data['metadata']['cell_id'],
            "plot_type": "combined_performance",
            "cycles": cycles
        }
    )

# Download plot data as CSV
@app.get("/download/{plot_type}")
async def download_data(plot_type: str):
    global processed_data
    
    if not processed_data or 'df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data has been processed. Please upload a file first.")
    
    df = processed_data['df']
    cell_id = processed_data['metadata']['cell_id']
    
    # Create a CSV buffer
    buffer = io.StringIO()
    
    if plot_type == "raw_data":
        # Download the complete processed dataframe
        df.to_csv(buffer, index=False)
        filename = f"{cell_id}_raw_data.csv"
    elif plot_type == "charge_discharge":
        # Extract charge-discharge data
        charge_data = df[df['Step_Type'] == 'Charge'][['Cycle_Index', 'Voltage', 'Capacity']]
        discharge_data = df[df['Step_Type'] == 'Discharge'][['Cycle_Index', 'Voltage', 'Capacity']]
        pd.concat([charge_data, discharge_data]).to_csv(buffer, index=False)
        filename = f"{cell_id}_charge_discharge_data.csv"
    elif plot_type == "capacity":
        # Extract capacity data
        capacity_data = df[df['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max().reset_index()
        capacity_data.to_csv(buffer, index=False)
        filename = f"{cell_id}_capacity_data.csv"
    elif plot_type == "soh":
        # Extract state of health data
        discharge_capacity = df[df['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max()
        initial_capacity = discharge_capacity.iloc[0]
        soh = (discharge_capacity / initial_capacity) * 100
        soh_df = pd.DataFrame({'Cycle_Index': soh.index, 'State_of_Health': soh.values})
        soh_df.to_csv(buffer, index=False)
        filename = f"{cell_id}_soh_data.csv"
    elif plot_type == "coulombic_efficiency":
        # Extract coulombic efficiency data
        discharge_capacity = df[df['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max()
        charge_capacity = df[df['Step_Type'] == 'Charge'].groupby('Cycle_Index')['Capacity'].max()
        ce = (discharge_capacity / charge_capacity) * 100
        ce_df = pd.DataFrame({'Cycle_Index': ce.index, 'Coulombic_Efficiency': ce.values})
        ce_df.to_csv(buffer, index=False)
        filename = f"{cell_id}_coulombic_efficiency_data.csv"
    elif plot_type == "voltage_time":
        # Extract voltage-time data
        voltage_time = df[['Test_Time', 'Voltage']]
        voltage_time.to_csv(buffer, index=False)
        filename = f"{cell_id}_voltage_time_data.csv"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid plot type: {plot_type}")
    
    # Prepare response
    buffer.seek(0)
    
    # Return the CSV file
    return StreamingResponse(
        io.StringIO(buffer.getvalue()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# Run the app with uvicorn if executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 