# Battery Lab - Electrochemical Test Data Analyzer

A web application for analyzing and visualizing electrochemical test data from coin cells.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-STREAMLIT-CLOUD-URL-HERE/)

## Features

- Upload Excel files containing electrochemical test data
- Input metadata for coin cells (Cell ID, Anode Type, Cathode Type, etc.)
- Generate interactive visualizations:
  - Charge-Discharge Curves
  - Capacity vs Cycle
  - State of Health
  - Coulombic Efficiency
  - Voltage vs Time
  - Differential Capacity (dQ/dV)
  - Combined Performance
- Download plot data as CSV for further analysis
- Interactive plot customization (cycle selection)

## Streamlit Cloud Deployment

This application is deployed on Streamlit Cloud for easy access:

1. Visit [https://YOUR-STREAMLIT-CLOUD-URL-HERE/](https://YOUR-STREAMLIT-CLOUD-URL-HERE/)
2. Upload your Excel file with electrochemical data
3. Fill in the metadata fields
4. Generate and interact with visualizations

## Local Setup

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/battery-lab.git
   cd battery-lab
   ```

2. Create a virtual environment (optional but recommended):
   ```
   # Using conda
   conda create -n battery-lab python=3.9
   conda activate battery-lab
   
   # Or using venv
   python -m venv battery-lab-env
   source battery-lab-env/bin/activate  # Linux/Mac
   # OR
   battery-lab-env\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements-streamlit.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

5. Access the application at [http://localhost:8501](http://localhost:8501)

## Application Workflow

1. **Upload Data**: Upload an Excel file with electrochemical data and provide cell metadata
2. **Process Data**: Click "Process Data" to analyze the file
3. **Select Visualization**: Choose from various plot types to analyze the data
4. **Customize Plots**: For some plots, specify which cycles to display
5. **Download Data**: Export the plot data as CSV for further analysis

## Project Structure

- `streamlit_app.py` - Streamlit application entry point
- `example_functions/lib/` - Library containing electrochemistry analysis tools
- `requirements-streamlit.txt` - Dependencies for Streamlit deployment

## Data Format

The application expects Excel files with the following data columns:
- Cycle_Index: Cycle number
- Voltage: Cell voltage (V)
- Current: Applied current (A)
- Test_Time: Elapsed test time
- Step_Type: Charge/Discharge indicators

## Creating Your Own Deployment

To deploy this app on your own Streamlit Cloud:

1. Fork this repository
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app, pointing to your forked repository
4. Set the main file path to `streamlit_app.py`
5. Deploy! 