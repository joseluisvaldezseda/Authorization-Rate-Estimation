# Credit Expansion Intelligence

A dual-model machine learning system for strategic credit portfolio optimization in Mexico, providing granular risk assessment and market penetration analysis at the neighborhood level.

## Live Demo

**[View Application](https://risk-rate-estimation.streamlit.app/)**

## Overview

This application combines two independent classification models to deliver actionable intelligence for credit expansion strategies:

- **Credit Risk Model**: Predicts default probability using socio-demographic indicators
- **Authorization Rate Model**: Estimates market penetration and approval likelihood

The system operates at the colonia (neighborhood) level, enabling precise geographic targeting rather than broad municipal generalizations.

## Features

**Strategy Matrix**
- Risk vs. Growth quadrant analysis
- Visual identification of expansion zones
- Top opportunity rankings by composite score

**Granular Explorer**
- Detailed neighborhood-level data
- Risk distribution analytics
- Authorization potential metrics

**Dynamic Filtering**
- Cascading geographic selection (State → Municipality)
- Customizable probability thresholds
- Real-time data updates

## Technical Stack

- **Framework**: Streamlit
- **Visualization**: Plotly Express & Graph Objects
- **ML Models**: XGBoost (saved as `.pkl`)
- **Data Processing**: Pandas, NumPy

## Model Architecture

### Risk Classification Model
- **Output**: Three-class probabilities (Low, Medium, High Risk)
- **Application**: Credit limit optimization and interest rate calibration

### Authorization Rate Model
- **Output**: Binary probability (High vs. Low authorization rate)
- **Application**: Marketing resource allocation and conversion forecasting

### Composite Scoring
```
Opportunity_Score = (Prob_Low_Risk × 0.6) + (Prob_High_Auth × 0.4)
```

## Data Structure

The application processes `tasa_autorizacion_riesgo_predicho.csv` with the following key fields:

- Geographic identifiers (State, Municipality, Locality, Neighborhood)
- Risk probabilities (High, Medium, Low)
- Authorization probabilities (High, Low)
- Predicted classifications

## Installation

```bash
git clone https://github.com/joseluisvaldezseda/risk-rate-estimation.git
cd risk-rate-estimation
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

```
streamlit
pandas
plotly
numpy
scikit-learn
xgboost
```

## Usage

1. Select geographic scope using sidebar filters
2. Adjust probability thresholds to define opportunity criteria
3. Analyze results through three analytical views:
   - Strategy Matrix for portfolio positioning
   - Granular Explorer for detailed neighborhood data
   - Model Logic for methodology transparency

## File Structure

```
.
├── app.py                                    # Main application
├── tasa_autorizacion_riesgo_predicho.csv    # Predictions dataset
├── modelo_xgboost_optimizado.pkl            # Trained model
├── label_encoder.pkl                         # Encoding artifacts
├── Modelo Tasa de Autorizacion.ipynb        # Model development notebook
├── requirements.txt                          # Dependencies
└── README.md                                 # Documentation
```

## Data Engineering

- **Granularity**: Colonia-level analysis across Mexico
- **ETL Pipeline**: Automated extraction of socio-demographic metrics
- **Encoding**: Latin-1 to UTF-8 conversion for proper Spanish character handling

## Business Applications

- Credit product expansion planning
- Risk-adjusted pricing strategies
- Geographic marketing optimization
- Portfolio diversification analysis

## Author

**JV.DATA**

Built with dual-model classification architecture for strategic credit intelligence.

## License

This project is available for educational and commercial use.
