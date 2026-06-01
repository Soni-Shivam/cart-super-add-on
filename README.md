# Cart Super Add-On (CSAO)

Welcome to my **Cart Super Add-On** Project! This project implements an advanced Hybrid Recommendation Engine designed to suggest intelligent, context-aware add-ons to users during the checkout process (similar to Zomato's intelligent cart recommendations).

#### ( Recommended ) For viewing this as a presentation  Please Visit [this Canva Link](https://canva.link/n7jxcgugfpk2x5k)

<img width="1271" height="560" alt="image" src="https://github.com/user-attachments/assets/390e7e6e-5518-46a1-922e-0a1ab0f448d7" />
<img width="579" height="664" alt="image" src="https://github.com/user-attachments/assets/193e1a7a-a545-4ba5-a6eb-3587e1cbd0e5" />
<img width="579" height="664" alt="Screenshot from 2026-06-01 21-53-46" src="https://github.com/user-attachments/assets/bc296427-f5a9-4ce8-a30e-b7b453da1611" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8d0e2b59-faeb-4c37-aae6-215041547cf8" />

<img width="1267" height="639" alt="image" src="https://github.com/user-attachments/assets/595d4ba3-8066-43ec-8055-20af964bbbbf" />
<img width="1267" height="639" alt="image" src="https://github.com/user-attachments/assets/6f856ded-eef0-4512-817c-db55eac8e437" />


## Overview

The CSAO engine is built using a powerful machine learning pipeline comprising a PyTorch neural backbone and a LightGBM re-ranker. It factors in multiple dimensions of context to provide the best add-on recommendations:
- **Cart Context:** What the user is currently ordering.
- **User History:** Past orders, Average Order Value (AOV), and favorite cuisines.
- **Environmental Variables:** Time of day, day of the week (weekend/weekday), and city.

The repository includes both a Streamlit frontend for demonstration/exploration and a FastAPI backend designed to serve a Single Page Application (SPA).

## Project Structure

- `app.py`: A Streamlit application that provides a visual dashboard to simulate users (New, Regular, Power User), explore different restaurant menus, add items to the cart, and see real-time AI-driven recommendations.
- `api.py`: A robust FastAPI backend exposing REST endpoints for the frontend SPA, including `/cart/recommend` for engine predictions and `/checkout` to simulate placing orders.
- `engine.py`: The core algorithmic recommendation engine. It handles data processing, feature computation, and calls the underlying PyTorch and LightGBM models.
- `train_offline.py`: The offline training script. It synthesizes a corpus, trains the neural backbone and re-ranker, and generates all required model artifacts.
- `csao/`: Contains core configuration and taxonomies (menus, categories, cuisines).
- `artifacts/`: Directory where the trained model weights and generated synthetic databases are stored.
- `frontend/`: Contains the static HTML/CSS/JS files for the SPA served by FastAPI.

## Prerequisites

- Python 3.8+
- The required dependencies listed in `requirements.txt`.

## Getting Started

### 1. Install Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

*(Note: Depending on your specific setup, you might need additional packages like `torch`, `lightgbm`, `fastapi`, `uvicorn`, `streamlit` if they are not fully captured in the local requirements).*

### 2. Run Offline Training

Before launching either the Streamlit app or the FastAPI server, you **must** run the offline training pipeline. This script generates the synthetic user database, trains the models, and serializes the artifacts so the application can boot instantly without training on the fly.

```bash
python train_offline.py
```

This will create an `artifacts/` folder containing files like `neural_model.pt`, `lgbm_model.txt`, `user_db.pkl`, and more.

### 3. Launch the Application

You have two choices for running the application interface:

#### Option A: Streamlit Dashboard (Interactive Simulator)
Launch the Streamlit app to explore the smart cart functionality with an interactive UI.

```bash
streamlit run app.py
```
Open the provided local URL (usually `http://localhost:8501`) in your browser.

#### Option B: FastAPI Backend + SPA
If you prefer to run the standalone API which serves the frontend static files:

```bash
python api.py
```
The API server will start on `http://localhost:8000`. You can access the UI directly from that URL.

## Key Features

- **Cold-Start Handling:** Seamlessly routes recommendations for new items or users without prior history.
- **Semantic Taste Profiles:** Matches add-ons based on semantic similarities.
- **Inferred Meal Gap Vector:** Analyzes the cart to guess if the order is missing a main, side, beverage, or dessert, and recommends accordingly.
- **Live Pipeline Execution Logs:** In the Streamlit app, you can view the actual AI thinking process, inference latency, and feature state dictionaries.

## License
Feel free to fork.
