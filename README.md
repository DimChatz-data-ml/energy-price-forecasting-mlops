# Energy Price Prediction Project (MLOps)

This project is a complete system for predicting energy prices. It handles everything from training the model to monitoring the results in real-time.

## How it Works (Architecture)
The project runs on **Docker**, which means it works the same way on any computer. It includes:

* **FastAPI:** The "brain" that receives data and gives back predictions.
* **PostgreSQL:** The database that saves our training results and every prediction we make.
* **MLflow:** A tool to track our experiments and save different versions of our models.
* **Grafana:** A dashboard to see our predictions in beautiful charts.
* **Adminer:** A simple web page to manage our database.

## 🚀 How to Run the Project

### 1. Start the System
Open your terminal in the project folder and run:
```bash
docker-compose up -d
```

### 2. Train the Model
To train the model and save the results (like RMSE and MAE) in MLflow, run:
```bash
python src/train.py
```

### 3. Get Predictions
Go to `http://localhost:8000/docs` (Swagger UI). 
When you use the `POST /predict` button, the system calculates the price and saves it automatically in the PostgreSQL database.

### 4. Live Monitoring
Open Grafana at `http://localhost:3000` (User: `admin` / Pass: `admin`).
You can see all the saved predictions by running this simple SQL query:
```sql
SELECT 
  timestamp::timestamp AS "time", 
  prediction_eur_mwh AS "Price (€/MWh)",
  model_alias AS "Model"
FROM api_predictions 
ORDER BY 1 DESC;
```

## Data Storage
* **Files:** We use `.csv` and `.parquet` files for raw and cleaned data.
* **Database:** We use PostgreSQL to keep a history of our model's performance and predictions.

## Future Steps (Roadmap)
* **Better Hardware:** Upgrade to 24GB RAM to run larger and more complex models.
* **Automation:** Use **Apache Airflow** to automate the training process every day.
* **Model Registry:** Improve how we choose the best model (Champion) to put into production.

