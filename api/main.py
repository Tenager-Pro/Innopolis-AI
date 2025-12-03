from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
import logging
from datetime import datetime

from .models import (
    CustomerFeatures, ChurnPrediction, BatchPredictionRequest,
    BatchPredictionResponse, ModelInfo, HealthCheck
)
from .predict import predictor
from .database import db_manager

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API для предсказания оттока клиентов с использованием ML моделей",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Зависимости
def get_predictor():
    return predictor

def get_db_manager():
    return db_manager

@app.get("/", include_in_schema=False)
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check(
    predictor_obj: ChurnPredictor = Depends(get_predictor),
    db_manager_obj: DatabaseManager = Depends(get_db_manager)
):
    """Проверка здоровья API"""
    return HealthCheck(
        status="healthy",
        model_loaded=predictor_obj.is_loaded,
        database_connected=db_manager_obj.is_connected,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(predictor_obj: ChurnPredictor = Depends(get_predictor)):
    """Получение информации о модели"""
    try:
        model_info = predictor_obj.get_model_info()
        return ModelInfo(**model_info)
    except Exception as e:
        logger.error(f"Ошибка получения информации о модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения информации о модели: {str(e)}"
        )

@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(
    features: CustomerFeatures,
    customer_id: Optional[int] = None,
    threshold: float = 0.5,
    save_to_db: bool = True,
    predictor_obj: ChurnPredictor = Depends(get_predictor),
    db_manager_obj: DatabaseManager = Depends(get_db_manager)
):
    """Предсказание оттока для одного клиента"""
    try:
        # Валидация threshold
        if not 0 <= threshold <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Threshold должен быть между 0 и 1"
            )
        
        # Предсказание
        prediction = predictor_obj.predict_single(
            features, customer_id, threshold
        )
        
        # Сохранение в БД если требуется
        if save_to_db and customer_id:
            db_manager_obj.save_prediction(
                customer_id, 
                features.dict(), 
                prediction.dict()
            )
        
        logger.info(f"✅ Предсказание для клиента {customer_id}: {prediction.prediction}")
        return prediction
        
    except ValueError as e:
        logger.error(f"Ошибка валидации: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ошибка валидации данных: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка предсказания: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(
    request: BatchPredictionRequest,
    threshold: float = 0.5,
    save_to_db: bool = True,
    predictor_obj: ChurnPredictor = Depends(get_predictor),
    db_manager_obj: DatabaseManager = Depends(get_db_manager)
):
    """Пакетное предсказание оттока для нескольких клиентов"""
    try:
        # Валидация threshold
        if not 0 <= threshold <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Threshold должен быть между 0 и 1"
            )
        
        # Валидация customer_ids
        if request.customer_ids and len(request.customer_ids) != len(request.customers):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Количество customer_ids должно совпадать с количеством customers"
            )
        
        # Пакетное предсказание
        predictions, churn_rate = predictor_obj.predict_batch(
            request.customers, 
            request.customer_ids, 
            threshold
        )
        
        # Сохранение в БД если требуется
        if save_to_db and request.customer_ids:
            for i, prediction in enumerate(predictions):
                if i < len(request.customer_ids):
                    db_manager_obj.save_prediction(
                        request.customer_ids[i],
                        request.customers[i].dict(),
                        prediction.dict()
                    )
        
        logger.info(f"✅ Пакетное предсказание: {len(predictions)} клиентов, отток: {churn_rate:.2%}")
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            churn_rate=churn_rate
        )
        
    except Exception as e:
        logger.error(f"Ошибка пакетного предсказания: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка пакетного предсказания: {str(e)}"
        )

@app.get("/customer/{customer_id}/predict")
async def predict_existing_customer(
    customer_id: int,
    threshold: float = 0.5,
    predictor_obj: ChurnPredictor = Depends(get_predictor),
    db_manager_obj: DatabaseManager = Depends(get_db_manager)
):
    """Предсказание для существующего клиента по ID"""
    try:
        # Получение признаков из БД
        customer_data = db_manager_obj.get_customer_features(customer_id)
        
        if not customer_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Клиент с ID {customer_id} не найден"
            )
        
        # Преобразование в CustomerFeatures
        # Убираем служебные поля
        features_dict = {k: v for k, v in customer_data.items() 
                        if k not in ['CustomerID', 'Churn']}
        
        features = CustomerFeatures(**features_dict)
        
        # Предсказание
        prediction = predictor_obj.predict_single(features, customer_id, threshold)
        
        # Сохранение в БД
        db_manager_obj.save_prediction(customer_id, features_dict, prediction.dict())
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка предсказания для клиента {customer_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка предсказания: {str(e)}"
        )

@app.get("/customer/{customer_id}/history")
async def get_prediction_history(
    customer_id: int,
    limit: int = 10,
    db_manager_obj: DatabaseManager = Depends(get_db_manager)
):
    """История предсказаний для клиента"""
    try:
        history = db_manager_obj.get_predictions_history(customer_id, limit)
        return {
            "customer_id": customer_id,
            "predictions_history": history,
            "total_predictions": len(history)
        }
    except Exception as e:
        logger.error(f"Ошибка получения истории для клиента {customer_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения истории: {str(e)}"
        )

@app.get("/features/list")
async def get_feature_list(predictor_obj: ChurnPredictor = Depends(get_predictor)):
    """Получение списка признаков модели"""
    try:
        return {
            "features": predictor_obj.feature_names,
            "total_features": len(predictor_obj.feature_names)
        }
    except Exception as e:
        logger.error(f"Ошибка получения списка признаков: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения списка признаков: {str(e)}"
        )

# Обработчики ошибок
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Необработанная ошибка: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Внутренняя ошибка сервера"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
