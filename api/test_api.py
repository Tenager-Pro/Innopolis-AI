import pytest
import asyncio
from fastapi.testclient import TestClient
from .main import app
from .models import CustomerFeatures
import json

# Тестовый клиент
client = TestClient(app)

# Тестовые данные
SAMPLE_FEATURES = {
    "Tenure": 12.5,
    "CityTier": 2.0,
    "WarehouseToHome": 8.0,
    "HourSpendOnApp": 3.2,
    "NumberOfDeviceRegistered": 3.0,
    "SatisfactionScore": 4.0,
    "NumberOfAddress": 5.0,
    "Complain": 0.0,
    "OrderAmountHikeFromlastYear": 15.0,
    "CouponUsed": 4.0,
    "OrderCount": 25.0,
    "DaySinceLastOrder": 3.0,
    "CashbackAmount": 250.0
}

class TestChurnAPI:
    """Тесты для API предсказания оттока"""
    
    def test_health_check(self):
        """Тест health check эндпоинта"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "database_connected" in data
    
    def test_model_info(self):
        """Тест получения информации о модели"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "feature_count" in data
        assert "feature_names" in data
    
    def test_predict_single_valid(self):
        """Тест предсказания с валидными данными"""
        response = client.post("/predict", json=SAMPLE_FEATURES)
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert "prediction" in data
        assert "model_version" in data
        assert 0 <= data["churn_probability"] <= 1
        assert isinstance(data["prediction"], bool)
    
    def test_predict_single_with_customer_id(self):
        """Тест предсказания с customer_id"""
        features_with_id = SAMPLE_FEATURES.copy()
        response = client.post("/predict?customer_id=12345", json=features_with_id)
        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] == 12345
    
    def test_predict_single_custom_threshold(self):
        """Тест предсказания с кастомным threshold"""
        response = client.post("/predict?threshold=0.3", json=SAMPLE_FEATURES)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
    
    def test_predict_single_invalid_threshold(self):
        """Тест предсказания с невалидным threshold"""
        response = client.post("/predict?threshold=1.5", json=SAMPLE_FEATURES)
        assert response.status_code == 400
    
    def test_predict_single_invalid_features(self):
        """Тест предсказания с невалидными признаками"""
        invalid_features = SAMPLE_FEATURES.copy()
        invalid_features["SatisfactionScore"] = 10.0  # Невалидное значение
        
        response = client.post("/predict", json=invalid_features)
        assert response.status_code == 400
    
    def test_predict_batch_valid(self):
        """Тест пакетного предсказания с валидными данными"""
        batch_request = {
            "customers": [SAMPLE_FEATURES, SAMPLE_FEATURES],
            "customer_ids": [1001, 1002]
        }
        
        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total_customers" in data
        assert "churn_rate" in data
        assert len(data["predictions"]) == 2
        assert 0 <= data["churn_rate"] <= 1
    
    def test_predict_batch_mismatched_ids(self):
        """Тест пакетного предсказания с несовпадающими ID"""
        batch_request = {
            "customers": [SAMPLE_FEATURES],
            "customer_ids": [1001, 1002]  # Больше ID чем клиентов
        }
        
        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 400
    
    def test_get_features_list(self):
        """Тест получения списка признаков"""
        response = client.get("/features/list")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert "total_features" in data
        assert isinstance(data["features"], list)
    
    def test_predict_existing_customer_not_found(self):
        """Тест предсказания для несуществующего клиента"""
        # Предполагаем, что клиента с ID 99999 нет в базе
        response = client.get("/customer/99999/predict")
        assert response.status_code == 404
    
    def test_prediction_history_not_found(self):
        """Тест истории предсказаний для несуществующего клиента"""
        response = client.get("/customer/99999/history")
        assert response.status_code == 200  # Возвращает пустую историю
    
    def test_invalid_json(self):
        """Тест с невалидным JSON"""
        response = client.post("/predict", data="invalid json")
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Тест с отсутствующими обязательными полями"""
        incomplete_features = {"Tenure": 12.5}  # Только одно поле
        
        response = client.post("/predict", json=incomplete_features)
        assert response.status_code == 422  # Pydantic validation error

# Интеграционные тесты
class TestIntegration:
    """Интеграционные тесты"""
    
    def test_full_workflow(self):
        """Тест полного workflow"""
        # 1. Проверка здоровья
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Получение информации о модели
        info_response = client.get("/model/info")
        assert info_response.status_code == 200
        
        # 3. Одиночное предсказание
        predict_response = client.post("/predict", json=SAMPLE_FEATURES)
        assert predict_response.status_code == 200
        
        # 4. Пакетное предсказание
        batch_request = {
            "customers": [SAMPLE_FEATURES, SAMPLE_FEATURES],
            "customer_ids": [2001, 2002]
        }
        batch_response = client.post("/predict/batch", json=batch_request)
        assert batch_response.status_code == 200
        
        # 5. Получение списка признаков
        features_response = client.get("/features/list")
        assert features_response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])