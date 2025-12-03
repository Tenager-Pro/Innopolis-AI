import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from .models import CustomerFeatures, ChurnPrediction

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnPredictor:
    """Класс для предсказания оттока клиентов"""
    
    def __init__(self, model_path: str = "models/final/churn_model.pkl", 
                 feature_names_path: str = "models/final/feature_names.pkl"):
        self.model = None
        self.feature_names = None
        self.model_version = "1.0.0"
        self.is_loaded = False
        self.model_path = model_path
        self.feature_names_path = feature_names_path
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Загрузка модели и feature names"""
        try:
            self.model = joblib.load(self.model_path)
            self.feature_names = joblib.load(self.feature_names_path)
            self.is_loaded = True
            logger.info(f"✅ Модель успешно загружена. Признаки: {len(self.feature_names)}")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            self.is_loaded = False
            raise
    
    def _preprocess_features(self, features: CustomerFeatures) -> pd.DataFrame:
        """Предобработка признаков для модели"""
        # Конвертируем в словарь
        features_dict = features.dict()
        
        # Создаем DataFrame с правильным порядком признаков
        input_data = {}
        for feature in self.feature_names:
            if feature in features_dict:
                input_data[feature] = [features_dict[feature]]
            else:
                # Если признак отсутствует, используем среднее значение
                logger.warning(f"Признак {feature} отсутствует в входных данных")
                input_data[feature] = [0.0]  # Заглушка
        
        return pd.DataFrame(input_data)
    
    def _calculate_additional_features(self, features: CustomerFeatures) -> CustomerFeatures:
        """Вычисление дополнительных признаков если они не предоставлены"""
        features_dict = features.dict()
        
        # Вычисляем AvgOrdersPerMonth если не предоставлен
        if features_dict.get('AvgOrdersPerMonth') is None and features_dict.get('Tenure') > 0:
            features_dict['AvgOrdersPerMonth'] = features_dict.get('OrderCount', 0) / features_dict['Tenure']
        
        # Вычисляем AvgCashbackPerOrder если не предоставлен
        if features_dict.get('AvgCashbackPerOrder') is None and features_dict.get('OrderCount') > 0:
            features_dict['AvgCashbackPerOrder'] = features_dict.get('CashbackAmount', 0) / features_dict['OrderCount']
        
        # Вычисляем EngagementScore если не предоставлен
        if features_dict.get('EngagementScore') is None:
            features_dict['EngagementScore'] = features_dict.get('HourSpendOnApp', 0) * features_dict.get('OrderCount', 0)
        
        # Вычисляем SatisfactionComplainRatio если не предоставлен
        if features_dict.get('SatisfactionComplainRatio') is None:
            complain = features_dict.get('Complain', 0)
            features_dict['SatisfactionComplainRatio'] = (
                features_dict.get('SatisfactionScore', 0) / (complain + 1)
            )
        
        return CustomerFeatures(**features_dict)
    
    def predict_single(self, features: CustomerFeatures, customer_id: int = None, 
                      threshold: float = 0.5) -> ChurnPrediction:
        """Предсказание для одного клиента"""
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена")
        
        try:
            # Вычисляем дополнительные признаки
            processed_features = self._calculate_additional_features(features)
            
            # Преобразуем в DataFrame
            input_df = self._preprocess_features(processed_features)
            
            # Предсказание
            churn_probability = self.model.predict_proba(input_df)[0, 1]
            prediction = churn_probability > threshold
            
            return ChurnPrediction(
                customer_id=customer_id,
                churn_probability=round(churn_probability, 4),
                prediction=bool(prediction),
                model_version=self.model_version
            )
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            raise
    
    def predict_batch(self, customers: List[CustomerFeatures], 
                     customer_ids: List[int] = None, 
                     threshold: float = 0.5) -> Tuple[List[ChurnPrediction], float]:
        """Пакетное предсказание для нескольких клиентов"""
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена")
        
        predictions = []
        
        for i, customer_features in enumerate(customers):
            customer_id = customer_ids[i] if customer_ids and i < len(customer_ids) else None
            
            prediction = self.predict_single(customer_features, customer_id, threshold)
            predictions.append(prediction)
        
        # Вычисляем общий процент оттока
        churn_rate = sum(1 for p in predictions if p.prediction) / len(predictions)
        
        return predictions, churn_rate
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена")
        
        try:
            # Загружаем model card если существует
            import json
            with open("models/final/model_card.json", "r") as f:
                model_card = json.load(f)
        except:
            model_card = {
                "model_name": "Customer Churn Predictor",
                "model_type": type(self.model).__name__,
                "performance": {},
                "feature_count": len(self.feature_names),
                "training_date": "unknown"
            }
        
        return {
            "model_name": model_card.get("model_name", "Customer Churn Predictor"),
            "model_version": self.model_version,
            "model_type": model_card.get("model_type", type(self.model).__name__),
            "performance": model_card.get("performance", {}),
            "feature_count": len(self.feature_names),
            "training_date": model_card.get("training_date", "unknown"),
            "feature_names": self.feature_names,
            "is_loaded": self.is_loaded
        }

# Глобальный экземпляр предсказателя
predictor = ChurnPredictor()