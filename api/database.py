from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Менеджер для работы с базой данных"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = None
        self.is_connected = False
        self._connect()
    
    def _connect(self) -> None:
        """Подключение к базе данных"""
        try:
            self.engine = create_engine(self.connection_string)
            # Тестовое подключение
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.is_connected = True
            logger.info("✅ Успешное подключение к базе данных")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к базе данных: {e}")
            self.is_connected = False
    
    def save_prediction(self, customer_id: int, features: Dict, 
                       prediction: Dict, table_name: str = "churn_predictions") -> bool:
        """Сохранение предсказания в базу данных"""
        if not self.is_connected:
            logger.warning("База данных не подключена, пропускаем сохранение")
            return False
        
        try:
            # Создаем таблицу если не существует
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                customer_id INTEGER,
                features JSONB,
                churn_probability FLOAT,
                prediction BOOLEAN,
                model_version VARCHAR(50),
                prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(create_table_query))
                
                # Вставляем предсказание
                insert_query = f"""
                INSERT INTO {table_name} 
                (customer_id, features, churn_probability, prediction, model_version)
                VALUES (:customer_id, :features, :churn_probability, :prediction, :model_version)
                """
                
                conn.execute(text(insert_query), {
                    'customer_id': customer_id,
                    'features': str(features),
                    'churn_probability': prediction['churn_probability'],
                    'prediction': prediction['prediction'],
                    'model_version': prediction['model_version']
                })
                conn.commit()
            
            logger.info(f"✅ Предсказание для клиента {customer_id} сохранено в БД")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Ошибка сохранения в БД: {e}")
            return False
    
    def get_customer_features(self, customer_id: int) -> Optional[Dict]:
        """Получение признаков клиента из базы данных"""
        if not self.is_connected:
            return None
        
        try:
            query = """
            SELECT * FROM features_layer 
            WHERE CustomerID = :customer_id
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'customer_id': customer_id})
                row = result.fetchone()
                
                if row:
                    return dict(row._mapping)
                else:
                    return None
                    
        except SQLAlchemyError as e:
            logger.error(f"Ошибка получения данных клиента: {e}")
            return None
    
    def get_predictions_history(self, customer_id: int, limit: int = 10) -> List[Dict]:
        """История предсказаний для клиента"""
        if not self.is_connected:
            return []
        
        try:
            query = """
            SELECT * FROM churn_predictions 
            WHERE customer_id = :customer_id 
            ORDER BY prediction_timestamp DESC 
            LIMIT :limit
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {
                    'customer_id': customer_id, 
                    'limit': limit
                })
                return [dict(row._mapping) for row in result]
                
        except SQLAlchemyError as e:
            logger.error(f"Ошибка получения истории предсказаний: {e}")
            return []

# Глобальный экземпляр менеджера БД
# В реальном приложении connection_string должен быть в конфигурации
db_manager = DatabaseManager("postgresql://postgres:password@localhost:5432/customer_analysis_db")
