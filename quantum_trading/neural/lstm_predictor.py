"""
LSTM Neural Predictor para QUALIA
Módulo para previsão do momento ótimo de entrada em operações usando redes neurais LSTM
"""

import numpy as np
import pandas as pd
import logging
import os
import time
from datetime import datetime, timedelta
import joblib
from typing import Dict, List, Any, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger("lstm_predictor")

class LSTMPredictor:
    """
    Preditor Neural usando LSTM para identificar momentos ótimos de entrada
    em operações de arbitragem, baseado em padrões históricos.
    """
    
    def __init__(self, 
                model_path: str = "models/lstm_predictor.h5",
                lookback_periods: int = 10,
                prediction_threshold: float = 0.7,
                features: List[str] = None,
                batch_size: int = 32,
                epochs: int = 100):
        """
        Inicializa o preditor LSTM
        
        Args:
            model_path: Caminho para salvar/carregar o modelo
            lookback_periods: Número de períodos anteriores para considerar
            prediction_threshold: Limiar para considerar uma previsão como positiva
            features: Lista de features a serem usadas para treinamento
            batch_size: Tamanho do batch para treinamento
            epochs: Número máximo de épocas para treinamento
        """
        self.model_path = model_path
        self.lookback_periods = lookback_periods
        self.prediction_threshold = prediction_threshold
        self.model = None
        self.scaler = None
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Definir features padrão se não especificado
        self.features = features or [
            'spread', 'volume_a', 'volume_b', 'entropy', 
            'fractal_dimension', 'volatility_a', 'volatility_b'
        ]
        
        # Status do modelo
        self.is_trained = False
        self.last_training_time = None
        self.performance_metrics = {}
        
        # Tentativa de carregar modelo existente
        self._try_load_model()
        
        logger.info("LSTM Predictor inicializado com período de análise de "
                   f"{lookback_periods} intervalos e limiar de {prediction_threshold}")
    
    def _try_load_model(self) -> bool:
        """
        Tenta carregar um modelo pré-treinado
        
        Returns:
            True se carregou com sucesso, False caso contrário
        """
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                
                # Carregar scaler se existir
                scaler_path = self.model_path.replace('.h5', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    
                self.is_trained = True
                logger.info(f"Modelo LSTM carregado de {self.model_path}")
                return True
            else:
                logger.info("Nenhum modelo LSTM pré-treinado encontrado")
                return False
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def _create_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Cria a arquitetura do modelo LSTM
        
        Args:
            input_shape: Formato dos dados de entrada (sequências, features)
            
        Returns:
            Modelo Keras configurado
        """
        model = Sequential()
        
        # Primeira camada LSTM com retorno de sequências
        model.add(LSTM(128, return_sequences=True, 
                      input_shape=input_shape,
                      dropout=0.2, recurrent_dropout=0.2))
        model.add(BatchNormalization())
        
        # Segunda camada LSTM
        model.add(LSTM(64, return_sequences=False, dropout=0.2))
        model.add(BatchNormalization())
        
        # Camadas densas
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        
        # Camada de saída - probabilidade de entrada ótima
        model.add(Dense(1, activation='sigmoid'))
        
        # Compilar modelo
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.AUC()])
        
        return model
    
    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara sequências de dados para o LSTM
        
        Args:
            data: DataFrame com os dados
            
        Returns:
            X: Sequências de features
            y: Targets (1 para momento ótimo, 0 caso contrário)
        """
        # Selecionar features relevantes
        features_df = data[self.features].copy()
        
        # Normalizar dados
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            normalized_data = self.scaler.fit_transform(features_df)
        else:
            normalized_data = self.scaler.transform(features_df)
        
        # Target: 1 se a próxima operação foi lucrativa acima do threshold
        if 'profitable' in data.columns:
            targets = data['profitable'].values
        else:
            # Se não houver coluna explícita, considerar operações com lucro > 0
            if 'profit' in data.columns:
                targets = (data['profit'] > 0).astype(int).values
            else:
                logger.warning("Nenhuma coluna 'profitable' ou 'profit' encontrada. "
                              "Usando targets aleatórios para teste.")
                targets = np.random.randint(0, 2, size=len(data))
        
        # Criar sequências
        X, y = [], []
        for i in range(len(normalized_data) - self.lookback_periods):
            X.append(normalized_data[i:i + self.lookback_periods])
            y.append(targets[i + self.lookback_periods])
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Treina o modelo LSTM com dados históricos
        
        Args:
            data: DataFrame com histórico de oportunidades e operações
            validation_split: Proporção dos dados para validação
            
        Returns:
            Dicionário com métricas de performance do treinamento
        """
        logger.info(f"Iniciando treinamento do modelo LSTM com {len(data)} registros")
        
        # Preparar dados
        X, y = self._prepare_sequences(data)
        logger.info(f"Preparadas {len(X)} sequências de treinamento")
        
        # Split treino/validação
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=True
        )
        
        # Criar diretório para o modelo se não existir
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Checkpoint para salvar o melhor modelo
        checkpoint = ModelCheckpoint(
            self.model_path, 
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        # Early stopping para evitar overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Criar modelo
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self._create_model(input_shape)
        
        # Treinar modelo
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Calcular métricas no conjunto de validação
        val_loss, val_accuracy, val_auc = self.model.evaluate(X_val, y_val, verbose=0)
        
        # Salvar scaler
        scaler_path = self.model_path.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Atualizar status
        self.is_trained = True
        self.last_training_time = datetime.now()
        
        # Armazenar métricas
        self.performance_metrics = {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_auc': val_auc,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss']),
            'training_date': self.last_training_time.isoformat(),
            'model_path': self.model_path,
            'lookback_periods': self.lookback_periods,
            'num_features': len(self.features),
            'sample_size': len(X)
        }
        
        logger.info(f"Treinamento concluído em {training_time:.2f}s. "
                   f"Acurácia: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
        
        return self.performance_metrics
    
    def predict(self, 
               current_data: pd.DataFrame, 
               return_probability: bool = False) -> Union[bool, float]:
        """
        Prediz se o momento atual é ótimo para entrar em uma operação
        
        Args:
            current_data: DataFrame com os dados atuais e históricos recentes
            return_probability: Se True, retorna a probabilidade ao invés de bool
            
        Returns:
            True se for momento ótimo, False caso contrário
            ou valor de probabilidade se return_probability=True
        """
        if not self.is_trained or self.model is None:
            logger.warning("Modelo não treinado. Impossível realizar previsões.")
            return False if not return_probability else 0.0
        
        if len(current_data) < self.lookback_periods:
            logger.warning(f"Dados insuficientes para previsão. "
                          f"Necessário pelo menos {self.lookback_periods} registros.")
            return False if not return_probability else 0.0
        
        try:
            # Selecionar apenas as features usadas no modelo
            features_df = current_data[self.features].tail(self.lookback_periods).copy()
            
            # Normalizar dados
            normalized_data = self.scaler.transform(features_df)
            
            # Converter para o formato de entrada do modelo
            model_input = np.array([normalized_data])
            
            # Realizar previsão
            probability = self.model.predict(model_input, verbose=0)[0][0]
            
            if return_probability:
                return float(probability)
            else:
                return probability >= self.prediction_threshold
                
        except Exception as e:
            logger.error(f"Erro ao fazer previsão: {e}")
            return False if not return_probability else 0.0
    
    def predict_optimal_entry(self, 
                           market_data: pd.DataFrame,
                           min_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Analisa os dados de mercado e identifica o momento ótimo de entrada
        
        Args:
            market_data: DataFrame com dados de mercado recentes
            min_confidence: Confiança mínima para recomendar entrada
            
        Returns:
            Dicionário com resultado da análise e recomendação
        """
        if not self.is_trained or self.model is None:
            return {
                'should_enter': False,
                'confidence': 0.0,
                'reason': 'Modelo não treinado'
            }
        
        if len(market_data) < self.lookback_periods:
            return {
                'should_enter': False,
                'confidence': 0.0,
                'reason': f'Dados insuficientes. Necessário {self.lookback_periods} registros.'
            }
        
        # Obter previsão como probabilidade
        confidence = self.predict(market_data, return_probability=True)
        
        # Decidir com base na confiança e threshold
        should_enter = confidence >= max(self.prediction_threshold, min_confidence)
        
        # Preparar análise de features principais para justificar a decisão
        if should_enter and len(market_data) > 0:
            # Analisar tendências recentes das principais features
            last_record = market_data.iloc[-1]
            feature_analysis = {}
            
            for feature in self.features:
                if feature in last_record:
                    feature_analysis[feature] = last_record[feature]
            
            # Classificar a decisão
            if confidence >= 0.9:
                confidence_level = "muito alta"
            elif confidence >= 0.8:
                confidence_level = "alta"
            elif confidence >= 0.7:
                confidence_level = "moderada"
            else:
                confidence_level = "baixa"
                
            result = {
                'should_enter': should_enter,
                'confidence': float(confidence),
                'confidence_level': confidence_level,
                'reason': f'Confiança {confidence_level} ({confidence:.2%}) baseada na análise de padrões históricos',
                'features': feature_analysis
            }
        else:
            result = {
                'should_enter': should_enter,
                'confidence': float(confidence),
                'reason': f'Confiança insuficiente ({confidence:.2%})'
            }
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo atual
        
        Returns:
            Dicionário com informações do modelo
        """
        info = {
            'is_trained': self.is_trained,
            'model_path': self.model_path,
            'lookback_periods': self.lookback_periods,
            'prediction_threshold': self.prediction_threshold,
            'features': self.features,
            'last_training_time': self.last_training_time.isoformat() 
                                  if self.last_training_time else None
        }
        
        # Adicionar métricas se disponíveis
        if self.performance_metrics:
            info['performance'] = self.performance_metrics
            
        # Adicionar summary do modelo se estiver treinado
        if self.is_trained and self.model is not None:
            model_summary = []
            self.model.summary(print_fn=lambda x: model_summary.append(x))
            info['model_summary'] = '\n'.join(model_summary)
            
            # Adicionar número de parâmetros
            info['total_params'] = self.model.count_params()
        
        return info 