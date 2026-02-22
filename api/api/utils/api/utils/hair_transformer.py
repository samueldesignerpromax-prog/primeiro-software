import cv2
import numpy as np
from typing import Dict, Any, Optional, List
import colorsys

class HairTransformer:
    def __init__(self):
        """Inicializa transformador de cabelo"""
        # Mapeamento de cores em RGB e HSV
        self.hair_colors = {
            "natural": {
                "rgb": [101, 67, 33],  # Marrom médio
                "hsv_range": ([0, 0, 0], [180, 255, 150])
            },
            "blonde": {
                "rgb": [255, 235, 170],  # Loiro
                "hsv_range": ([20, 50, 150], [35, 255, 255])
            },
            "brown": {
                "rgb": [139, 69, 19],  # Marrom
                "hsv_range": ([0, 50, 50], [20, 255, 150])
            },
            "black": {
                "rgb": [30, 30, 30],  # Preto
                "hsv_range": ([0, 0, 0], [180, 255, 60])
            },
            "red": {
                "rgb": [255, 60, 60],  # Vermelho
                "hsv_range": ([0, 100, 100], [10, 255, 255])
            },
            "blue": {
                "rgb": [70, 130, 255],  # Azul
                "hsv_range": ([100, 100, 100], [130, 255, 255])
            },
            "pink": {
                "rgb": [255, 182, 193],  # Rosa
                "hsv_range": ([160, 50, 150], [180, 255, 255])
            },
            "purple": {
                "rgb": [128, 0, 128],  # Roxo
                "hsv_range": ([130, 100, 50], [160, 255, 255])
            }
        }
        
        self.hair_styles = {
            "natural": {},
            "curly": {"curl_intensity": 0.7},
            "straight": {"smooth_factor": 0.8},
            "wavy": {"wave_intensity": 0.5},
            "volume": {"volume_factor": 1.3}
        }
    
    def transform_hair(self, image: np.ndarray, face_info: Dict, 
                       color: str = "natural", style: Optional[str] = None) -> np.ndarray:
        """
        Aplica transformação no cabelo
        
        Args:
            image: Imagem BGR
            face_info: Informações do rosto
            color: Cor desejada
            style: Estilo desejado
            
        Returns:
            Imagem transformada
        """
        result = image.copy()
        
        # Obter região do cabelo
        hair_region = self._get_hair_mask(image, face_info)
        
        if hair_region is None or np.sum(hair_region) == 0:
            return result
        
        # Aplicar cor ao cabelo
        if color != "natural":
            result = self._apply_hair_color(result, hair_region, color)
        
        # Aplicar estilo
        if style and style in self.hair_styles:
            result = self._apply_hair_style(result, hair_region, style)
        
        return result
    
    def preview_hair_color(self, image: np.ndarray, face_info: Dict, color: str) -> np.ndarray:
        """Preview rápido da cor do cabelo"""
        return self.transform_hair(image, face_info, color, None)
    
    def _get_hair_mask(self, image: np.ndarray, face_info: Dict) -> np.ndarray:
        """
        Cria máscara para o cabelo usando técnicas de segmentação
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Obter região estimada do cabelo
        if "regions" in face_info:
            # Usar região da testa como referência
            if "face_oval" in face_info["regions"]:
                face_bbox = face_info["bbox"]
                x, y, width, height = face_bbox
                
                # Região acima da testa
                hair_top = max(0, y - int(height * 0.6))
                hair_bottom = y + int(height * 0.1)
                hair_left = max(0, x - int(width * 0.2))
                hair_right = min(w, x + width + int(width * 0.2))
                
                # Criar máscara gradiente
                for i in range(hair_top, hair_bottom):
                    for j in range(hair_left, hair_right):
                        # Distância do rosto
                        dist_to_face = max(0, (hair_bottom - i) / (hair_bottom - hair_top))
                        mask[i, j] = int(255 * dist_to_face)
        
        return mask
    
    def _apply_hair_color(self, image: np.ndarray, mask: np.ndarray, color: str) -> np.ndarray:
        """
        Aplica nova cor ao cabelo
        """
        result = image.copy()
        
        if color not in self.hair_colors:
            return result
        
        target_color = self.hair_colors[color]["rgb"]
        
        # Converter para HSV para melhor manipulação de cor
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Aplicar cor apenas onde a máscara > 0
        mask_indices = mask > 0
        
        if np.any(mask_indices):
            # Preservar luminosidade original
            original_value = hsv[:, :, 2][mask_indices]
            
            # Nova cor em HSV
            target_hsv = cv2.cvtColor(
                np.uint8([[target_color]]), 
                cv2.COLOR_RGB2HSV
            )[0][0]
            
            # Aplicar novo matiz mantendo saturação e valor originais
            hsv[:, :, 0][mask_indices] = target_hsv[0]
            
            # Ajustar saturação (misturar com original)
            hsv[:, :, 1][mask_indices] = np.clip(
                hsv[:, :, 1][mask_indices] * 1.2, 0, 255
            ).astype(np.uint8)
            
            # Converter de volta para BGR
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result
    
    def _apply_hair_style(self, image: np.ndarray, mask: np.ndarray, style: str) -> np.ndarray:
        """
        Aplica estilo ao cabelo (alisar, encaracolar, etc)
        """
        result = image.copy()
        
        if style == "straight":
            # Suavizar textura do cabelo
            kernel_size = (5, 5)
            result = cv2.GaussianBlur(result, kernel_size, 0)
            
            # Aplicar apenas na região do cabelo
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = (result * mask_3channel + image * (1 - mask_3channel)).astype(np.uint8)
        
        elif style == "volume":
            # Aumentar volume (simulado com brilho)
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            mask_indices = mask > 0
            
            if np.any(mask_indices):
                hsv[:, :, 2][mask_indices] = np.clip(
                    hsv[:, :, 2][mask_indices] * 1.2, 0, 255
                ).astype(np.uint8)
            
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result
    
    def get_available_colors(self) -> List[str]:
        """Retorna lista de cores disponíveis"""
        return list(self.hair_colors.keys())
    
    def get_available_styles(self) -> List[str]:
        """Retorna lista de estilos disponíveis"""
        return list(self.hair_styles.keys())
    
    def is_ready(self) -> bool:
        return True
