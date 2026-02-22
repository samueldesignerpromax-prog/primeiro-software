import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from skimage import exposure, color

class SkinTransformer:
    def __init__(self):
        """Inicializa transformador de pele"""
        
        # Tons de pele em RGB e faixas HSV
        self.skin_tones = {
            "natural": {
                "rgb": [241, 194, 154],
                "hsv_range": ([0, 20, 70], [20, 150, 255])
            },
            "lighter": {
                "rgb": [255, 235, 210],
                "hsv_range": ([0, 10, 180], [20, 80, 255])
            },
            "darker": {
                "rgb": [160, 110, 70],
                "hsv_range": ([0, 40, 40], [20, 200, 140])
            },
            "tan": {
                "rgb": [222, 184, 135],
                "hsv_range": ([0, 30, 100], [25, 150, 220])
            },
            "porcelain": {
                "rgb": [255, 240, 230],
                "hsv_range": ([0, 5, 200], [20, 50, 255])
            },
            "olive": {
                "rgb": [200, 170, 120],
                "hsv_range": ([10, 30, 100], [30, 150, 200])
            },
            "honey": {
                "rgb": [238, 180, 120],
                "hsv_range": ([5, 40, 120], [25, 180, 240])
            },
            "caramel": {
                "rgb": [210, 150, 100],
                "hsv_range": ([0, 50, 80], [20, 200, 200])
            }
        }
        
        # Cores de olhos
        self.eye_colors = {
            "natural": [101, 67, 33],
            "blue": [0, 0, 255],
            "green": [0, 128, 0],
            "brown": [101, 67, 33],
            "hazel": [150, 120, 70]
        }
    
    def adjust_skin_tone(self, image: np.ndarray, landmarks: List, target_tone: str) -> np.ndarray:
        """
        Ajusta o tom de pele
        
        Args:
            image: Imagem BGR
            landmarks: Landmarks faciais
            target_tone: Tom desejado
            
        Returns:
            Imagem com tom de pele ajustado
        """
        result = image.copy()
        
        # Criar máscara para a pele
        skin_mask = self._create_skin_mask(image, landmarks)
        
        if target_tone in self.skin_tones:
            target_color = self.skin_tones[target_tone]["rgb"]
            
            # Aplicar correção de cor apenas na pele
            result = self._apply_skin_color(result, skin_mask, target_color)
        
        return result
    
    def smooth_skin(self, image: np.ndarray, landmarks: List, intensity: float = 0.7) -> np.ndarray:
        """
        Suaviza a pele (efeito de pele perfeita)
        
        Args:
            image: Imagem BGR
            landmarks: Landmarks faciais
            intensity: Intensidade da suavização (0-1)
            
        Returns:
            Imagem com pele suavizada
        """
        result = image.copy()
        
        # Criar máscara para a pele
        skin_mask = self._create_skin_mask(image, landmarks)
        
        # Aplicar bilateral filter para suavizar mantendo bordas
        smoothed = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Misturar imagem original com suavizada usando máscara
        mask_3channel = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR) / 255.0
        mask_3channel = mask_3channel * intensity
        
        result = (result * (1 - mask_3channel) + smoothed * mask_3channel).astype(np.uint8)
        
        return result
    
    def adjust_eye_color(self, image: np.ndarray, landmarks: List, color: str) -> np.ndarray:
        """
        Ajusta a cor dos olhos
        
        Args:
            image: Imagem BGR
            landmarks: Landmarks faciais
            color: Cor desejada
            
        Returns:
            Imagem com cor dos olhos ajustada
        """
        result = image.copy()
        
        if color not in self.eye_colors:
            return result
        
        # Extrair região dos olhos dos landmarks
        eye_regions = self._extract_eye_regions(image, landmarks)
        
        if eye_regions:
            target_color = self.eye_colors[color]
            
            for eye_region in eye_regions:
                x, y, w, h = eye_region
                if w > 0 and h > 0:
                    # Criar máscara circular para o olho
                    eye_mask = np.zeros((h, w), dtype=np.uint8)
                    center = (w // 2, h // 2)
                    radius = min(w, h) // 2
                    cv2.circle(eye_mask, center, radius, 255, -1)
                    
                    # Aplicar cor apenas na íris
                    eye_area = result[y:y+h, x:x+w]
                    
                    # Ajustar matiz
                    eye_hsv = cv2.cvtColor(eye_area, cv2.COLOR_BGR2HSV)
                    target_hsv = cv2.cvtColor(
                        np.uint8([[target_color]]), 
                        cv2.COLOR_RGB2HSV
                    )[0][0]
                    
                    # Aplicar nova cor apenas onde a máscara é > 0
                    mask_indices = eye_mask > 0
                    eye_hsv[:, :, 0][mask_indices] = target_hsv[0]
                    
                    eye_area_adjusted = cv2.cvtColor(eye_hsv, cv2.COLOR_HSV2BGR)
                    
                    # Combinar com máscara
                    eye_area = eye_area * (1 - eye_mask[:, :, np.newaxis]/255) + \
                              eye_area_adjusted * (eye_mask[:, :, np.newaxis]/255)
                    
                    result[y:y+h, x:x+w] = eye_area.astype(np.uint8)
        
        return result
    
    def preview_skin_tone(self, image: np.ndarray, face_info: Dict, tone: str) -> np.ndarray:
        """Preview rápido do tom de pele"""
        return self.adjust_skin_tone(image, face_info["landmarks"], tone)
    
    def preview_eye_color(self, image: np.ndarray, face_info: Dict, color: str) -> np.ndarray:
        """Preview rápido da cor dos olhos"""
        return self.adjust_eye_color(image, face_info["landmarks"], color)
    
    def _create_skin_mask(self, image: np.ndarray, landmarks: List) -> np.ndarray:
        """
        Cria máscara para a pele baseada nos landmarks e cor da pele
        """
        h, w = image.shape[:2]
        skin_mask = np.zeros((h, w), dtype=np.uint8)
        
        if landmarks:
            # Converter landmarks para array numpy
            points = np.array([(p[0], p[1]) for p in landmarks])
            
            # Criar máscara convex hull do rosto
            hull = cv2.convexHull(points)
            cv2.fillPoly(skin_mask, [hull], 255)
            
            # Remover olhos, boca, etc
            # (simplificado - em produção, usar segmentação mais precisa)
            
        return skin_mask
    
    def _apply_skin_color(self, image: np.ndarray, mask: np.ndarray, target_color: List) -> np.ndarray:
        """
        Aplica nova cor à pele mantendo textura
        """
        result = image.copy()
        
        # Converter para espaço de cores LAB para melhor transferência de cor
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Aplicar apenas onde a máscara > 0
        mask_indices = mask > 0
        
        if np.any(mask_indices):
            # Transferir cor mantendo luminância
            # (simplificado - em produção, usar técnicas mais avançadas)
            
            # Ajustar canais a e b
            target_lab = cv2.cvtColor(
                np.uint8([[target_color]]), 
                cv2.COLOR_RGB2LAB
            )[0][0]
            
            # Preservar luminância original (canal L)
            # Ajustar canais de cor (a e b) em direção ao alvo
            alpha = 0.7  # intensidade da transferência
            
            lab[:, :, 1][mask_indices] = (lab[:, :, 1][mask_indices] * (1 - alpha) + 
                                          target_lab[1] * alpha).astype(np.uint8)
            lab[:, :, 2][mask_indices] = (lab[:, :, 2][mask_indices] * (1 - alpha) + 
                                          target_lab[2] * alpha).astype(np.uint8)
            
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _extract_eye_regions(self, image: np.ndarray, landmarks: List) -> List:
        """Extrai regiões dos olhos dos landmarks"""
        eye_regions = []
        h, w = image.shape[:2]
        
        # Índices aproximados para olhos
        left_eye_indices = list(range(36, 42))  # Pontos do olho esquerdo
        right_eye_indices = list(range(42, 48))  # Pontos do olho direito
        
        for indices in [left_eye_indices, right_eye_indices]:
            eye_points = []
            for idx in indices:
                if idx < len(landmarks):
                    x, y, _ = landmarks[idx]
                    eye_points.append([x, y])
            
            if eye_points:
                eye_points = np.array(eye_points)
                x_min = max(0, np.min(eye_points[:, 0]) - 5)
                y_min = max(0, np.min(eye_points[:, 1]) - 5)
                x_max = min(w, np.max(eye_points[:, 0]) + 5)
                y_max = min(h, np.max(eye_points[:, 1]) + 5)
                
                eye_regions.append([int(x_min), int(y_min), 
                                   int(x_max - x_min), int(y_max - y_min)])
        
        return eye_regions
    
    def get_available_tones(self) -> List[str]:
        """Retorna lista de tons disponíveis"""
        return list(self.skin_tones.keys())
    
    def get_available_eye_colors(self) -> List[str]:
        """Retorna lista de cores de olho disponíveis"""
        return list(self.eye_colors.keys())
    
    def is_ready(self) -> bool:
        return True
