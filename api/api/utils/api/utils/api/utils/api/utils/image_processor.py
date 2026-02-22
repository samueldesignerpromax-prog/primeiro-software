import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, Optional, Union

class ImageProcessor:
    def __init__(self):
        """Processador geral de imagens"""
        pass
    
    def resize_image(self, image: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """
        Redimensiona imagem mantendo proporção
        """
        h, w = image.shape[:2]
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        return image
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Melhora qualidade da imagem (contraste, nitidez)
        """
        # Ajustar contraste
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Aumentar nitidez
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def bytes_to_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Converte bytes para imagem OpenCV
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    
    def image_to_bytes(self, image: np.ndarray, format: str = '.jpg') -> bytes:
        """
        Converte imagem OpenCV para bytes
        """
        _, buffer = cv2.imencode(format, image)
        return buffer.tobytes()
    
    def blend_images(self, img1: np.ndarray, img2: np.ndarray, 
                     mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Combina duas imagens usando máscara
        """
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        mask = mask.astype(np.float32) / 255.0
        result = (img1 * (1 - mask) + img2 * mask).astype(np.uint8)
        
        return result
    
    def get_image_info(self, image: np.ndarray) -> dict:
        """
        Retorna informações da imagem
        """
        h, w = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        
        return {
            "width": w,
            "height": h,
            "channels": channels,
            "dtype": str(image.dtype),
            "size_bytes": image.nbytes
        }
