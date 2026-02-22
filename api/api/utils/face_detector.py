import cv2
import numpy as np
import mediapipe as mp
import dlib
from typing import List, Dict, Any, Optional

class FaceDetector:
    def __init__(self):
        """Inicializa detectores de rosto e landmarks"""
        # MediaPipe para detecção rápida
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configurações
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            min_detection_confidence=0.5
        )
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 para curta distância, 1 para longa
            min_detection_confidence=0.5
        )
        
        # Mapeamento de índices para partes do rosto
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
                     402, 317, 14, 87, 178, 88, 95]
        
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
                         160, 161, 246]
        
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
                          386, 385, 384, 398]
        
        self.LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
        self.RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
        
        self.NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 99, 100, 101, 102,
                     103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
                     116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
                     129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
                     142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
                     155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
                     168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
                     181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
                     194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
                     207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
                     220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232,
                     233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
                     246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258,
                     259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
                     272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284,
                     285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297,
                     298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310,
                     311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,
                     324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336,
                     337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,
                     350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362,
                     363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375,
                     376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388,
                     389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401,
                     402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414,
                     415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427,
                     428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440,
                     441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453,
                     454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466,
                     467, 468]
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta rostos e landmarks faciais
        
        Args:
            image: Imagem em formato BGR (OpenCV)
            
        Returns:
            Lista de dicionários com informações de cada rosto
        """
        # Converter BGR para RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Detectar faces com MediaPipe
        results = self.face_mesh.process(image_rgb)
        
        faces = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extrair landmarks
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = landmark.z
                    landmarks.append((x, y, z))
                
                # Calcular bounding box
                xs = [p[0] for p in landmarks]
                ys = [p[1] for p in landmarks]
                
                x_min = max(0, min(xs))
                y_min = max(0, min(ys))
                x_max = min(w, max(xs))
                y_max = min(h, max(ys))
                
                # Extrair regiões específicas
                face_regions = self._extract_face_regions(landmarks, w, h)
                
                faces.append({
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "landmarks": landmarks,
                    "regions": face_regions,
                    "confidence": 1.0,
                    "center": [int((x_min + x_max) / 2), int((y_min + y_max) / 2)]
                })
        
        # Se não detectou com MediaPipe, tenta com detecção básica
        if not faces:
            detection_results = self.face_detection.process(image_rgb)
            if detection_results.detections:
                for detection in detection_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x_min = int(bboxC.xmin * w)
                    y_min = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)
                    
                    faces.append({
                        "bbox": [x_min, y_min, width, height],
                        "landmarks": [],
                        "regions": {},
                        "confidence": detection.score[0],
                        "center": [x_min + width//2, y_min + height//2]
                    })
        
        return faces
    
    def _extract_face_regions(self, landmarks: List, img_w: int, img_h: int) -> Dict:
        """Extrai regiões específicas do rosto"""
        regions = {}
        
        # Mapeamento de regiões
        region_indices = {
            "face_oval": self.FACE_OVAL,
            "lips": self.LIPS,
            "left_eye": self.LEFT_EYE,
            "right_eye": self.RIGHT_EYE,
            "left_eyebrow": self.LEFT_EYEBROW,
            "right_eyebrow": self.RIGHT_EYEBROW,
            "nose": self.NOSE
        }
        
        for region_name, indices in region_indices.items():
            region_points = []
            for idx in indices:
                if idx < len(landmarks):
                    x, y, z = landmarks[idx]
                    region_points.append([x, y])
            
            if region_points:
                region_points = np.array(region_points)
                x_min = max(0, np.min(region_points[:, 0]))
                y_min = max(0, np.min(region_points[:, 1]))
                x_max = min(img_w, np.max(region_points[:, 0]))
                y_max = min(img_h, np.max(region_points[:, 1]))
                
                regions[region_name] = {
                    "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                    "points": region_points.tolist()
                }
        
        return regions
    
    def get_hair_region(self, face_info: Dict, img_shape: tuple) -> np.ndarray:
        """
        Estima a região do cabelo baseado na posição do rosto
        """
        h, w = img_shape[:2]
        bbox = face_info["bbox"]
        x, y, width, height = bbox
        
        # Região acima do rosto (cabelo)
        hair_x = max(0, x - int(width * 0.2))
        hair_y = max(0, y - int(height * 0.6))
        hair_width = min(w - hair_x, int(width * 1.4))
        hair_height = min(y - hair_y, int(height * 0.5))
        
        return np.array([hair_x, hair_y, hair_width, hair_height])
    
    def get_forehead_region(self, face_info: Dict, img_shape: tuple) -> np.ndarray:
        """Extrai região da testa"""
        bbox = face_info["bbox"]
        x, y, width, height = bbox
        
        forehead_x = x
        forehead_y = y
        forehead_width = width
        forehead_height = int(height * 0.25)
        
        return np.array([forehead_x, forehead_y, forehead_width, forehead_height])
    
    def is_ready(self) -> bool:
        """Verifica se o detector está pronto"""
        return True
