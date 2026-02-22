from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile
from typing import Optional
import base64
import uuid
from mangum import Mangum

# Importando módulos utilitários
from api.utils.face_detector import FaceDetector
from api.utils.hair_transformer import HairTransformer
from api.utils.skin_transformer import SkinTransformer
from api.utils.image_processor import ImageProcessor

# Inicializando componentes
app = FastAPI(
    title="API de Transformação com IA",
    description="Transforme fotos com IA - Modifique cabelo, pele e muito mais",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar processadores
face_detector = FaceDetector()
hair_transformer = HairTransformer()
skin_transformer = SkinTransformer()
image_processor = ImageProcessor()

# Criar diretório temporário
TEMP_DIR = "/tmp" if os.path.exists("/tmp") else "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {
        "message": "🚀 API de Transformação com IA",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "transform": "/transform",
            "detect": "/detect-face",
            "preview": "/preview-transform",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "face_detector": face_detector.is_ready(),
            "hair_transformer": hair_transformer.is_ready(),
            "skin_transformer": skin_transformer.is_ready()
        }
    }

@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):
    """
    Detecta rostos na imagem enviada
    """
    try:
        # Ler imagem
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # Converter RGB para BGR se necessário
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Detectar rostos
        faces = face_detector.detect_faces(image_np)
        
        # Retornar informações
        return JSONResponse({
            "success": True,
            "faces_detected": len(faces),
            "face_locations": [
                {
                    "x": int(f["bbox"][0]),
                    "y": int(f["bbox"][1]),
                    "width": int(f["bbox"][2]),
                    "height": int(f["bbox"][3]),
                    "confidence": float(f["confidence"])
                } for f in faces
            ]
        })
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/transform")
async def transform_image(
    file: UploadFile = File(...),
    hair_color: str = Form("natural"),
    skin_tone: str = Form("natural"),
    hair_style: Optional[str] = Form(None),
    eye_color: Optional[str] = Form(None),
    smooth_skin: bool = Form(False),
    return_base64: bool = Form(False)
):
    """
    Aplica transformações na imagem usando IA
    
    Opções de hair_color: natural, blonde, brown, black, red, blue, pink, purple
    Opções de skin_tone: natural, lighter, darker, tan, porcelain, olive, honey, caramel
    Opções de hair_style: curly, straight, wavy, volume (opcional)
    Opções de eye_color: natural, blue, green, brown, hazel (opcional)
    """
    try:
        # Validar entrada
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
        
        # Ler imagem
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # Converter para formato OpenCV (BGR)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
        
        # 1. Detectar rosto e landmarks
        faces = face_detector.detect_faces(image_bgr)
        
        if not faces:
            raise HTTPException(status_code=400, detail="Nenhum rosto detectado na imagem")
        
        # Pegar o primeiro rosto detectado
        face = faces[0]
        
        # 2. Aplicar transformações
        transformed_image = image_bgr.copy()
        
        # Modificar tom de pele
        if skin_tone != "natural":
            transformed_image = skin_transformer.adjust_skin_tone(
                transformed_image, 
                face["landmarks"], 
                skin_tone
            )
        
        # Suavizar pele
        if smooth_skin:
            transformed_image = skin_transformer.smooth_skin(
                transformed_image, 
                face["landmarks"]
            )
        
        # Modificar cor do cabelo
        if hair_color != "natural" or hair_style:
            transformed_image = hair_transformer.transform_hair(
                transformed_image,
                face,
                color=hair_color,
                style=hair_style
            )
        
        # Modificar cor dos olhos
        if eye_color and eye_color != "natural":
            transformed_image = skin_transformer.adjust_eye_color(
                transformed_image,
                face["landmarks"],
                eye_color
            )
        
        # Converter de volta para RGB
        transformed_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        
        # Salvar resultado temporário
        output_filename = f"transformed_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        # Salvar imagem
        cv2.imwrite(output_path, transformed_image)
        
        if return_base64:
            # Converter para base64
            _, buffer = cv2.imencode('.jpg', transformed_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return JSONResponse({
                "success": True,
                "image_base64": f"data:image/jpeg;base64,{img_base64}",
                "transformations": {
                    "hair_color": hair_color,
                    "skin_tone": skin_tone,
                    "hair_style": hair_style,
                    "eye_color": eye_color,
                    "smooth_skin": smooth_skin
                }
            })
        else:
            # Retornar URL temporária
            return JSONResponse({
                "success": True,
                "image_url": f"/api/temp/{output_filename}",
                "transformations": {
                    "hair_color": hair_color,
                    "skin_tone": skin_tone,
                    "hair_style": hair_style,
                    "eye_color": eye_color,
                    "smooth_skin": smooth_skin
                }
            })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preview-transform")
async def preview_transform(
    file: UploadFile = File(...),
    transform_type: str = Form("hair"),
    value: str = Form(...)
):
    """
    Preview de transformação específica
    transform_type: hair, skin, eyes
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # Converter para BGR
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
        
        # Detectar rosto
        faces = face_detector.detect_faces(image_bgr)
        
        if not faces:
            raise HTTPException(status_code=400, detail="Nenhum rosto detectado")
        
        face = faces[0]
        preview_image = image_bgr.copy()
        
        # Aplicar transformação específica
        if transform_type == "hair":
            preview_image = hair_transformer.preview_hair_color(preview_image, face, value)
        elif transform_type == "skin":
            preview_image = skin_transformer.preview_skin_tone(preview_image, face, value)
        elif transform_type == "eyes":
            preview_image = skin_transformer.preview_eye_color(preview_image, face, value)
        else:
            raise HTTPException(status_code=400, detail="Tipo de transformação inválido")
        
        # Converter para base64 para preview rápido
        _, buffer = cv2.imencode('.jpg', preview_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse({
            "success": True,
            "preview": f"data:image/jpeg;base64,{img_base64}"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/temp/{filename}")
async def get_temp_file(filename: str):
    """Retorna arquivo temporário"""
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Arquivo não encontrado")

@app.get("/available-options")
async def get_available_options():
    """Retorna todas as opções disponíveis para transformação"""
    return JSONResponse({
        "hair_colors": hair_transformer.get_available_colors(),
        "skin_tones": skin_transformer.get_available_tones(),
        "hair_styles": hair_transformer.get_available_styles(),
        "eye_colors": skin_transformer.get_available_eye_colors()
    })

# Handler para Vercel
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
