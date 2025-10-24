from typing import List
from pydantic import BaseModel

class Annotation(BaseModel):
    object_recognition: int
    text_language: int

class Dataset(BaseModel):
    category: int
    identifier: str
    label_path: str
    name: str
    src_path: str
    type: int

class Images(BaseModel):
    acquistion_location: str
    application_field: str
    background: int
    data_captured: str
    height: int
    identifier: str
    media_type: int
    pen_color: str
    pen_type: int
    type: str
    width: int
    writer_age: int
    writer_sex: int
    written_content: int

class BBox(BaseModel):
    data: str
    id: int
    x: List[int]
    y: List[int]

class Label(BaseModel):
    Annotation: Annotation
    Dataset: Dataset
    Images: Images
    bbox: List[BBox]

