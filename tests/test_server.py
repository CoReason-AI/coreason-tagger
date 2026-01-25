# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from coreason_tagger.config import settings
from coreason_tagger.server import app
from fastapi.testclient import TestClient


def test_health_check() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ready", "model": settings.NER_MODEL_NAME}


def test_tag_endpoint() -> None:
    with TestClient(app) as client:
        payload = {"text": "Patient has severe headache.", "labels": ["Symptom"], "strategy": "SPEED_GLINER"}
        response = client.post("/tag", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
