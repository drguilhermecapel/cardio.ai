#!/usr/bin/env python3
"""
ECG ANALYZER AVANÇADO - PARTE 7: SISTEMA COMPLETO INTEGRADO
Script principal que integra todos os componentes em um sistema unificado
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import yaml

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar todos os componentes
try:
    from ecg_analyzer_part1 import AdvancedECGPreprocessor
    from ecg_analyzer_part2 import AdvancedWaveDelineator
    from ec